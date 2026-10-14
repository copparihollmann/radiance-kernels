"""Check the decomposed Qwen3 MoE graph stages against a fused calculation."""

from __future__ import annotations

import unittest

import numpy as np

import moe_ops
import stitch


def fused_control(hidden: np.ndarray, weights: dict[str, np.ndarray]
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the fused routing/FFN equation without following graph stages."""
    logits = hidden @ weights["gate.weight"].T
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=-1, keepdims=True)
    selected = np.argsort(-probabilities, axis=-1)[..., :8]
    scores = np.take_along_axis(probabilities, selected, axis=-1)
    scores /= scores.sum(axis=-1, keepdims=True)
    result = np.zeros_like(hidden)
    for token in range(hidden.shape[1]):
        for slot, expert in enumerate(selected[0, token]):
            gate_up = weights["experts.gate_up_proj"][expert] @ hidden[0, token]
            gate, up = np.split(gate_up, 2)
            activated = (gate / (1 + np.exp(-gate))) * up
            result[0, token] += (scores[0, token, slot]
                                 * (weights["experts.down_proj"][expert] @ activated))
    return result, selected, scores


class SparseExpertStagesTest(unittest.TestCase):
    def test_graph_stage_sequence_matches_fused_equation(self) -> None:
        spec = stitch.model_specs()["qwen3_30b_a3b"].copy()
        spec.update(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                    vocab_size=64, num_experts=16, num_experts_per_tok=8,
                    moe_intermediate_size=16)
        graph = stitch.build("qwen3_30b_a3b", prefill=7, decode_steps=0,
                             specs={"qwen3_30b_a3b": spec})
        rng = np.random.default_rng(29)
        hidden = rng.normal(size=(1, 7, 32)).astype(np.float32) * 0.4
        weights = {
            "gate.weight": rng.normal(size=(16, 32)).astype(np.float32) * 0.15,
            "experts.gate_up_proj": rng.normal(size=(16, 32, 32)).astype(np.float32) * 0.15,
            "experts.down_proj": rng.normal(size=(16, 32, 16)).astype(np.float32) * 0.15,
        }
        expected_output, expected_ids, expected_scores = fused_control(hidden, weights)
        values = {
            "prefill.layer00.ffn_norm": hidden,
            "prefill.layer00.router_logits": hidden @ weights["gate.weight"].T,
        }
        names = {
            "layers.0.experts.gate_up_proj": "experts.gate_up_proj",
            "layers.0.experts.down_proj": "experts.down_proj",
        }
        def load(name: str, shape: tuple[int, ...]) -> np.ndarray:
            value = weights[names[name]]
            self.assertEqual(value.shape, shape)
            return value
        stages = [stage for stage in graph.stages if stage["op"] in moe_ops.OPS]
        self.assertEqual([stage["op"] for stage in stages], [
            "softmax", "topk_indices", "topk_scores", "expert_group",
            "expert_gate_up", "expert_swiglu", "expert_down",
            "expert_weighted_reduce"])
        for stage in stages:
            args = [values[name] for name in stage["reads"]]
            values[stage["writes"]] = moe_ops.execute_stage(stage, args, load)
        prefix = "prefill.layer00."
        np.testing.assert_array_equal(values[prefix + "topk_indices"], expected_ids)
        np.testing.assert_allclose(values[prefix + "topk_scores"], expected_scores,
                                   rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(values[prefix + "expert_reduce"], expected_output,
                                   rtol=2e-5, atol=2e-6)
        raw_scores = np.take_along_axis(values[prefix + "router_probs"],
                                        expected_ids, axis=-1)
        wrong_output = np.sum(
            values[prefix + "expert_down"] * raw_scores[..., None],
            axis=-2)
        self.assertGreater(float(np.max(np.abs(wrong_output - expected_output))),
                           1e-4)


if __name__ == "__main__":
    unittest.main()
