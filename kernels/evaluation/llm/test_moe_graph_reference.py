"""Check a reduced Qwen3 MoE decoder through prefill and dependent decode."""

from __future__ import annotations

import unittest

import numpy as np

import moe_graph_reference
import stitch


class Qwen3MoEGraphTest(unittest.TestCase):
    @staticmethod
    def spec() -> dict:
        spec = stitch.model_specs()["qwen3_30b_a3b"].copy()
        spec.update(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                    vocab_size=64, num_experts=16, num_experts_per_tok=8,
                    moe_intermediate_size=16)
        return spec

    def graph(self, prefill: int, decode: int, generation: str) -> stitch.Graph:
        return stitch.build("qwen3_30b_a3b", prefill=prefill,
                            decode_steps=decode, generation=generation,
                            specs={"qwen3_30b_a3b": self.spec()})

    def test_teacher_forced_cache_matches_full_causal_pass(self) -> None:
        incremental = self.graph(3, 2, "teacher_forced")
        sequence = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        inputs = {"prefill.token_ids": sequence[:3][None],
                  "decode0.token_ids": sequence[3:4][None],
                  "decode1.token_ids": sequence[4:5][None]}
        stepped = moe_graph_reference.execute(incremental, inputs)
        full = self.graph(5, 0, "teacher_forced")
        all_at_once = moe_graph_reference.execute(
            full, {"prefill.token_ids": sequence[None]})
        np.testing.assert_allclose(stepped[incremental.outputs[-1]],
                                   all_at_once[full.outputs[0]][:, -1:, :],
                                   rtol=1e-4, atol=1e-5)
        changed = dict(inputs)
        changed["decode1.token_ids"] = np.array([[6]], dtype=np.int32)
        other = moe_graph_reference.execute(incremental, changed)
        self.assertGreater(float(np.max(np.abs(
            other[incremental.outputs[-1]] - stepped[incremental.outputs[-1]]))),
            1e-6)

    def test_greedy_decode_follows_its_own_logits(self) -> None:
        greedy = self.graph(3, 2, "greedy")
        keep = ("decode0.token_ids", "decode1.token_ids")
        result = moe_graph_reference.execute(
            greedy, {"prefill.token_ids": np.array([[1, 2, 3]], dtype=np.int32)},
            retain=keep)
        first = int(result[keep[0]][0, 0])
        second = int(result[keep[1]][0, 0])
        self.assertEqual(first, int(np.argmax(result[greedy.outputs[0]][0, -1])))
        self.assertEqual(second, int(np.argmax(result[greedy.outputs[1]][0, -1])))
        full = self.graph(5, 0, "teacher_forced")
        all_at_once = moe_graph_reference.execute(
            full, {"prefill.token_ids": np.array([[1, 2, 3, first, second]],
                                                   dtype=np.int32)})
        np.testing.assert_allclose(result[greedy.outputs[-1]],
                                   all_at_once[full.outputs[0]][:, -1:, :],
                                   rtol=1e-4, atol=1e-5)

    def test_expert_weight_limit_precedes_allocation(self) -> None:
        graph = stitch.Graph("qwen3_30b_a3b", self.spec())
        grouped = graph.input("grouped", (1, 1, 1, 1))
        indices = graph.input("indices", (1, 1, 1), "int32")
        graph.add("projected", "expert_gate_up", (grouped, indices),
                  (1, 1, 1, 2), attrs={"parameters": {
                      "layers.0.experts.gate_up_proj": [1_000_000, 2, 1]}})
        graph.outputs.append("projected")
        with self.assertRaisesRegex(ValueError, "weight exceeds numerical control limit"):
            moe_graph_reference.execute(
                graph, {grouped: np.ones((1, 1, 1, 1), dtype=np.float32),
                        indices: np.zeros((1, 1, 1), dtype=np.int32)},
                max_tensor_elements=100)


if __name__ == "__main__":
    unittest.main()
