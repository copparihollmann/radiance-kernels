"""Checks for model topology, cache handoffs, and numerical controls."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference import check_small, execute, reduced_spec
from stitch import build, model_specs


class ModelStitchTest(unittest.TestCase):
    def test_decoder_graphs_have_pinned_layers_and_complete_cache_chain(self):
        for model in ("tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma_2_2b_it"):
            with self.subTest(model=model):
                spec = model_specs()[model]
                graph = build(model, batch=1, prefill=4, decode_steps=2)
                self.assertEqual(len(graph.outputs), 3)
                self.assertFalse(graph.report()["device_execution"])
                for layer in range(spec["num_hidden_layers"]):
                    cache = graph.tensors[f"decode1.layer{layer:02d}.k_cache"]
                    self.assertEqual(cache["shape"],
                                     [1, 6, spec["num_key_value_heads"], spec["head_dim"]])
                attention = [stage for stage in graph.stages
                             if stage["op"] == "causal_gqa"]
                self.assertEqual(len(attention), spec["num_hidden_layers"] * 3)
                self.assertTrue(all(len(stage["reads"]) == 3 for stage in attention))

    def test_model_specific_attention_and_bias(self):
        deepseek = build("deepseek_r1_distill_qwen_1_5b", decode_steps=0)
        self.assertEqual(sum(stage["op"] == "bias_add" for stage in deepseek.stages),
                         28 * 3)
        self.assertTrue(all(stage["attrs"]["window"] is None
                            for stage in deepseek.stages if stage["op"] == "causal_gqa"))
        gemma = build("gemma_2_2b_it", decode_steps=0)
        windows = [stage["attrs"]["window"] for stage in gemma.stages
                   if stage["op"] == "causal_gqa"]
        self.assertEqual(windows, [4096 if i % 2 == 0 else None for i in range(26)])
        self.assertEqual(gemma.stages[-1]["op"], "softcap")

    def test_smolvla_covers_vision_vlm_and_action_denoising(self):
        graph = build("smolvla_base")
        ops = [stage["op"] for stage in graph.stages]
        self.assertEqual(ops.count("patch_embed"), 3)
        self.assertEqual(ops.count("vlm_decoder_layer"), 16)
        self.assertEqual(ops.count("expert_self_attention_layer"), 80)
        self.assertEqual(ops.count("expert_cross_attention_layer"), 80)
        self.assertEqual(ops.count("euler_step"), 10)
        self.assertEqual(graph.tensors[graph.outputs[0]]["shape"], [1, 50, 32])
        self.assertEqual(graph.tensors["vlm.layer00.k_cache"]["shape"], [1, 241, 5, 64])
        stages = {stage["id"]: stage for stage in graph.stages}
        self.assertEqual(stages["denoise0.euler"]["attrs"]["step_size"], -0.1)
        self.assertIn("vlm.layer00.k_cache", stages["denoise0.expert00.k_append"]["reads"])
        self.assertIn("denoise0.expert00.k_crop",
                      stages["denoise1.expert00.k_append"]["reads"])
        self.assertIn("vlm.layer01.k_cache", stages["denoise1.expert01"]["reads"])
        self.assertFalse(graph.report()["device_execution"])

    def test_cached_decode_equals_full_prefill_on_small_numerical_controls(self):
        for model in ("tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma_2_2b_it"):
            with self.subTest(model=model):
                self.assertTrue(check_small(model)["passed"])

    def test_decode_control_detects_changed_token(self):
        model = "tinyllama"
        spec = reduced_spec(model_specs()[model])
        graph = build(model, prefill=3, decode_steps=1, specs={model: spec})
        prefix = np.array([[1, 2, 3]], dtype=np.int32)
        first = execute(graph, {"prefill.token_ids": prefix,
                                "decode0.token_ids": np.array([[4]], dtype=np.int32)})
        second = execute(graph, {"prefill.token_ids": prefix,
                                 "decode0.token_ids": np.array([[5]], dtype=np.int32)})
        self.assertGreater(float(np.max(np.abs(first[graph.outputs[-1]] -
                                            second[graph.outputs[-1]]))), 1e-4)

    def test_greedy_logits_feed_the_next_cached_step(self):
        model = "tinyllama"
        spec = reduced_spec(model_specs()[model])
        greedy = build(model, prefill=3, decode_steps=2, specs={model: spec},
                       generation="greedy")
        token_ids = np.array([[1, 2, 3]], dtype=np.int32)
        generated = execute(greedy, {"prefill.token_ids": token_ids})
        forced = build(model, prefill=3, decode_steps=2, specs={model: spec})
        fed = execute(forced, {
            "prefill.token_ids": token_ids,
            "decode0.token_ids": generated["decode0.token_ids"],
            "decode1.token_ids": generated["decode1.token_ids"],
        })
        self.assertEqual(generated["decode0.token_ids"].dtype, np.int32)
        self.assertTrue(np.array_equal(generated[greedy.outputs[-1]],
                                       fed[forced.outputs[-1]]))


if __name__ == "__main__":
    unittest.main()
