"""Checks for model topology, cache handoffs, and numerical controls."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference import check_small, execute, reduced_spec
from stitch import build, model_specs


class ModelStitchTest(unittest.TestCase):
    def test_full_decoder_graph_keeps_logical_head_and_cache_axes(self):
        for model in ("tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma_2_2b_it"):
            with self.subTest(model=model):
                spec = model_specs()[model]
                graph = build(model, batch=1, prefill=3, decode_steps=1)
                h = spec["hidden_size"]
                qh = spec["num_attention_heads"]
                kvh = spec["num_key_value_heads"]
                d = spec["head_dim"]
                f = spec["intermediate_size"]
                expected = {
                    "prefill.embedding": [1, 3, h],
                    "prefill.layer00.q_proj": [1, 3, qh, d],
                    "prefill.layer00.k_proj": [1, 3, kvh, d],
                    "prefill.layer00.v_proj": [1, 3, kvh, d],
                    "prefill.layer00.q_rope": [1, 3, qh, d],
                    "prefill.layer00.k_cache": [1, 3, kvh, d],
                    "prefill.layer00.attention": [1, 3, qh * d],
                    "prefill.layer00.o_proj": [1, 3, h],
                    "prefill.layer00.gate_proj": [1, 3, f],
                    "decode0.layer00.k_cache": [1, 4, kvh, d],
                    "decode0.layer00.v_cache": [1, 4, kvh, d],
                }
                for name, shape in expected.items():
                    self.assertEqual(graph.tensors[name]["shape"], shape, name)
                attention = next(stage for stage in graph.stages
                                 if stage["id"] == "decode0.layer00.attention")
                self.assertEqual(attention["attrs"]["query_start"], 3)
                self.assertEqual(attention["attrs"]["q_heads"], qh)
                self.assertEqual(attention["attrs"]["kv_heads"], kvh)
                self.assertEqual(attention["attrs"]["head_dim"], d)

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
        self.assertEqual(ops.count("embedding_scale"), 4)
        self.assertEqual(ops.count("masked_gqa"), 16 + 10 * 16)
        self.assertEqual(ops.count("gated_activation"), 16 + 10 * 16)
        self.assertEqual(ops.count("euler_step"), 10)
        self.assertEqual(graph.tensors[graph.outputs[0]]["shape"], [1, 50, 32])
        self.assertEqual(graph.tensors["vlm.layer00.k_cache"]["shape"], [1, 241, 5, 64])
        self.assertEqual(graph.tensors["camera0.scale"]["shape"], [1, 64, 960])
        self.assertEqual(graph.tensors["prefix.attention_mask"]["shape"], [1, 241, 241])
        self.assertEqual(graph.tensors["denoise0.attention_mask"]["shape"], [1, 50, 291])
        self.assertEqual(graph.tensors["denoise0.position_ids"]["shape"], [1, 50])
        stages = {stage["id"]: stage for stage in graph.stages}
        self.assertEqual(stages["denoise0.euler"]["attrs"]["step_size"], -0.1)
        self.assertEqual(stages["vlm.layer00.k_cache"]["op"], "rope_with_positions")
        self.assertIn("vlm.layer00.k_proj", stages["vlm.layer00.k_cache"]["reads"])
        self.assertIn("vlm.layer00.k_cache", stages["denoise0.expert00.k_append"]["reads"])
        self.assertIn("vlm.layer00.k_cache",
                      stages["denoise1.expert00.k_append"]["reads"])
        self.assertIn("vlm.layer01.k_cache",
                      stages["denoise1.expert01.k_cross_proj"]["reads"])
        self.assertIn("denoise1.expert01.cross_attention_mask",
                      stages["denoise1.expert01.attention"]["reads"])
        self.assertIn("denoise1.local_position_ids",
                      stages["denoise1.expert01.q_rope"]["reads"])
        self.assertNotIn("cache_crop", ops)
        self.assertFalse(graph.report()["device_execution"])
        self.assertFalse(graph.report()["network_execution_validated"])

    def test_smolvla_rejects_unsupported_shape_changes(self):
        original = model_specs()["smolvla_base"]
        for change in ({"image_size": 500}, {"num_expert_layers": 8}):
            with self.subTest(change=change):
                spec = {**original, **change}
                with self.assertRaises(ValueError):
                    build("smolvla_base", specs={"smolvla_base": spec})

    def test_smolvla_schedule_preserves_nested_loops_and_action_queue(self):
        graph = build("smolvla_base")
        schedule = graph.report()["execution_schedule"]
        self.assertEqual(schedule["entrypoint"], "sample_actions")
        queue = schedule["policy_select_action"]
        self.assertEqual(queue["queued_actions_per_refill"], 50)
        self.assertEqual(queue["returned_action_dim"], 6)
        self.assertEqual(queue["model_calls_per_refill"], 1)

        prefix = schedule["prefix_once_per_refill"]
        self.assertEqual(len(prefix["camera_branches"]), 3)
        self.assertEqual(len(prefix["vlm_layers"]), 16)
        self.assertEqual(prefix["vlm_layers"][0], "vlm.layer00.ffn_residual")
        self.assertEqual(len(prefix["read_only_cache"]), 32)
        self.assertEqual(prefix["vision_encoder_internal_layers"], "opaque")

        loop = schedule["denoise_loop"]
        self.assertEqual(len(loop["iterations"]), 10)
        self.assertEqual(loop["expert_attention_mode"], ["self", "cross"] * 8)
        self.assertEqual(loop["step_size"], -0.1)
        stage_ids = {stage["id"]: stage for stage in graph.stages}
        scheduled = list(prefix["stages"])
        carried_action = "action.noise"
        for step, iteration in enumerate(loop["iterations"]):
            self.assertEqual(iteration["step"], step)
            self.assertAlmostEqual(iteration["time"], 1.0 - step / 10)
            self.assertEqual(iteration["action_input"], carried_action)
            self.assertEqual(len(iteration["expert_layers"]), 16)
            self.assertEqual(iteration["stages"][0], f"denoise{step}.timestep")
            self.assertEqual(iteration["stages"][-1], f"denoise{step}.euler")
            timestep = stage_ids[f"denoise{step}.timestep"]
            self.assertEqual(timestep["op"], "timestep_constant")
            self.assertAlmostEqual(timestep["attrs"]["value"], iteration["time"])
            self.assertEqual(stage_ids[f"denoise{step}.action_in_linear"]["reads"],
                             [carried_action])
            self.assertEqual(stage_ids[f"denoise{step}.time_embedding"]["reads"],
                             [timestep["id"]])
            self.assertEqual(stage_ids[f"denoise{step}.action_time_concat"]["reads"],
                             [f"denoise{step}.action_in",
                              f"denoise{step}.time_broadcast"])
            self.assertEqual(stage_ids[f"denoise{step}.action_embed"]["attrs"]["parameter"],
                             "model.action_time_mlp_out.bias")
            self.assertEqual(stage_ids[f"denoise{step}.euler"]["reads"],
                             [carried_action, f"denoise{step}.action_out"])
            for cache in prefix["read_only_cache"]:
                self.assertIn(cache, graph.tensors)
                self.assertNotIn(cache, iteration["stages"])
            scheduled.extend(iteration["stages"])
            carried_action = iteration["action_output"]
        self.assertEqual(scheduled, [stage["id"] for stage in graph.stages])
        self.assertEqual(graph.outputs, [carried_action])

    def test_smolvla_rejects_zero_length_denoise_loop(self):
        spec = {**model_specs()["smolvla_base"], "num_denoise_steps": 0}
        with self.assertRaises(ValueError):
            build("smolvla_base", specs={"smolvla_base": spec})

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
