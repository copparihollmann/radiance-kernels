"""Checks for model topology, cache handoffs, and numerical controls."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference import check_small, execute, parameter, reduced_spec
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

    def test_olmo3_preserves_projection_norms_and_local_global_order(self):
        spec = model_specs()["olmo3_7b"]
        self.assertEqual(spec["num_hidden_layers"], 32)
        self.assertEqual(spec["layer_types"],
                         ["sliding_attention", "sliding_attention",
                          "sliding_attention", "full_attention"] * 8)
        graph = build("olmo3_7b", prefill=4, decode_steps=2,
                      generation="greedy")
        stages = {stage["id"]: stage for stage in graph.stages}
        self.assertEqual(len(graph.outputs), 3)
        self.assertEqual(graph.execution_schedule["entrypoint"], "causal_generate")
        for phase, tokens, past in (("prefill", 4, 0),
                                    ("decode0", 1, 4), ("decode1", 1, 5)):
            attention = [stage for stage in graph.stages
                         if stage["id"].startswith(phase + ".layer")
                         and stage["op"] == "causal_gqa"]
            self.assertEqual([stage["attrs"]["window"] for stage in attention],
                             [4096, 4096, 4096, None] * 8)
            for layer in (0, 3, 31):
                base = f"{phase}.layer{layer:02d}"
                self.assertEqual(stages[f"{base}.q_norm"]["reads"],
                                 [f"{base}.q_proj"])
                self.assertEqual(stages[f"{base}.q_heads"]["reads"],
                                 [f"{base}.q_norm"])
                self.assertEqual(stages[f"{base}.q_rope"]["reads"],
                                 [f"{base}.q_heads"])
                self.assertEqual(graph.tensors[f"{base}.q_norm"]["shape"],
                                 [1, tokens, 4096])
                self.assertEqual(stages[f"{base}.post_attn_norm"]["reads"],
                                 [f"{base}.o_proj"])
                self.assertEqual(stages[f"{base}.gate_proj"]["reads"],
                                 [f"{base}.attn_residual"])
                self.assertEqual(stages[f"{base}.post_ffn_norm"]["reads"],
                                 [f"{base}.down_proj"])
                self.assertEqual(stages[f"{base}.attention"]["attrs"]["query_start"],
                                 past)
                self.assertEqual(graph.tensors[f"{base}.k_cache"]["shape"],
                                 [1, past + tokens, 32, 128])
        self.assertEqual(stages["decode0.layer00.k_cache"]["reads"],
                         ["prefill.layer00.k_cache", "decode0.layer00.k_rope"])
        self.assertEqual(stages["decode1.token_ids"]["reads"],
                         ["decode0.lm_head"])
        self.assertEqual(stages["prefill.lm_head"]["attrs"]["parameter"],
                         "lm_head")
        self.assertFalse(graph.report()["device_execution"])

    def test_qwen3_moe_exposes_router_experts_and_cache_order(self):
        spec = model_specs()["qwen3_30b_a3b"]
        self.assertEqual((spec["num_hidden_layers"], spec["num_experts"],
                          spec["num_experts_per_tok"]), (48, 128, 8))
        graph = build("qwen3_30b_a3b", prefill=3, decode_steps=2,
                      generation="greedy")
        stages = {stage["id"]: stage for stage in graph.stages}
        self.assertEqual(len(graph.outputs), 3)
        self.assertEqual(len([stage for stage in graph.stages
                              if stage["op"] == "expert_weighted_reduce"]), 48 * 3)
        for phase, tokens, past in (("prefill", 3, 0),
                                    ("decode0", 1, 3), ("decode1", 1, 4)):
            for layer in (0, 47):
                base = f"{phase}.layer{layer:02d}"
                self.assertEqual(graph.tensors[f"{base}.q_proj"]["shape"],
                                 [1, tokens, 32, 128])
                self.assertEqual(graph.tensors[f"{base}.k_proj"]["shape"],
                                 [1, tokens, 4, 128])
                self.assertEqual(stages[f"{base}.q_norm"]["reads"],
                                 [f"{base}.q_proj"])
                self.assertEqual(stages[f"{base}.q_rope"]["reads"],
                                 [f"{base}.q_norm"])
                self.assertEqual(stages[f"{base}.router_logits"]["reads"],
                                 [f"{base}.ffn_norm"])
                self.assertEqual(stages[f"{base}.topk_scores"]["reads"],
                                 [f"{base}.router_probs", f"{base}.topk_indices"])
                self.assertTrue(stages[f"{base}.topk_scores"]["attrs"]["renormalize"])
                self.assertEqual(graph.tensors[f"{base}.topk_indices"]["shape"],
                                 [1, tokens, 8])
                self.assertEqual(stages[f"{base}.expert_reduce"]["reads"],
                                 [f"{base}.expert_down", f"{base}.topk_scores"])
                self.assertEqual(stages[f"{base}.attention"]["attrs"]["query_start"],
                                 past)
                self.assertEqual(graph.tensors[f"{base}.k_cache"]["shape"],
                                 [1, past + tokens, 4, 128])
        self.assertEqual(stages["decode0.layer00.k_cache"]["reads"],
                         ["prefill.layer00.k_cache", "decode0.layer00.k_rope"])
        self.assertEqual(stages["decode1.token_ids"]["reads"],
                         ["decode0.lm_head"])
        self.assertEqual(graph.tensors["prefill.lm_head"]["shape"],
                         [1, 3, spec["vocab_size"]])
        self.assertFalse(graph.report()["device_execution"])

    def test_held_out_mistral_uses_dense_full_attention_and_gqa(self):
        spec = model_specs()["mistral_7b_v03"]
        self.assertEqual((spec["num_hidden_layers"], spec["hidden_size"],
                          spec["num_attention_heads"], spec["num_key_value_heads"]),
                         (32, 4096, 32, 8))
        graph = build("mistral_7b_v03", prefill=3, decode_steps=2,
                      generation="greedy")
        stages = {stage["id"]: stage for stage in graph.stages}
        self.assertEqual(len(graph.outputs), 3)
        self.assertEqual(len([stage for stage in graph.stages
                              if stage["op"] == "causal_gqa"]), 32 * 3)
        self.assertEqual(len([stage for stage in graph.stages
                              if stage["op"] == "bias_add"]), 0)
        for phase, tokens, past in (("prefill", 3, 0),
                                    ("decode0", 1, 3), ("decode1", 1, 4)):
            for layer in (0, 31):
                base = f"{phase}.layer{layer:02d}"
                self.assertEqual(graph.tensors[f"{base}.q_proj"]["shape"],
                                 [1, tokens, 32, 128])
                self.assertEqual(graph.tensors[f"{base}.k_proj"]["shape"],
                                 [1, tokens, 8, 128])
                self.assertEqual(stages[f"{base}.q_proj"]["reads"],
                                 [f"{base}.attn_norm"])
                self.assertEqual(stages[f"{base}.q_rope"]["reads"],
                                 [f"{base}.q_proj"])
                self.assertEqual(stages[f"{base}.attention"]["attrs"]["window"],
                                 None)
                self.assertEqual(stages[f"{base}.attention"]["attrs"]["query_start"],
                                 past)
                self.assertEqual(graph.tensors[f"{base}.k_cache"]["shape"],
                                 [1, past + tokens, 8, 128])
        self.assertEqual(stages["decode1.token_ids"]["reads"],
                         ["decode0.lm_head"])
        self.assertEqual(stages["prefill.lm_head"]["attrs"]["parameter"],
                         "lm_head")
        self.assertFalse(graph.report()["device_execution"])

    def test_nemotron_carries_distinct_mamba_and_attention_state(self):
        spec = model_specs()["nemotron3_nano_4b"]
        self.assertEqual((spec["layer_kinds"].count("mamba2"),
                          spec["layer_kinds"].count("full_attention"),
                          spec["layer_kinds"].count("mlp_only")), (21, 4, 17))
        graph = build("nemotron3_nano_4b", prefill=3, decode_steps=2,
                      generation="greedy")
        stages = {stage["id"]: stage for stage in graph.stages}
        self.assertEqual(len(graph.outputs), 3)
        self.assertEqual(graph.execution_schedule["layer_kinds"],
                         spec["layer_kinds"])
        self.assertIn("same prefill", graph.execution_schedule["reference_partition"])
        self.assertEqual(graph.report()["pinned_attention_cache_handoff"],
                         "missing_in_block_call")
        self.assertEqual(len(graph.execution_schedule["decode_loop"][0]
                             ["previous_cache"]), 25)
        for layer, kind in enumerate(spec["layer_kinds"]):
            pre = f"prefill.layer{layer:02d}"
            dec = f"decode0.layer{layer:02d}"
            self.assertEqual(stages[f"{dec}.pre_norm"]["reads"],
                             [f"decode0.embedding" if layer == 0 else
                              f"decode0.layer{layer-1:02d}.residual"])
            if kind == "mamba2":
                self.assertEqual(stages[f"{pre}.in_proj"]["attrs"]["weight_shape"],
                                 [3136, 17504])
                self.assertEqual(graph.tensors[f"{pre}.conv_state"]["shape"],
                                 [1, 9728, 4])
                self.assertEqual(graph.tensors[f"{pre}.ssm_state"]["shape"],
                                 [1, 96, 80, 128])
                self.assertEqual(graph.tensors[f"{pre}.ssm_state"]["dtype"],
                                 "fp32")
                self.assertEqual(stages[f"{dec}.conv_state"]["reads"],
                                 [f"{pre}.conv_state", f"{dec}.conv_input"])
                self.assertIn(f"{pre}.ssm_state",
                              stages[f"{dec}.selective_scan"]["reads"])
                self.assertNotIn(f"{pre}.k_cache", stages)
            elif kind == "full_attention":
                self.assertEqual(stages[f"{pre}.q_proj"]["attrs"]["weight_shape"],
                                 [3136, 40 * 128])
                self.assertEqual(stages[f"{dec}.k_cache"]["reads"],
                                 [f"{pre}.k_cache", f"{dec}.k_proj"])
                self.assertEqual(graph.tensors[f"{dec}.k_cache"]["shape"],
                                 [1, 4, 8, 128])
                self.assertFalse(stages[f"{dec}.attention"]["attrs"]["rotary"])
                self.assertNotIn(f"{pre}.ssm_state", stages)
            else:
                self.assertEqual(stages[f"{pre}.activation"]["op"],
                                 "relu_squared")
                self.assertNotIn(str(layer),
                                 graph.execution_schedule["decode_loop"][0]
                                 ["previous_cache"])
        self.assertEqual(stages["decode1.token_ids"]["reads"],
                         ["decode0.lm_head"])
        self.assertEqual(stages["prefill.last_hidden"]["shape"], [1, 1, 3136])
        self.assertEqual(stages["prefill.lm_head"]["shape"], [1, 1, 131072])
        self.assertFalse(graph.report()["device_execution"])

    def test_qwen3_order_head_norm_rope_and_tied_head(self):
        spec = model_specs()["qwen3_0_6b"]
        graph = build("qwen3_0_6b", prefill=3, decode_steps=1)
        self.assertEqual(spec["num_hidden_layers"], 28)
        self.assertEqual(spec["rope_theta"], 1_000_000)
        stages = {stage["id"]: stage for stage in graph.stages}
        for phase, tokens, past in (("prefill", 3, 0), ("decode0", 1, 3)):
            base = f"{phase}.layer00"
            self.assertEqual(stages[f"{base}.q_norm"]["reads"],
                             [f"{base}.q_proj"])
            self.assertEqual(stages[f"{base}.k_norm"]["reads"],
                             [f"{base}.k_proj"])
            self.assertEqual(stages[f"{base}.q_rope"]["reads"],
                             [f"{base}.q_norm"])
            self.assertEqual(stages[f"{base}.k_rope"]["reads"],
                             [f"{base}.k_norm"])
            self.assertEqual(stages[f"{base}.q_rope"]["attrs"],
                             {"position_start": past, "theta": 1_000_000})
            self.assertEqual(graph.tensors[f"{base}.q_norm"]["shape"],
                             [1, tokens, 16, 128])
            self.assertEqual(graph.tensors[f"{base}.k_norm"]["shape"],
                             [1, tokens, 8, 128])
            self.assertEqual(stages[f"{base}.attention"]["attrs"]["window"], None)
        self.assertEqual(stages["prefill.lm_head"]["attrs"]["parameter"],
                         stages["prefill.embedding"]["attrs"]["parameter"])
        self.assertTrue(stages["prefill.lm_head"]["attrs"]["weight_transposed"])

        reduced = reduced_spec(spec)
        small = build("qwen3_0_6b", prefill=3, decode_steps=0,
                      specs={"qwen3_0_6b": reduced})
        values = execute(small, {"prefill.token_ids":
                                 np.array([[1, 2, 3]], dtype=np.int32)})
        table = parameter("embed_tokens", (64, 32))
        self.assertTrue(np.array_equal(values["prefill.embedding"],
                                       table[[1, 2, 3]][None]))
        expected = values["prefill.final_norm"] @ table.T
        self.assertTrue(np.allclose(values["prefill.lm_head"], expected,
                                    rtol=0, atol=1e-6))
        self.assertTrue(check_small("qwen3_0_6b")["passed"])

    def test_qwen3_schedule_carries_kv_and_greedy_tokens(self):
        graph = build("qwen3_0_6b", prefill=3, decode_steps=2,
                      generation="greedy")
        schedule = graph.execution_schedule
        self.assertEqual(schedule["entrypoint"], "causal_generate")
        steps = schedule["decode_loop"]
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["greedy_source_logits"], graph.outputs[0])
        self.assertEqual(steps[1]["greedy_source_logits"], graph.outputs[1])
        self.assertEqual(steps[0]["token_input"], "decode0.token_ids")
        self.assertEqual(steps[1]["previous_cache"]["27"],
                         steps[0]["updated_cache"]["27"])
        self.assertEqual(steps[1]["updated_cache"]["27"],
                         ["decode1.layer27.k_cache", "decode1.layer27.v_cache"])
        scheduled = (schedule["prefill_stages"] +
                     [name for step in steps for name in step["stages"]])
        self.assertEqual(scheduled, [stage["id"] for stage in graph.stages])

    def test_smolvla_covers_vision_vlm_and_action_denoising(self):
        graph = build("smolvla_base")
        ops = [stage["op"] for stage in graph.stages]
        self.assertEqual(ops.count("patch_embed"), 3)
        self.assertEqual(ops.count("bidirectional_mha"), 3 * 12)
        self.assertEqual(ops.count("layernorm"), 3 * (2 * 12 + 1))
        self.assertEqual(ops.count("pixel_shuffle"), 3)
        self.assertEqual(ops.count("embedding_scale"), 4)
        self.assertEqual(ops.count("masked_gqa"), 16 + 10 * 16)
        self.assertEqual(ops.count("gated_activation"), 16 + 10 * 16)
        self.assertEqual(ops.count("euler_step"), 10)
        self.assertEqual(graph.tensors[graph.outputs[0]]["shape"], [1, 50, 32])
        self.assertEqual(graph.tensors["vlm.layer00.k_cache"]["shape"], [1, 241, 5, 64])
        self.assertEqual(graph.tensors["camera0.scale"]["shape"], [1, 64, 960])
        self.assertEqual(graph.tensors["camera0.connector_shuffle"]["shape"],
                         [1, 64, 16 * 768])
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
        self.assertTrue(all(len(branch["vision_layers"]) == 12
                            for branch in prefix["camera_branches"]))
        self.assertEqual(len(prefix["vlm_layers"]), 16)
        self.assertEqual(prefix["vlm_layers"][0], "vlm.layer00.ffn_residual")
        self.assertEqual(len(prefix["read_only_cache"]), 32)
        self.assertEqual(prefix["vision_encoder_layers_per_camera"], 12)

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

    def test_full_decoder_depth_preserves_cached_decode(self):
        for model, layers in (("tinyllama", 22),
                              ("deepseek_r1_distill_qwen_1_5b", 28),
                              ("gemma_2_2b_it", 26)):
            with self.subTest(model=model):
                result = check_small(model, layers)
                self.assertEqual(result["layers"], layers)
                self.assertLess(result["maximum_absolute_error"], 1e-5)
                self.assertGreater(result["changed_decode_token_max_abs_delta"], 0)

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
