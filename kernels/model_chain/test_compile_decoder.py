"""Run generated decoder C++ against the independent NumPy graph values."""

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from compile_decoder import emit_stage, generate, verify_native
import reference
import stitch


class DecoderCompilerTest(unittest.TestCase):
    def test_decoder_families_and_greedy_cache_handoff(self):
        cases = [
            ("tinyllama", 1, "teacher_forced"),
            ("deepseek_r1_distill_qwen_1_5b", 1, "teacher_forced"),
            ("gemma_2_2b_it", 1, "teacher_forced"),
            ("qwen3_0_6b", 2, "teacher_forced"),
            ("qwen3_0_6b", 2, "greedy"),
            ("tinyllama", 2, "greedy"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for model, decode, generation in cases:
                with self.subTest(model=model, decode=decode, generation=generation):
                    root = Path(temporary) / f"{model}-{decode}-{generation}"
                    target = generate(model, 1, 3, decode, generation, root,
                                      device_check_all_stages=True)
                    result = verify_native(target)
                    manifest = json.loads((target / "manifest.json").read_text())
                    self.assertIn("errors=0", result)
                    self.assertLess(float(result.split("max_abs_error=")[1]), 1e-4)
                    self.assertEqual(manifest["native_verified_float_stages"],
                                     manifest["stages"] -
                                     (decode if generation == "greedy" else 0))
                    self.assertEqual(manifest["native_verified_integer_stages"],
                                     decode if generation == "greedy" else 0)
                    self.assertEqual(len(manifest["verified_tensors"]),
                                     manifest["stages"])
                    self.assertEqual(manifest["checkpoint_weights"], False)

    def test_intermediate_reference_perturbation_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = generate("tinyllama", 1, 1, 0, "teacher_forced",
                              Path(temporary), device_check_all_stages=True)
            self.assertIn("errors=0", verify_native(target))
            source = target / "native.cpp"
            changed, replacements = re.subn(
                r"(alignas\(64\) float v_gold_prefill_embedding\[\d+\] = \{)[^,]+",
                r"\g<1>999.0f", source.read_text(), count=1)
            self.assertEqual(replacements, 1)
            source.write_text(changed)
            with self.assertRaises(subprocess.CalledProcessError):
                verify_native(target)

    def test_explicit_teacher_forced_token_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = generate("tinyllama", 1, 3, 2, "teacher_forced", root,
                              device_check_all_stages=True,
                              token_ids=[1, 2, 3, 4, 5])
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertIn("errors=0", verify_native(target))
            self.assertEqual(manifest["input_token_ids"], {
                "prefill.token_ids": [1, 2, 3],
                "decode0.token_ids": [4],
                "decode1.token_ids": [5],
            })
            self.assertTrue(manifest["token_sequence_explicit"])
            with self.assertRaises(ValueError):
                generate("tinyllama", 1, 3, 2, "teacher_forced", root,
                         token_ids=[1, 2, 3, 4])
            with self.assertRaises(ValueError):
                generate("tinyllama", 1, 3, 2, "greedy", root,
                         token_ids=[1, 2, 3, 4, 5])

    def test_explicit_prefill_tokens_feed_greedy_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = generate("tinyllama", 1, 3, 2, "greedy",
                              Path(temporary), device_check_all_stages=True,
                              token_ids=[1, 2, 3])
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertEqual(manifest["input_token_ids"],
                             {"prefill.token_ids": [1, 2, 3]})
            self.assertEqual(manifest["native_verified_integer_stages"], 2)
            self.assertIn("errors=0", verify_native(target))

    def test_qwen3_fp16_tied_head_emits_transposed_embedding_read(self):
        spec = reference.reduced_spec(stitch.model_specs()["qwen3_0_6b"])
        graph = stitch.build("qwen3_0_6b", prefill=1, decode_steps=0,
                             specs={"qwen3_0_6b": spec})
        head = next(stage for stage in graph.stages
                    if stage["id"] == "prefill.lm_head")
        projection = next(stage for stage in graph.stages
                          if stage["id"] == "prefill.layer00.q_proj")
        self.assertIn("linear_f16_tied(", emit_stage(0, head, graph, "weight", "fp16"))
        self.assertIn("linear_f16(", emit_stage(0, projection, graph, "weight", "fp16"))

    def test_qwen3_last_prefill_logit_maps_final_hidden_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = generate("qwen3_0_6b", 2, 3, 2, "greedy", root / "last",
                              device_check_all_stages=True, stages_per_object=20,
                              token_ids=[1, 2, 3], prefill_logits="last")
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertEqual(manifest["prefill_logits_policy"], "last")
            self.assertEqual(manifest["stages"], 126)
            self.assertIn("errors=0", verify_native(target))
            control = generate("qwen3_0_6b", 2, 3, 2, "greedy", root / "all",
                               device_check_all_stages=True, stages_per_object=20,
                               token_ids=[1, 2, 3])
            self.assertEqual(json.loads((control / "manifest.json").read_text())
                             ["stages"], 125)
            header = (Path(__file__).parent / "pipeline_select_last.hpp").read_text()
            self.assertEqual(header.count("tokens - 1"), 1)
            (target / "pipeline_select_last.hpp").write_text(
                header.replace("tokens - 1", "0"))
            with self.assertRaises(subprocess.CalledProcessError):
                verify_native(target)

    def test_qwen3_moe_sparse_stages_join_prefill_and_greedy_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = generate("qwen3_30b_a3b", 2, 3, 2, "greedy",
                              Path(temporary), device_check_all_stages=True,
                              token_ids=[1, 2, 3], stages_per_object=20)
            manifest = json.loads((target / "manifest.json").read_text())
            result = verify_native(target)
            self.assertIn("errors=0", result)
            self.assertLess(float(result.split("max_abs_error=")[1]), 1e-4)
            self.assertEqual(manifest["stages"], 155)
            self.assertEqual(manifest["native_verified_float_stages"] +
                             manifest["native_verified_integer_stages"],
                             manifest["stages"])
            self.assertEqual(manifest["native_verified_integer_stages"], 8)
            self.assertIn('"pipeline_moe.hpp"',
                          (target / "stage_chunk_000.cpp").read_text())
            with self.assertRaisesRegex(ValueError, "weight-image export is not implemented"):
                generate("qwen3_30b_a3b", 2, 3, 2, "greedy",
                         Path(temporary), checkpoint_dir=Path(temporary),
                         weight_image=Path(temporary) / "missing.json")
            header = (Path(__file__).parent / "pipeline_moe.hpp").read_text()
            self.assertEqual(header.count("if (renormalize)"), 1)
            (target / "pipeline_moe.hpp").write_text(
                header.replace("if (renormalize)", "if (false && renormalize)"))
            with self.assertRaises(subprocess.CalledProcessError):
                verify_native(target)

    def test_two_cluster_buffers_and_second_output_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = generate("tinyllama", 1, 2, 1, "greedy",
                              Path(temporary), batch_clusters=2)
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertEqual(manifest["batch_clusters"], 2)
            self.assertIn("errors=0", verify_native(target))
            source = target / "native.cpp"
            text = source.read_text()
            self.assertIn("v_decode0_token_ids[i + 16u]", text)
            needle = "  uint32_t errors = 0; float maximum = 0.0f;"
            self.assertEqual(text.count(needle), 1)
            source.write_text(text.replace(
                needle, "  v_decode0_token_ids[16] = -1;\n" + needle))
            with self.assertRaises(subprocess.CalledProcessError):
                verify_native(target)


if __name__ == "__main__":
    unittest.main()
