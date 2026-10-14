"""Run generated decoder C++ against the independent NumPy graph values."""

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from compile_decoder import generate, verify_native


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

    def test_qwen3_checkpoint_mode_requires_weight_mapping(self):
        with self.assertRaisesRegex(ValueError, "Qwen3 checkpoint weight-image"):
            generate("qwen3_0_6b", 1, 1, 0, "teacher_forced",
                     Path("unused"), device_check_all_stages=True,
                     stages_per_object=16, checkpoint_dir=Path("unused"),
                     weight_image=Path("unused"))

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
