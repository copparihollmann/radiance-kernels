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


if __name__ == "__main__":
    unittest.main()
