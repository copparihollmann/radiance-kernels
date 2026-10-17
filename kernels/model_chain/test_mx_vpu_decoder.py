"""Real decoder graph handoffs through Rakanic's pinned C API model."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from compile_decoder import generate, verify_native


@unittest.skipUnless(os.environ.get("RADIANCE_LIBGEMMINI_ARCHIVE"), "needs pinned libgemmini archive")
class MxVpuDecoderChecks(unittest.TestCase):
    def test_three_decoder_graphs_keep_greedy_dependencies_and_report_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            for model in ("tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma_2_2b_it"):
                with self.subTest(model=model):
                    target = generate(model, 2, 2, 2, "greedy", Path(directory),
                                      device_check_all_stages=True, execution_backend="rakanic_mx_vpu")
                    self.assertIn("errors=0", verify_native(target))
                    data = json.loads((target/"manifest.json").read_text())
                    self.assertEqual(len(data["stage_backends"]), data["stages"])
                    self.assertEqual(len(data["format_comparison"]["greedy_tokens"]), 2)
                    self.assertGreater(data["native_model_check"]["mx_commands"], 0)
                    self.assertGreater(data["native_model_check"]["vpu_commands"], 0)
                    self.assertEqual(data["native_model_check"]["libgemmini_commit"],
                                     "d5e5e8e35d4efe77323f0d495146704d4db53d71")
                    self.assertFalse(data["native_model_check"]["full_checkpoint_execution"])
                    if model == "gemma_2_2b_it":
                        self.assertFalse(data["format_comparison"]["original_greedy_exact"])

    def test_disabling_matrix_commands_is_detected_by_graph_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            target = generate("tinyllama", 1, 1, 0, "teacher_forced", Path(directory),
                              device_check_all_stages=True, execution_backend="rakanic_mx_vpu")
            runtime = target/"mx_vpu_runtime.cpp"
            text = runtime.read_text()
            old = "  auto* d = device();"
            self.assertEqual(text.count(old), 1)
            runtime.write_text(text.replace(old, "  if (funct == 8) return;\n"+old))
            with self.assertRaises(subprocess.CalledProcessError):
                verify_native(target)

    def test_query_projection_policy_keeps_other_linears_on_muon(self):
        with tempfile.TemporaryDirectory() as directory:
            target = generate("tinyllama", 1, 1, 0, "teacher_forced", Path(directory),
                              device_check_all_stages=True, execution_backend="rakanic_mx_vpu",
                              mx_linear_policy="q_proj")
            self.assertIn("errors=0", verify_native(target))
            manifest = json.loads((target/"manifest.json").read_text())
            self.assertEqual(sum(s["backend"] == "fp8_mx" for s in manifest["stage_backends"]), 1)
            self.assertLess(manifest["mx_linear_macs"], manifest["total_linear_macs"])


if __name__ == "__main__":
    unittest.main()
