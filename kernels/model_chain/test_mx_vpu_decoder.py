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
                                     "54492d510b9a2655ca886bad5675091af414e986")
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

    def test_two_replica_pure_helper_calls_keep_complete_graph_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            target = generate("tinyllama", 1, 2, 1, "greedy", Path(directory),
                              device_check_all_stages=True, batch_clusters=2,
                              execution_backend="rakanic_mx_vpu", mx_linear_policy="q_proj",
                              fallback_call_abi=True)
            self.assertIn("errors=0", verify_native(target))
            text = (target/"native.cpp").read_text()
            self.assertIn("rk_fallback_call", text)
            self.assertNotIn("(uintptr_t)(v_prefill", text)
            self.assertIn("v_decode0_token_ids[i + 16u]", text)
            needle = "  uint32_t errors = 0; float maximum = 0.0f;"
            self.assertEqual(text.count(needle), 1)
            (target/"native.cpp").write_text(text.replace(needle, "  v_decode0_token_ids[16] = -1;\n"+needle))
            with self.assertRaises(subprocess.CalledProcessError):
                verify_native(target)


if __name__ == "__main__":
    unittest.main()
