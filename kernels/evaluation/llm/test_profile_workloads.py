"""Check that shape traces retain model execution semantics."""

import unittest

from profile_workloads import profile


class WorkloadProfileTest(unittest.TestCase):
    def test_decoder_prefill_and_cached_decode_remain_distinct(self):
        result = profile("tinyllama", batch=1, prefill=2, decode_steps=2)
        self.assertEqual(result["stage_count"], 1133)
        self.assertEqual(list(result["phase_summary"]),
                         ["prefill", "decode0", "decode1"])
        self.assertEqual(result["stages"][-1]["id"], "decode1.lm_head")
        self.assertGreater(result["phase_summary"]["prefill"]["linear_macs"], 0)
        self.assertGreater(result["phase_summary"]["decode1"]["linear_macs"], 0)

    def test_batch_scales_linear_work_without_changing_stage_order(self):
        first = profile("tinyllama", batch=1, prefill=2, decode_steps=0)
        second = profile("tinyllama", batch=2, prefill=2, decode_steps=0)
        self.assertEqual([item["id"] for item in first["stages"]],
                         [item["id"] for item in second["stages"]])
        self.assertEqual(second["totals"]["linear_macs"],
                         2 * first["totals"]["linear_macs"])

    def test_smolvla_prefix_and_ten_denoising_iterations_are_preserved(self):
        result = profile("smolvla_base", batch=1, prefill=1, decode_steps=0)
        schedule = result["execution_schedule"]
        self.assertEqual(result["stage_count"], 3673)
        self.assertEqual(len(schedule["prefix_once_per_refill"]["stages"]), 913)
        self.assertEqual(len(schedule["denoise_loop"]["iterations"]), 10)
        self.assertEqual(schedule["policy_select_action"]["queued_actions_per_refill"], 50)
        self.assertEqual(result["stages"][-1]["id"], "denoise9.euler")


if __name__ == "__main__":
    unittest.main()
