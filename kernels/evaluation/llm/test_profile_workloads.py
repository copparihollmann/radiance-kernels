"""Check that shape traces retain model execution semantics."""

import unittest

from profile_workloads import live_tensor_footprint, profile
from stitch import Graph


class WorkloadProfileTest(unittest.TestCase):
    def test_live_footprint_retains_outputs_and_counts_repeated_reads_once(self):
        graph = Graph("toy", {"source": "test"})
        graph.input("a", (2,))
        graph.input("b", (3,), "int32")
        graph.add("c", "add", ("a", "b"), (2,))
        graph.add("d", "add", ("c", "c"), (2,))
        graph.outputs = ["d"]
        result = live_tensor_footprint(graph, "toy")
        self.assertEqual(result["peak_bytes"], 28)
        self.assertEqual(result["peak_stage"], "c")
        self.assertEqual(result["retained_output_bytes"], 8)

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

    def test_packed_weights_flag_an_oversized_gemma_shape(self):
        fixed = profile("gemma_2_2b_it", batch=1, prefill=2, decode_steps=2)
        medium = profile("gemma_2_2b_it", batch=1, prefill=128, decode_steps=1)
        large = profile("gemma_2_2b_it", batch=8, prefill=128, decode_steps=1)
        self.assertEqual(fixed["packed_weights"]["image_size_bytes"], 2019513344)
        self.assertEqual(fixed["packed_weights"]["used_logical_parameter_count"], 289)
        self.assertFalse(fixed["totals"]["packed_image_plus_largest_tensor_exceeds_2gib"])
        self.assertTrue(medium["totals"]["packed_image_plus_largest_tensor_exceeds_2gib"])
        self.assertTrue(large["totals"]["packed_image_plus_largest_tensor_exceeds_2gib"])
        self.assertEqual(large["totals"]["largest_mapped_tensor_bytes"], 1048576000)
        self.assertGreaterEqual(large["totals"]["peak_live_mapped_tensor_bytes"],
                                large["totals"]["largest_mapped_tensor_bytes"])


if __name__ == "__main__":
    unittest.main()
