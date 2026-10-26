"""Focused tests for the faithful BabelStream-derived Dot generator."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

RUN_PATH = Path(__file__).resolve().parent / "run.py"
RUN_SPEC = importlib.util.spec_from_file_location("stream_dot_run", RUN_PATH)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
run = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(run)


class StreamDotRunTest(unittest.TestCase):
    def test_partition_covers_each_element_once(self):
        for sms in (1, 2, 4):
            ranges = [run.partition(1048592, sms, sm) for sm in range(sms)]
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], 1048592)
            self.assertTrue(all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])))
            self.assertTrue(all(begin % 16 == 0 for begin, _ in ranges))

    def test_records_upstream_and_exact_partial_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run.prepare(256, Path(directory), sms=4,
                                 loader="logical", warps_per_core=2)
            self.assertEqual(result["reference_repository"], "UoB-HPC/BabelStream")
            self.assertEqual(result["reference_revision"], run.UPSTREAM_REVISION)
            self.assertEqual(result["output_element_count"], 4)
            self.assertEqual(len(result["expected_partial_words"]), 4)
            self.assertEqual(result["correctness"],
                             "exact-device-order-partials-host-final-scalar-and-guards")
            config = (run.GENERATED / "config.h").read_text()
            self.assertIn("#define STREAM_DOT_EXPECTED_FINAL_WORD ", config)
            self.assertEqual(
                result["expected_final_word"],
                f"{run.float_word(run.expected_final_sum(run.expected_partials(256, 4, 2))):08x}",
            )
            self.assertLess(result["relative_error_vs_scalar_golden"], 1e-4)

    def test_sm_count_changes_only_reduction_partition(self):
        one = run.expected_partials(1024, 1, 8)
        two = run.expected_partials(1024, 2, 8)
        four = run.expected_partials(1024, 4, 8)
        self.assertEqual((len(one), len(two), len(four)), (1, 2, 4))
        self.assertEqual(two[0], two[1])
        self.assertTrue(all(value == four[0] for value in four))

    def test_requires_robust_prerequisites(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires robust_end_drain"):
                run.prepare(256, Path(directory), robust_spin_until_reset=True)
            with self.assertRaisesRegex(ValueError, "requires robust_end_drain"):
                run.prepare(256, Path(directory), robust_serial_fence=True)


if __name__ == "__main__":
    unittest.main()
