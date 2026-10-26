"""Focused tests for the faithful Vortex/Rodinia Nearn generator."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

RUN_PATH = Path(__file__).resolve().parent / "run.py"
RUN_SPEC = importlib.util.spec_from_file_location("nearn_run", RUN_PATH)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
run = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(run)


class NearnRunTest(unittest.TestCase):
    def test_reference_generation_is_reproducible(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run.prepare(32, Path(first), sms=4, loader="logical")
            b = run.prepare(32, Path(second), sms=4, loader="logical")
            self.assertEqual(a["input_sha256"], b["input_sha256"])
            self.assertEqual(a["expected_sha256"], b["expected_sha256"])
            self.assertEqual(a["expected_digest"], b["expected_digest"])
            self.assertEqual(a["output_element_count"], 32)
            self.assertEqual(a["reference_revision"], run.REFERENCE_REVISION)
            self.assertEqual(a["reference_kernel"], run.REFERENCE_KERNEL)
            self.assertEqual(a["reference_kernel_sha256"],
                             run.REFERENCE_KERNEL_SHA256)
            self.assertEqual(len(a["expected_top_indices"]), run.TOP_K)
            self.assertIn("#define NIGHTLY_CLUSTERS 4",
                          run.GENERATED.joinpath("config.h").read_text())

    def test_reference_matches_explicit_distance(self):
        query_lat = run.f32(3.0)
        query_lng = run.f32(4.0)
        output = run.reference_output(query_lat, query_lng, [(0.0, 0.0)])
        self.assertEqual(run.float_word(output[0]), run.float_word(5.0))

    def test_reference_matches_original_4096_record_import(self):
        query_lat, query_lng, locations = run.reference_dataset(4096)
        output = run.reference_output(query_lat, query_lng, locations)
        words = [run.float_word(value) for value in output]
        self.assertEqual(run.digest(words), 0xEAE785DADD0EC6A7)
        self.assertEqual(
            sorted(range(len(output)), key=lambda i: (output[i], i))[:run.TOP_K],
            [2414, 2114, 2609, 1950, 828],
        )

    def test_robust_completion_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run.prepare(
                8, Path(directory), robust_end_drain=True,
                robust_spin_until_reset=True, robust_serial_fence=True,
                robust_post_spin=20000,
            )
            self.assertTrue(result["robust_spin_until_reset"])
            self.assertTrue(result["robust_serial_fence"])
            with self.assertRaisesRegex(ValueError, "requires robust_end_drain"):
                run.prepare(8, Path(directory), robust_spin_until_reset=True)

    def test_host_checker_keeps_full_validation_in_one_record_pass(self):
        host = RUN_PATH.with_name("host.cpp").read_text()
        self.assertEqual(
            host.count("for (uint32_t i = 0; i < NEARN_RECORDS; ++i)"), 1)
        self.assertIn("distance < selected_distance[rank]", host)
        self.assertIn("selected[rank] != expected_top[rank]", host)


if __name__ == "__main__":
    unittest.main()
