"""Focused correctness and ownership tests for the Radiance sfilter port."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("radiance_sfilter", ROOT / "sfilter.py")
assert SPEC is not None and SPEC.loader is not None
sfilter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sfilter)


class SFilterTest(unittest.TestCase):
    def test_current_48x48_dataset_is_bit_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            manifests = [
                sfilter.prepare(48, Path(directory) / f"{sms}sm", sms=sms)
                for sms in (1, 2, 4)
            ]
        for manifest in manifests:
            self.assertTrue(manifest["dataset_contract_preserved"])
            self.assertEqual(manifest["expected_digest"], "ccf06e93998e0682")
            self.assertEqual(manifest["input_sha256"], manifests[0]["input_sha256"])
            self.assertEqual(manifest["expected_sha256"], manifests[0]["expected_sha256"])
            self.assertEqual(manifest["max_ulp"], 4)

    def test_ownership_is_exact_disjoint_and_cache_line_safe(self):
        size = 48
        expected = {
            x + y * size for y in range(1, size - 1)
            for x in range(1, size - 1)
        }
        for sms in (1, 2, 4):
            owners = sfilter.validate_ownership(size, sms)
            claimed = []
            for owner in owners:
                self.assertEqual(owner["begin_cache_line_offset"], 0)
                self.assertEqual(owner["end_cache_line_offset"], 0)
                for y in range(owner["row_begin_inclusive"],
                               owner["row_end_exclusive"]):
                    claimed.extend(x + y * size for x in range(1, size - 1))
            self.assertEqual(set(claimed), expected)
            self.assertEqual(len(claimed), len(expected))
            self.assertEqual(len(claimed), len(set(claimed)))

    def test_four_sm_manifest_records_auditable_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = sfilter.prepare(48, Path(directory), sms=4)
            persisted = json.loads(Path(directory, "result.json").read_text())
        self.assertEqual(manifest, persisted)
        self.assertEqual(manifest["partition_row_boundaries"], [1, 13, 25, 36, 47])
        self.assertEqual(
            [owner["physical_threadblock_id"]
             for owner in manifest["partition_ownership"]],
            [0, 1, 2, 3],
        )
        self.assertTrue(manifest["partition_complete"])
        self.assertFalse(manifest["partition_overlap"])
        self.assertTrue(manifest["partition_cache_line_safe"])
        self.assertTrue(manifest["scalar_only"])
        self.assertTrue(manifest["no_vpu"])

    def test_reference_keeps_zero_boundary_and_pinned_sum(self):
        source = [sfilter.f32(1.0)] * (16 * 16)
        output = sfilter.reference_output(source, 16)
        for y in range(16):
            for x in range(16):
                expected = 0.0 if x in (0, 15) or y in (0, 15) else 9.0
                self.assertEqual(sfilter.float_word(output[x + y * 16]),
                                 sfilter.float_word(expected))

    def test_invalid_non_aligned_size_and_sm_count_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "multiple of 16"):
                sfilter.prepare(48 + 1, Path(directory))
            with self.assertRaisesRegex(ValueError, "sms must"):
                sfilter.prepare(48, Path(directory), sms=3)

    def test_scaled_dataset_is_explicitly_derived(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = sfilter.prepare(64, Path(directory), sms=4)
        self.assertFalse(manifest["dataset_contract_preserved"])
        self.assertIn("derived-64x64", manifest["dataset_contract"])
        self.assertEqual(
            manifest["benchmark_contract"],
            "radiance-scaled-sfilter-vortex-equation-derived-input",
        )

    def test_checker_and_kernel_cover_required_contract(self):
        host = (ROOT / "host.cpp").read_text()
        kernel = (ROOT / "kernel_impl.hpp").read_text()
        self.assertIn("for (uint32_t y = 0; y < SFILTER_SIZE; ++y)", host)
        self.assertIn("SFILTER_MAX_ULP", host)
        self.assertIn("guards_intact", host)
        self.assertIn("actual_digest", host)
        self.assertIn("reference_digest_ok", host)
        self.assertIn("kSFilterRowBoundaries[threadblock_id]", kernel)
        self.assertIn("#pragma clang fp contract(off)", kernel)
        for forbidden in ("gemmini", "mxgemm", "vsetvli", "vector"):
            self.assertNotIn(forbidden, kernel.lower())


if __name__ == "__main__":
    unittest.main()
