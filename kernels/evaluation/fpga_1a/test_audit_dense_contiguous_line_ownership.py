#!/usr/bin/env python3
"""Regression tests for the dense output-line ownership audit.

Everything here runs on tiny synthetic sizes, so it needs no artifact and no
device.  The first two tests are the bug: a sub-line partition granule puts two
SMs in one 64-byte line, and the detector has to say so.
"""

import importlib.util
from pathlib import Path
import unittest


MODULE = Path(__file__).with_name("audit_dense_contiguous_line_ownership.py")
SPEC = importlib.util.spec_from_file_location("dense_line_ownership", MODULE)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class SubLineGranuleTests(unittest.TestCase):
    """40 float32 elements, 16 per 64-byte line, split across 2 SMs."""

    def test_sub_line_granule_splits_a_line(self):
        """granule 4 (16 bytes) -- the unsafe variant.  Must fail."""
        report = AUDIT.audit(40, 4, (1, 2), granule=4)
        self.assertEqual(report["status"], "fail")
        by_sms = {c["sm_count"]: c for c in report["checks"]}
        self.assertEqual(by_sms[1]["status"], "pass")
        self.assertEqual(by_sms[2]["status"], "fail")
        self.assertFalse(by_sms[2]["cache_line_exclusive"])
        # Coverage is still exact -- the defect is sharing, not loss.
        self.assertTrue(by_sms[2]["coverage_exactly_once"])
        self.assertEqual(by_sms[2]["split_line_count"], 1)
        split = by_sms[2]["split_lines"][0]
        self.assertEqual(split["line"], 1)
        self.assertEqual(split["first_element"], 16)
        self.assertEqual(split["writer_sms"], [0, 1])
        self.assertEqual(split["first_byte_offset_per_sm"], {"0": 0, "1": 16})

    def test_line_granule_keeps_the_line_on_one_sm(self):
        """granule 16 (64 bytes) -- the device.  Same array, must pass."""
        report = AUDIT.audit(40, 4, (1, 2), granule=16)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["partition_cache_line_safe"])
        self.assertTrue(report["partition_complete"])
        self.assertEqual(report["partition_overlap"], 0)
        for check in report["checks"]:
            self.assertEqual(check["split_line_count"], 0)
            self.assertEqual(check["total_elements_owned"], 40)


class DeployedShapeTests(unittest.TestCase):
    def test_stream_shape_is_exclusive_and_exact(self):
        report = AUDIT.audit(1 << 20, 4, (1, 2, 4), granule=16)
        self.assertEqual(report["status"], "pass")
        for check in report["checks"]:
            self.assertTrue(check["cache_line_exclusive"])
            self.assertTrue(check["coverage_exactly_once"])
            self.assertEqual(check["total_elements_owned"], 1 << 20)

    def test_partial_last_line_is_owned_once(self):
        report = AUDIT.audit(146689, 4, (1, 2, 4), schedule="block-cyclic",
                             chunk_elements=4080)
        self.assertEqual(report["status"], "pass")
        for check in report["checks"]:
            self.assertEqual(check["unowned_elements"], 0)
            self.assertEqual(check["doubly_owned_elements"], 0)

    def test_granule_one_is_safe_only_for_some_sm_counts(self):
        """gemv partitions 256 bf16 rows with granule 1.

        It is exclusive at 1, 2 and 4 SMs because 256/n happens to be a whole
        number of 32-row lines, not because the partition enforces it.  At 3
        SMs the same code splits a line.  This is why the check is computed per
        deployed SM count rather than argued from the source.
        """
        self.assertEqual(AUDIT.audit(256, 2, (1, 2, 4), granule=1)["status"],
                         "pass")
        report = AUDIT.audit(256, 2, (3,), granule=1)
        self.assertEqual(report["status"], "fail")
        split = report["checks"][0]["split_lines"][0]
        self.assertEqual(split["line"], 2)
        self.assertEqual(split["writer_sms"], [0, 1])


class UnsafeScheduleTests(unittest.TestCase):
    def test_round_robin_shreds_every_line(self):
        report = AUDIT.audit(1024, 4, (1, 2, 4), schedule="round-robin")
        self.assertEqual(report["status"], "fail")
        by_sms = {c["sm_count"]: c for c in report["checks"]}
        self.assertEqual(by_sms[1]["status"], "pass")
        self.assertEqual(by_sms[2]["split_line_count"], 64)
        self.assertEqual(by_sms[4]["split_line_count"], 64)
        self.assertEqual(by_sms[4]["split_lines"][0]["writer_sms"], [0, 1, 2, 3])
        # Every element is still written exactly once, which is precisely why
        # a digest over the result cannot see this.
        for check in report["checks"]:
            self.assertTrue(check["coverage_exactly_once"])

    def test_per_sm_scalar_output_array_shares_one_line(self):
        """stream_dot writes `partial[block_id]` into a 16-byte array.

        One naturally aligned 32-bit store per SM, every store into the same
        64-byte line.
        """
        for sms in (2, 4):
            report = AUDIT.audit(sms, 4, (sms,), schedule="round-robin")
            self.assertEqual(report["status"], "fail")
            split = report["checks"][0]["split_lines"][0]
            self.assertEqual(split["line"], 0)
            self.assertEqual(split["writer_sms"], list(range(sms)))


class RepeatStrideTests(unittest.TestCase):
    def test_line_aligned_stride_is_accepted(self):
        report = AUDIT.audit(4096, 4, (1, 2, 4), granule=16,
                             repeat_stride=4096)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["checks"][0]["repeat_stride_line_aligned"])

    def test_misaligned_stride_is_rejected(self):
        report = AUDIT.audit(4096, 4, (1, 2), granule=16, repeat_stride=4090)
        self.assertEqual(report["status"], "fail")
        for check in report["checks"]:
            self.assertFalse(check["repeat_stride_line_aligned"])
            self.assertFalse(check["cache_line_exclusive"])
            self.assertEqual(check["split_line_count"], 0)


class SliceAndValidationTests(unittest.TestCase):
    def test_slices_cover_each_element_exactly_once(self):
        for schedule, kwargs in (("contiguous", {"granule": 16}),
                                 ("block-cyclic", {"chunk_elements": 7}),
                                 ("round-robin", {})):
            for elements in (1, 15, 16, 17, 100, 257):
                for sms in (1, 2, 3, 4, 8):
                    seen = [0] * elements
                    for start, stop, step, sm in AUDIT.slices_for(
                            elements, sms, schedule,
                            kwargs.get("granule", 1),
                            kwargs.get("chunk_elements", 0)):
                        self.assertLess(sm, sms)
                        for index in range(start, stop, step):
                            seen[index] += 1
                    self.assertTrue(all(count == 1 for count in seen),
                                    f"{schedule} {elements} {sms}")

    def test_rejects_bad_configuration(self):
        with self.assertRaisesRegex(ValueError, "elements"):
            AUDIT.audit(0, 4, (1,))
        with self.assertRaisesRegex(ValueError, "element_bytes"):
            AUDIT.audit(16, 0, (1,))
        with self.assertRaisesRegex(ValueError, "multiple"):
            AUDIT.audit(16, 7, (1,))
        with self.assertRaisesRegex(ValueError, "positive"):
            AUDIT.audit(16, 4, (0,))
        with self.assertRaisesRegex(ValueError, "schedule"):
            AUDIT.slices_for(16, 1, "whatever", 1, 0)
        with self.assertRaisesRegex(ValueError, "chunk_elements"):
            AUDIT.slices_for(16, 1, "block-cyclic", 1, 0)

    def test_matches_the_partition_header_static_asserts(self):
        """mu_partition.h carries these as static_asserts; keep in step."""
        self.assertEqual(AUDIT.contiguous_partition(4102, 4, 0, 16), (0, 1040))
        self.assertEqual(AUDIT.contiguous_partition(4102, 4, 1, 16)[0], 1040)
        self.assertEqual(AUDIT.contiguous_partition(4102, 4, 3, 16)[1], 4102)
        self.assertEqual(AUDIT.contiguous_partition(256, 4, 2, 8), (128, 192))


if __name__ == "__main__":
    unittest.main()
