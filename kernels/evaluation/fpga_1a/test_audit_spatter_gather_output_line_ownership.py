#!/usr/bin/env python3
"""Regression tests for the Gather output-line ownership audit.

Tiny synthetic sizes only; no suite and no device.  The bug test here is a
block-cyclic chunk that is not a whole number of 64-byte lines, which is the
one way this otherwise dense, directly indexed path can share a line.
"""

import importlib.util
from pathlib import Path
import unittest


MODULE = Path(__file__).with_name("audit_spatter_gather_output_line_ownership.py")
SPEC = importlib.util.spec_from_file_location("gather_output_line_ownership",
                                              MODULE)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ChunkWidthTests(unittest.TestCase):
    """32 owner slots of 8 bytes: 8 slots per 64-byte line."""

    def test_sub_line_chunk_splits_an_output_line(self):
        """chunk = 4 slots = 32 bytes -- the unsafe variant.  Must fail."""
        report = AUDIT.audit(32, 32, 10, 1, (1, 2), schedule="block-cyclic",
                             chunk_elements=4)
        self.assertEqual(report["status"], "fail")
        by_sms = {c["sm_count"]: c for c in report["checks"]}
        self.assertEqual(by_sms[1]["status"], "pass")
        self.assertEqual(by_sms[2]["status"], "fail")
        self.assertFalse(by_sms[2]["cache_line_exclusive"])
        self.assertTrue(by_sms[2]["coverage_exactly_once"])
        self.assertTrue(by_sms[2]["transfer_coverage_exactly_once"])
        split = by_sms[2]["split_lines"][0]
        self.assertEqual(split["line"], 0)
        self.assertEqual(split["writer_sms"], [0, 1])
        self.assertEqual(split["first_byte_offset_per_sm"], {"0": 0, "1": 32})

    def test_line_multiple_chunk_keeps_the_line_on_one_sm(self):
        """chunk = 8 slots = 64 bytes -- safe.  Same array, must pass."""
        report = AUDIT.audit(32, 32, 10, 1, (1, 2, 4), schedule="block-cyclic",
                             chunk_elements=8)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["partition_cache_line_safe"])
        for check in report["checks"]:
            self.assertEqual(check["split_line_count"], 0)
            self.assertEqual(check["transfers_executed"], 320)


class DeployedShapeTests(unittest.TestCase):
    def test_contiguous_granule_eight_is_exclusive(self):
        """spatter-0 / spatter-4: 256 owner slots, granule 8 = one line."""
        report = AUDIT.audit(256, 256, 1024, 1, (1, 2, 4), granule=8)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["transfers_expected"], 262144)
        self.assertEqual(report["transfers_executed"], 262144)
        for check in report["checks"]:
            self.assertTrue(check["cache_line_exclusive"])
            self.assertEqual(sum(check["transfers_per_sm"]), 262144)

    def test_block_cyclic_1024_is_exclusive(self):
        """xrage-gather: SPATTER_GATHER_BLOCK_CYCLIC_ELEMENTS = 1024."""
        report = AUDIT.audit(8368968, 8368968, 1, 1, (1, 2, 4),
                             schedule="block-cyclic", chunk_elements=1024)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["transfers_executed"], 8368968)
        for check in report["checks"]:
            self.assertEqual(check["split_line_count"], 0)
            self.assertEqual(check["unowned_elements"], 0)
            self.assertEqual(check["doubly_owned_elements"], 0)

    def test_round_robin_is_the_unsafe_variant_and_must_fail(self):
        report = AUDIT.audit(256, 256, 4, 1, (1, 2, 4),
                             schedule="round-robin")
        self.assertEqual(report["status"], "fail")
        by_sms = {c["sm_count"]: c for c in report["checks"]}
        self.assertEqual(by_sms[1]["status"], "pass")
        self.assertEqual(by_sms[2]["split_line_count"], 32)
        self.assertEqual(by_sms[4]["split_line_count"], 32)


class TransferCoverageTests(unittest.TestCase):
    def test_wrap_spreads_iterations_over_owner_slots(self):
        # length 4, wrap 2 -> 8 owner slots; slots 0..3 take iterations 0,2,4
        # and slots 4..7 take iterations 1,3.
        counts = AUDIT.transfers_per_owner(8, 4, 5, 2)
        self.assertEqual(counts, [3, 3, 3, 3, 2, 2, 2, 2])
        self.assertEqual(sum(counts), 4 * 5)

    def test_owner_slot_with_no_iteration_fails_coverage(self):
        """count < wrap leaves owner slots that no iteration ever reaches."""
        report = AUDIT.audit(16, 4, 2, 4, (1, 2), granule=8)
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["transfer_coverage_exactly_once"])
        self.assertEqual(report["transfers_executed"], 8)
        self.assertEqual(report["transfers_expected"], 8)
        for check in report["checks"]:
            self.assertFalse(check["transfer_coverage_exactly_once"])
            self.assertEqual(check["status"], "fail")
            # The line map itself is clean; only the transfer map is not.
            self.assertTrue(check["cache_line_exclusive"])

    def test_rejects_inconsistent_owner_count(self):
        with self.assertRaisesRegex(ValueError, "length . wrap"):
            AUDIT.audit(10, 4, 4, 2, (1,))
        with self.assertRaisesRegex(ValueError, "wrap"):
            AUDIT.transfers_per_owner(4, 4, 4, 0)


if __name__ == "__main__":
    unittest.main()
