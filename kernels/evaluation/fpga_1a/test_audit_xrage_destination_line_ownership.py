#!/usr/bin/env python3
"""Regression tests for the destination-line ownership audit.

These run on tiny synthetic destination lists, so they need neither the xRAGE
trace nor any device. The first test is the one that would have caught the
shipped bug: a rank-aligned partition splits a 64-byte destination line, a
line-aligned one does not.
"""

import importlib.util
from pathlib import Path
import unittest


MODULE = Path(__file__).with_name("audit_xrage_destination_line_ownership.py")
SPEC = importlib.util.spec_from_file_location("xrage_line_ownership_audit", MODULE)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


# Ranks 7 and 8 (the granule-8 boundary at 2 SM) are destinations 56 and 57,
# which share 64-byte line 7.  Every other destination starts its own line.
SPLIT_AT_RANK_8 = [0, 8, 16, 24, 32, 40, 48, 56, 57, 64, 72, 80, 88, 96, 104, 112]


class DestinationLineOwnershipAuditTests(unittest.TestCase):
    def test_rank_boundaries_split_a_destination_line(self):
        """The pre-repair device: the audit must report the split, not hide it."""
        report = AUDIT.audit(SPLIT_AT_RANK_8, (1, 2), granule=8,
                             payload_bytes=8, line_bytes=64,
                             boundary_mode="rank")
        self.assertEqual(report["status"], "fail")
        by_sms = {c["sm_count"]: c for c in report["checks"]}
        self.assertEqual(by_sms[1]["status"], "pass")
        self.assertEqual(by_sms[2]["status"], "fail")
        self.assertFalse(by_sms[2]["cache_line_exclusive"])
        self.assertTrue(by_sms[2]["coverage_exactly_once"])
        split = by_sms[2]["split_lines"][0]
        self.assertEqual(split["shared_line"], 7)
        self.assertEqual(split["last_left_destination"], 56)
        self.assertEqual(split["first_right_destination"], 57)
        self.assertEqual(split["left_byte_offset"], 0)
        self.assertEqual(split["right_byte_offset"], 8)

    def test_line_aligned_boundaries_keep_the_line_on_one_sm(self):
        """The repaired device: the boundary moves off the line, nothing else."""
        report = AUDIT.audit(SPLIT_AT_RANK_8, (1, 2, 4), granule=8,
                             payload_bytes=8, line_bytes=64)
        self.assertEqual(report["boundary_mode"], "line-aligned")
        self.assertEqual(report["status"], "pass")
        for check in report["checks"]:
            self.assertTrue(check["cache_line_exclusive"])
            self.assertTrue(check["coverage_exactly_once"])
            self.assertEqual(check["split_line_count"], 0)
            self.assertEqual(sum(check["owned_groups"]), len(SPLIT_AT_RANK_8))
        by_sms = {c["sm_count"]: c for c in report["checks"]}
        self.assertEqual(by_sms[2]["group_ranks"], [0, 9])

    def test_boundaries_stay_contiguous_and_exhaustive(self):
        """Snapping must not drop, duplicate or reorder a destination group."""
        for destinations in (SPLIT_AT_RANK_8,
                             list(range(40)),
                             [3 * i for i in range(37)],
                             [0, 1, 2, 3, 4, 5, 6, 7],
                             [i // 4 * 64 + i % 4 for i in range(53)]):
            occupied = sorted(set(destinations))
            for sms in (1, 2, 3, 4, 8):
                bounds = AUDIT.partition_bounds(occupied, sms, 8, 8, 64,
                                                "line-aligned")
                self.assertEqual(bounds[0][0], 0)
                self.assertEqual(bounds[-1][1], len(occupied))
                for left, right in zip(bounds, bounds[1:]):
                    self.assertEqual(left[1], right[0])
                    self.assertLessEqual(left[0], left[1])

    def test_line_aligned_rank_is_monotone_and_lands_on_a_line_start(self):
        occupied = sorted(set(SPLIT_AT_RANK_8))
        previous = -1
        for rank in range(len(occupied) + 2):
            snapped = AUDIT.line_aligned_rank(occupied, rank, 8, 64)
            self.assertGreaterEqual(snapped, previous)
            previous = snapped
            self.assertGreaterEqual(snapped, min(rank, len(occupied)))
            if 0 < snapped < len(occupied):
                self.assertNotEqual(AUDIT.line_of(occupied[snapped], 8, 64),
                                    AUDIT.line_of(occupied[snapped - 1], 8, 64))

    def test_rejects_bad_configuration(self):
        with self.assertRaises(ValueError):
            AUDIT.audit([0, 1], (0,), 8, 8, 64)
        with self.assertRaises(ValueError):
            AUDIT.audit([0, 1], (1,), 8, 64, 8)
        with self.assertRaises(ValueError):
            AUDIT.partition_bounds([0, 1], 1, 8, 8, 64, "whatever")

    def test_matches_the_device_partition_on_a_dense_list(self):
        """With no holes, rank and line-aligned partitions must agree."""
        dense = list(range(1024))
        rank = AUDIT.partition_bounds(dense, 4, 8, 8, 64, "rank")
        aligned = AUDIT.partition_bounds(dense, 4, 8, 8, 64, "line-aligned")
        self.assertEqual(rank, aligned)


if __name__ == "__main__":
    unittest.main()
