#!/usr/bin/env python3
"""Regression tests for the parallel-collision Scatter line-ownership audit.

These run on tiny synthetic destination lists, so they need neither a Spatter
suite nor any device.  The first test is the bug in its own shape: the task
partition is aligned, the DESTINATIONS it reaches through the pattern
indirection are not, and two SMs end up in one 64-byte line.
"""

import importlib.util
from pathlib import Path
import unittest


MODULE = Path(__file__).with_name("audit_spatter_scatter_task_line_ownership.py")
SPEC = importlib.util.spec_from_file_location("scatter_task_line_ownership",
                                              MODULE)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


# 32 tasks, granule 16, 2 SMs -> the task boundary is task 16, which is a
# perfectly aligned task index.  The pattern sends task 15 to destination 120
# and task 16 to destination 121, and those two 8-byte payloads share 64-byte
# line 15.  This is the xRAGE defect transplanted onto the task schedule.
TORN = [8 * task for task in range(32)]
TORN[16] = 121

# Same list with the indirection repaired: task 16 starts a new line.
REPAIRED = list(TORN)
REPAIRED[16] = 128


class IndirectDestinationTests(unittest.TestCase):
    def test_aligned_task_boundary_still_splits_a_destination_line(self):
        report = AUDIT.audit(TORN, (1, 2), granule=16)
        self.assertEqual(report["status"], "fail")
        by_sms = {c["sm_count"]: c for c in report["checks"]}
        self.assertEqual(by_sms[1]["status"], "pass")
        self.assertEqual(by_sms[2]["status"], "fail")
        self.assertFalse(by_sms[2]["cache_line_exclusive"])
        # Every task still runs exactly once and no two tasks share an
        # address, so neither task coverage nor destination uniqueness -- nor
        # a digest over the output -- can see this.
        self.assertTrue(by_sms[2]["coverage_exactly_once"])
        self.assertTrue(by_sms[2]["destination_exclusive"])
        split = by_sms[2]["split_lines"][0]
        self.assertEqual(split["shared_line"], 15)
        self.assertEqual(split["left_sm"], 0)
        self.assertEqual(split["right_sm"], 1)
        self.assertEqual(split["boundary_task"], 16)
        self.assertEqual(split["last_left_destination"], 120)
        self.assertEqual(split["first_right_destination"], 121)
        self.assertEqual(split["left_byte_offset"], 0)
        self.assertEqual(split["right_byte_offset"], 8)

    def test_repaired_destinations_keep_each_line_on_one_sm(self):
        report = AUDIT.audit(REPAIRED, (1, 2, 4), granule=16)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["partition_cache_line_safe"])
        self.assertTrue(report["partition_complete"])
        self.assertEqual(report["partition_overlap"], 0)
        for check in report["checks"]:
            self.assertEqual(check["split_line_count"], 0)
            self.assertEqual(sum(check["tasks_per_sm"]), 32)


class DenseDestinationTests(unittest.TestCase):
    """The deployed gpu-stream cases: UNIFORM:256:1 makes dst == task."""

    def test_dense_destinations_are_exclusive_at_every_sm_count(self):
        report = AUDIT.audit(list(range(4096)), (1, 2, 4), granule=16)
        self.assertEqual(report["status"], "pass")
        for check in report["checks"]:
            self.assertTrue(check["cache_line_exclusive"])
            self.assertEqual(check["occupied_destinations"], 4096)

    def test_round_robin_is_the_unsafe_variant_and_must_fail(self):
        report = AUDIT.audit(list(range(4096)), (1, 2, 4),
                             schedule="round-robin")
        self.assertEqual(report["status"], "fail")
        by_sms = {c["sm_count"]: c for c in report["checks"]}
        self.assertEqual(by_sms[1]["status"], "pass")
        self.assertEqual(by_sms[2]["split_line_count"], 512)
        self.assertEqual(by_sms[4]["split_line_count"], 512)
        for check in report["checks"]:
            self.assertTrue(check["coverage_exactly_once"])


class CollisionTests(unittest.TestCase):
    def test_two_sms_writing_one_address_is_reported(self):
        destinations = [8 * task for task in range(32)]
        destinations[16] = destinations[0]      # SM1 reuses SM0's address
        report = AUDIT.audit(destinations, (2,), granule=16)
        self.assertEqual(report["status"], "fail")
        check = report["checks"][0]
        self.assertFalse(check["destination_exclusive"])
        self.assertEqual(check["cross_sm_destination_collisions"], 1)
        collision = check["destination_collisions"][0]
        self.assertEqual(collision["destination"], 0)
        self.assertEqual((collision["first_sm"], collision["second_sm"]), (0, 1))

    def test_same_sm_collision_is_counted_but_not_fatal(self):
        destinations = [8 * task for task in range(32)]
        destinations[1] = destinations[0]       # both tasks are on SM0
        report = AUDIT.audit(destinations, (2,), granule=16)
        check = report["checks"][0]
        self.assertEqual(check["cross_sm_destination_collisions"], 0)
        self.assertEqual(len(check["destination_collisions"]), 1)
        self.assertEqual(report["status"], "pass")


class ScheduleTests(unittest.TestCase):
    def test_task_owner_map_covers_every_task_once(self):
        for schedule in AUDIT.SCHEDULES:
            for tasks in (1, 15, 16, 17, 64, 257):
                for sms in (1, 2, 3, 4, 8):
                    owner = AUDIT.task_owners(tasks, sms, schedule, 16)
                    self.assertEqual(len(owner), tasks)
                    self.assertEqual(owner.count(AUDIT.UNOWNED), 0)
                    self.assertEqual(sum(owner.count(sm) for sm in range(sms)),
                                     tasks)

    def test_contiguous_owner_map_is_monotone(self):
        owner = AUDIT.task_owners(257, 4, "contiguous", 16)
        self.assertEqual(sorted(owner), list(owner))

    def test_rejects_bad_configuration(self):
        with self.assertRaisesRegex(ValueError, "at least one task"):
            AUDIT.audit([], (1,))
        with self.assertRaisesRegex(ValueError, "multiple"):
            AUDIT.audit([0, 1], (1,), payload_bytes=7)
        with self.assertRaisesRegex(ValueError, "positive"):
            AUDIT.audit([0, 1], (0,))
        with self.assertRaisesRegex(ValueError, "schedule"):
            AUDIT.task_owners(16, 1, "whatever", 16)
        with self.assertRaisesRegex(ValueError, "granule"):
            AUDIT.task_owners(16, 1, "contiguous", 0)

    def test_matches_the_partition_header_static_asserts(self):
        self.assertEqual(AUDIT.contiguous_partition(4102, 4, 0, 16), (0, 1040))
        self.assertEqual(AUDIT.contiguous_partition(256, 4, 2, 8), (128, 192))


if __name__ == "__main__":
    unittest.main()
