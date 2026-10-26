#!/usr/bin/env python3
"""Focused fail-closed tests for the xRAGE FPGA-plan validator."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit import (ARTIFACT_NAME, AuditError, TARGETS, expected_run_argv,
                   require, validate_target)
from run_exact_cyclotron import fnv1a_u32, nonzero_u64


def valid_target(target_id: str) -> dict:
    sms, memory, config = TARGETS[target_id]
    return {
        "id": target_id, "kind": "firesim", "sm_count": sms,
        "memory": memory, "loader": "direct-hashed", "bdf": "1a:00.0",
        "bitstream_config": config, "run_argv": expected_run_argv(),
    }


def valid_workload(target_id: str) -> dict:
    sms = TARGETS[target_id][0]
    return {"name": ARTIFACT_NAME.format(sms=sms),
            "correctness": "admissible-winner-set-checked"}


class FailClosedHelpersTest(unittest.TestCase):
    def test_require_accepts_true(self):
        require(True, "unused")

    def test_require_rejects_false(self):
        with self.assertRaisesRegex(AuditError, "sentinel"):
            require(False, "sentinel")

    def test_all_five_targets_pin_1a(self):
        self.assertEqual(len(TARGETS), 5)
        for target_id in TARGETS:
            self.assertEqual(valid_target(target_id)["bdf"], "1a:00.0")

    def test_command_is_fresh_programming_form(self):
        argv = expected_run_argv()
        self.assertEqual(argv[6:12], ["-b", "{bitstream_config}", "--bdf",
                                      "{bdf}", "--fastloadmem", "{elf}"])
        self.assertNotIn("--no-program", argv)

    def test_mutations_are_detectable(self):
        target_id = "firesim-2sm-hbm"
        target = valid_target(target_id)
        workload = valid_workload(target_id)
        self.assertEqual(validate_target(target, workload), target_id)
        cases = []
        wrong_bdf = copy.deepcopy(target); wrong_bdf["bdf"] = "1b:00.0"; cases.append((wrong_bdf, workload))
        reuse = copy.deepcopy(target); reuse["run_argv"].insert(-1, "--no-program"); cases.append((reuse, workload))
        wrong_config = copy.deepcopy(target); wrong_config["bitstream_config"] = "2SM_DRAM"; cases.append((wrong_config, workload))
        weak = copy.deepcopy(workload); weak["correctness"] = "guards-and-nonzero-exploratory"; cases.append((target, weak))
        for mutated_target, mutated_workload in cases:
            with self.subTest(target=mutated_target, workload=mutated_workload):
                with self.assertRaises(AuditError):
                    validate_target(mutated_target, mutated_workload)

    def test_snapshot_fnv_matches_cyclotron_word_hash(self):
        self.assertEqual(fnv1a_u32(bytes.fromhex("0100000002000000")),
                         0x082F2407B4E8902A)

    def test_snapshot_nonzero_u64_counts_complete_words(self):
        self.assertEqual(nonzero_u64(bytes(8) + bytes.fromhex("0100000000000000")), 1)


if __name__ == "__main__":
    unittest.main()
