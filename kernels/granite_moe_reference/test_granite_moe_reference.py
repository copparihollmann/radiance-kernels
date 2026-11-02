"""Tests for the bounded Granite 3.1 MoE routing reference."""

from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from granite_moe_reference import (
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NUM_ASSIGNMENTS,
    NUM_EXPERTS,
    NUM_TOKENS,
    OFFICIAL_GRANITE_CONFIG,
    TOP_K,
    build_oracle,
    deterministic_logits,
    deterministic_topk,
    fp32,
    selected_softmax_fp32,
    validate_structure,
)


class GraniteMoeReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.oracle = build_oracle()

    def test_official_shape_constants_are_recorded(self) -> None:
        self.assertEqual(NUM_TOKENS, 128)
        self.assertEqual(HIDDEN_SIZE, 1024)
        self.assertEqual(NUM_EXPERTS, 32)
        self.assertEqual(TOP_K, 8)
        self.assertEqual(INTERMEDIATE_SIZE, 512)
        self.assertEqual(OFFICIAL_GRANITE_CONFIG["router_weight_shape"], [32, 1024])
        self.assertEqual(
            OFFICIAL_GRANITE_CONFIG["gate_up_weight_shape"], [32, 1024, 1024]
        )
        self.assertEqual(
            OFFICIAL_GRANITE_CONFIG["down_weight_shape"], [32, 1024, 512]
        )

    def test_lower_expert_id_breaks_exact_topk_ties(self) -> None:
        logits = [-10.0] * NUM_EXPERTS
        for expert in (9, 2, 17, 4, 31, 1, 12, 7, 6):
            logits[expert] = 3.0
        # Nine experts tie for eight slots. The eight lowest IDs must win.
        self.assertEqual(deterministic_topk(logits), [1, 2, 4, 6, 7, 9, 12, 17])

    def test_selected_softmax_excludes_unselected_experts(self) -> None:
        selected = [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]
        actual = selected_softmax_fp32(selected)
        expected_exp = [fp32(math.exp(fp32(value - 4.0))) for value in selected]
        denominator = fp32(0.0)
        for value in expected_exp:
            denominator = fp32(denominator + value)
        expected = [fp32(value / denominator) for value in expected_exp]
        self.assertEqual(actual, expected)
        self.assertAlmostEqual(sum(actual), 1.0, places=6)

        # A 32-way softmax would reserve probability mass for these 24 logits.
        full_denominator = sum(math.exp(value - 4.0) for value in selected + [-3.0] * 24)
        full_selected_mass = sum(math.exp(value - 4.0) for value in selected) / full_denominator
        self.assertLess(full_selected_mass, 1.0)
        self.assertGreater(abs(sum(actual) - full_selected_mass), 1e-3)

    def test_synthetic_input_is_deterministic_and_complete(self) -> None:
        first = deterministic_logits()
        second = deterministic_logits()
        self.assertEqual(first, second)
        self.assertEqual(len(first), NUM_TOKENS)
        self.assertTrue(all(len(row) == NUM_EXPERTS for row in first))

    def test_all_1024_assignments_occur_exactly_once(self) -> None:
        packed = self.oracle.packed_assignments
        self.assertEqual(len(packed), NUM_ASSIGNMENTS)
        self.assertEqual(
            {assignment.assignment_id for assignment in packed},
            set(range(NUM_ASSIGNMENTS)),
        )
        self.assertEqual(
            {(assignment.token, assignment.topk_rank) for assignment in packed},
            {
                (token, rank)
                for token in range(NUM_TOKENS)
                for rank in range(TOP_K)
            },
        )
        self.assertEqual(sum(self.oracle.expert_counts), NUM_ASSIGNMENTS)

    def test_scatter_is_grouped_by_expert_and_stable(self) -> None:
        for expert in range(NUM_EXPERTS):
            begin = self.oracle.expert_offsets[expert]
            end = self.oracle.expert_offsets[expert + 1]
            group = self.oracle.packed_assignments[begin:end]
            self.assertTrue(all(item.expert == expert for item in group))
            ids = [item.assignment_id for item in group]
            self.assertEqual(ids, sorted(ids))

    def test_inverse_map_round_trips_every_assignment(self) -> None:
        for packed_index, assignment in enumerate(self.oracle.packed_assignments):
            self.assertEqual(
                self.oracle.assignment_to_packed[assignment.assignment_id], packed_index
            )

    def test_validator_rejects_duplicate_and_missing_assignment(self) -> None:
        broken = copy.deepcopy(self.oracle)
        packed = list(broken.packed_assignments)
        packed[-1] = packed[0]
        object.__setattr__(broken, "packed_assignments", packed)
        with self.assertRaises(ValueError):
            validate_structure(broken)


if __name__ == "__main__":
    unittest.main()
