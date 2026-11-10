#!/usr/bin/env python3
"""Deterministic Granite 3.1 MoE routing and scatter oracle.

This reference deliberately covers routing structure, not expert GEMM
arithmetic. It has no third-party dependencies and performs no hardware or
network access.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


NUM_TOKENS = 128
HIDDEN_SIZE = 1024
NUM_EXPERTS = 32
TOP_K = 8
INTERMEDIATE_SIZE = 512
NUM_ASSIGNMENTS = NUM_TOKENS * TOP_K

# Shapes from the official Granite 3.1 1B-A400M configuration/checkpoint.
# The combined projection contains 512 gate and 512 up outputs.
OFFICIAL_GRANITE_CONFIG = {
    "model": "ibm-granite/granite-3.1-1b-a400m-instruct",
    "hidden_size": HIDDEN_SIZE,
    "num_local_experts": NUM_EXPERTS,
    "num_experts_per_tok": TOP_K,
    "intermediate_size": INTERMEDIATE_SIZE,
    "router_weight_shape": [NUM_EXPERTS, HIDDEN_SIZE],
    "gate_up_weight_shape": [NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE],
    "down_weight_shape": [NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE],
    "representative_input_shape": [NUM_TOKENS, HIDDEN_SIZE],
    "representative_output_shape": [NUM_TOKENS, HIDDEN_SIZE],
}


def fp32(value: float) -> float:
    """Round a Python float to IEEE-754 binary32."""

    return struct.unpack("<f", struct.pack("<f", value))[0]


@dataclass(frozen=True)
class Assignment:
    """One token-to-expert assignment in canonical token-major order."""

    assignment_id: int
    token: int
    topk_rank: int
    expert: int
    weight: float


@dataclass(frozen=True)
class RoutingOracle:
    """Complete routing and stable-scatter result for one invocation."""

    top_indices: list[list[int]]
    top_weights: list[list[float]]
    expert_counts: list[int]
    expert_offsets: list[int]
    packed_assignments: list[Assignment]
    assignment_to_packed: list[int]


def deterministic_logits(
    num_tokens: int = NUM_TOKENS,
    num_experts: int = NUM_EXPERTS,
) -> list[list[float]]:
    """Generate reproducible logits without requiring checkpoint data.

    Each token receives a rotated expert preference. Across 128 tokens this
    exercises every expert evenly while retaining a transparent expected
    ordering.
    """

    if num_tokens <= 0 or num_experts <= 0:
        raise ValueError("token and expert counts must be positive")
    rows: list[list[float]] = []
    for token in range(num_tokens):
        preferred = token % num_experts
        row = []
        for expert in range(num_experts):
            cyclic_distance = (expert - preferred) % num_experts
            # The token term shifts the row but cannot change its ranking.
            row.append(fp32(-float(cyclic_distance) + token * 0.0009765625))
        rows.append(row)
    return rows


def deterministic_topk(logits: Sequence[float], k: int = TOP_K) -> list[int]:
    """Return top-k expert IDs; exact ties prefer the lower expert ID."""

    if not 0 < k <= len(logits):
        raise ValueError("k must be in [1, len(logits)]")
    if any(not math.isfinite(value) for value in logits):
        raise ValueError("router logits must all be finite")
    return sorted(range(len(logits)), key=lambda expert: (-logits[expert], expert))[:k]


def selected_softmax_fp32(selected_logits: Sequence[float]) -> list[float]:
    """Compute softmax over only the supplied selected logits.

    Inputs, shifted values, exponentials, running sum, and final probabilities
    are rounded to binary32. ``math.exp`` itself is the host reference
    transcendental; this function does not claim a bit-accurate device exp.
    """

    if not selected_logits:
        raise ValueError("selected logits must not be empty")
    values = [fp32(value) for value in selected_logits]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("selected logits must all be finite")
    maximum = max(values)
    exponentials = [fp32(math.exp(fp32(value - maximum))) for value in values]
    denominator = fp32(0.0)
    for value in exponentials:
        denominator = fp32(denominator + value)
    return [fp32(value / denominator) for value in exponentials]


def route_logits(logits: Sequence[Sequence[float]], k: int = TOP_K) -> tuple[
    list[list[int]], list[list[float]]
]:
    """Select experts and selected-only softmax weights for every token."""

    if len(logits) != NUM_TOKENS:
        raise ValueError(f"expected {NUM_TOKENS} token rows, got {len(logits)}")
    top_indices: list[list[int]] = []
    top_weights: list[list[float]] = []
    for token, row in enumerate(logits):
        if len(row) != NUM_EXPERTS:
            raise ValueError(
                f"token {token}: expected {NUM_EXPERTS} expert logits, got {len(row)}"
            )
        indices = deterministic_topk(row, k)
        weights = selected_softmax_fp32([row[expert] for expert in indices])
        top_indices.append(indices)
        top_weights.append(weights)
    return top_indices, top_weights


def stable_scatter(
    top_indices: Sequence[Sequence[int]],
    top_weights: Sequence[Sequence[float]],
) -> tuple[list[int], list[int], list[Assignment], list[int]]:
    """Stably group canonical token-major assignments by expert."""

    if len(top_indices) != NUM_TOKENS or len(top_weights) != NUM_TOKENS:
        raise ValueError(f"expected routing data for {NUM_TOKENS} tokens")

    assignments: list[Assignment] = []
    buckets: list[list[Assignment]] = [[] for _ in range(NUM_EXPERTS)]
    for token, (indices, weights) in enumerate(zip(top_indices, top_weights)):
        if len(indices) != TOP_K or len(weights) != TOP_K:
            raise ValueError(f"token {token}: expected exactly {TOP_K} assignments")
        if len(set(indices)) != TOP_K:
            raise ValueError(f"token {token}: selected expert IDs are not unique")
        for rank, (expert, weight) in enumerate(zip(indices, weights)):
            if not 0 <= expert < NUM_EXPERTS:
                raise ValueError(f"token {token}: expert {expert} is out of range")
            assignment = Assignment(
                assignment_id=token * TOP_K + rank,
                token=token,
                topk_rank=rank,
                expert=expert,
                weight=fp32(weight),
            )
            assignments.append(assignment)
            buckets[expert].append(assignment)

    expert_counts = [len(bucket) for bucket in buckets]
    expert_offsets = [0]
    for count in expert_counts:
        expert_offsets.append(expert_offsets[-1] + count)
    packed_assignments = [assignment for bucket in buckets for assignment in bucket]
    assignment_to_packed = [-1] * len(assignments)
    for packed_index, assignment in enumerate(packed_assignments):
        assignment_to_packed[assignment.assignment_id] = packed_index
    return expert_counts, expert_offsets, packed_assignments, assignment_to_packed


def build_oracle(logits: Sequence[Sequence[float]] | None = None) -> RoutingOracle:
    """Build and validate the complete representative routing oracle."""

    if logits is None:
        logits = deterministic_logits()
    top_indices, top_weights = route_logits(logits)
    counts, offsets, packed, inverse = stable_scatter(top_indices, top_weights)
    oracle = RoutingOracle(top_indices, top_weights, counts, offsets, packed, inverse)
    validate_structure(oracle)
    return oracle


def validate_structure(oracle: RoutingOracle) -> None:
    """Prove zero-tolerance routing and stable-scatter invariants."""

    if len(oracle.top_indices) != NUM_TOKENS or len(oracle.top_weights) != NUM_TOKENS:
        raise ValueError("routing rows do not match the official token count")
    for token, (indices, weights) in enumerate(
        zip(oracle.top_indices, oracle.top_weights)
    ):
        if len(indices) != TOP_K or len(weights) != TOP_K:
            raise ValueError(f"token {token} does not contain exactly eight routes")
        if len(set(indices)) != TOP_K:
            raise ValueError(f"token {token} repeats a selected expert")
        if any(not 0 <= expert < NUM_EXPERTS for expert in indices):
            raise ValueError(f"token {token} contains an out-of-range expert")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError(f"token {token} contains an invalid routing weight")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=2e-7):
            raise ValueError(f"token {token} selected weights do not sum to one")
    if len(oracle.expert_counts) != NUM_EXPERTS:
        raise ValueError("expert count table has the wrong length")
    if len(oracle.expert_offsets) != NUM_EXPERTS + 1:
        raise ValueError("expert offset table has the wrong length")
    if len(oracle.packed_assignments) != NUM_ASSIGNMENTS:
        raise ValueError("packed assignment count is not exactly 1,024")
    if len(oracle.assignment_to_packed) != NUM_ASSIGNMENTS:
        raise ValueError("inverse assignment map has the wrong length")
    if sum(oracle.expert_counts) != NUM_ASSIGNMENTS:
        raise ValueError("expert counts do not sum to exactly 1,024")
    if oracle.expert_offsets[0] != 0 or oracle.expert_offsets[-1] != NUM_ASSIGNMENTS:
        raise ValueError("expert offsets do not span all assignments")

    expected_offsets = [0]
    for count in oracle.expert_counts:
        expected_offsets.append(expected_offsets[-1] + count)
    if oracle.expert_offsets != expected_offsets:
        raise ValueError("expert offsets are inconsistent with expert counts")

    seen_ids: set[int] = set()
    seen_token_ranks: set[tuple[int, int]] = set()
    for expert in range(NUM_EXPERTS):
        begin = oracle.expert_offsets[expert]
        end = oracle.expert_offsets[expert + 1]
        group = oracle.packed_assignments[begin:end]
        ids = [assignment.assignment_id for assignment in group]
        if ids != sorted(ids):
            raise ValueError(f"expert {expert} group is not stable token-major order")
        for packed_index, assignment in enumerate(group, start=begin):
            if assignment.expert != expert:
                raise ValueError(f"assignment in expert {expert} group has wrong expert")
            if assignment.assignment_id != assignment.token * TOP_K + assignment.topk_rank:
                raise ValueError("assignment ID is not canonical token-major ID")
            if oracle.top_indices[assignment.token][assignment.topk_rank] != expert:
                raise ValueError("packed expert does not match the selected route")
            expected_weight = oracle.top_weights[assignment.token][assignment.topk_rank]
            if assignment.weight != expected_weight:
                raise ValueError("packed weight does not match the selected route")
            if oracle.assignment_to_packed[assignment.assignment_id] != packed_index:
                raise ValueError("assignment inverse map is inconsistent")
            if assignment.assignment_id in seen_ids:
                raise ValueError("duplicate assignment ID")
            token_rank = (assignment.token, assignment.topk_rank)
            if token_rank in seen_token_ranks:
                raise ValueError("duplicate token/rank assignment")
            seen_ids.add(assignment.assignment_id)
            seen_token_ranks.add(token_rank)

    if seen_ids != set(range(NUM_ASSIGNMENTS)):
        raise ValueError("assignments are missing or outside the canonical ID range")
    expected_token_ranks = {
        (token, rank) for token in range(NUM_TOKENS) for rank in range(TOP_K)
    }
    if seen_token_ranks != expected_token_ranks:
        raise ValueError("not every token/rank pair occurs exactly once")


def oracle_as_dict(oracle: RoutingOracle) -> dict[str, object]:
    """Return a JSON-serializable oracle with an explicit precision contract."""

    return {
        "official_config": OFFICIAL_GRANITE_CONFIG,
        "scope": {
            "implemented": "router top-k, selected-only softmax, stable assignment scatter",
            "input": "deterministic synthetic FP32 router logits",
            "structural_exactness": "zero tolerance",
            "bf16_checkpoint_equivalence": False,
            "mxfp8_mapped_arithmetic_equivalence": False,
        },
        "routing": {
            "top_indices": oracle.top_indices,
            "top_weights": oracle.top_weights,
            "expert_counts": oracle.expert_counts,
            "expert_offsets": oracle.expert_offsets,
            "packed_assignments": [asdict(item) for item in oracle.packed_assignments],
            "assignment_to_packed": oracle.assignment_to_packed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the complete validated JSON oracle to this path",
    )
    args = parser.parse_args()

    oracle = build_oracle()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(oracle_as_dict(oracle), indent=2) + "\n")
        print(f"wrote validated {NUM_ASSIGNMENTS}-assignment oracle to {args.output}")
    else:
        summary = {
            "tokens": NUM_TOKENS,
            "experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "assignments": len(oracle.packed_assignments),
            "all_assignments_exactly_once": True,
            "expert_count_min": min(oracle.expert_counts),
            "expert_count_max": max(oracle.expert_counts),
            "numeric_claim": "routing structure only; not BF16/MXFP8 equivalence",
        }
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
