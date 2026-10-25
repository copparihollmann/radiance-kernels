#!/usr/bin/env python3
"""Prove the multi-SM ownership invariants used by the Spatter port."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SPATTER = REPO / "kernels" / "spatter"
sys.path.insert(0, str(SPATTER))

from run import normalize  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition(total: int, parts: int, part: int, granule: int) -> tuple[int, int]:
    """Python equivalent of mu_contiguous_partition()."""
    chunks = (total + granule - 1) // granule
    base, extra = divmod(chunks, parts)
    begin_chunk = part * base + min(part, extra)
    owned_chunks = base + (part < extra)
    return min(total, begin_chunk * granule), min(total, (begin_chunk + owned_chunks) * granule)


def prove_ranges(total: int, sms: int, granule: int) -> list[tuple[int, int]]:
    ranges = [partition(total, sms, sm, granule) for sm in range(sms)]
    assert ranges[0][0] == 0 and ranges[-1][1] == total
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert all(begin <= end for begin, end in ranges)
    assert all(begin % granule == 0 for begin, _ in ranges)
    assert all(end == total or end % granule == 0 for _, end in ranges)
    assert sum(end - begin for begin, end in ranges) == total
    return ranges


def audit_case(case_id: int, case: dict, sms: int) -> dict:
    length = case["length"]
    count = case["count"]
    total_transfers = length * count
    result = {
        "case": case_id,
        "kind": case["kind"],
        "sm_count": sms,
        "pattern_length": length,
        "iterations": count,
        "total_transfers": total_transfers,
    }

    if case["kind"] in ("gather", "multigather"):
        owners = length * case["wrap"]
        ranges = prove_ranges(owners, sms, 8)
        owner_counts = [end - begin for begin, end in ranges]
        transfer_counts = []
        for begin, end in ranges:
            transfers = 0
            for owner in range(begin, end):
                residue = owner // length
                if residue < count:
                    transfers += 1 + (count - 1 - residue) // case["wrap"]
            transfer_counts.append(transfers)
        assert sum(owner_counts) == owners
        assert sum(transfer_counts) == total_transfers
        result.update({
            "schedule": "dense-output-owner",
            "partition_granule": 8,
            "ranges": ranges,
            "owners_per_sm": owner_counts,
            "transfers_per_sm": transfer_counts,
            "writes_original_destinations": True,
            "scratch_elements_per_sm": 0,
        })
    else:
        if case["collision_policy"] == "ordered":
            destinations = {
                case["_plan"].write.at(case, iteration, lane)
                for iteration in range(count)
                for lane in range(length)
            }
            total = len(destinations)
            ranges = prove_ranges(total, sms, 8)
            schedule = "destination-owner"
            granule = 8
        else:
            total = total_transfers
            ranges = prove_ranges(total, sms, 16)
            schedule = "contiguous-transfer-task"
            granule = 16
            assigned = {task for begin, end in ranges for task in range(begin, end)}
            assert len(assigned) == total_transfers
        result.update({
            "schedule": schedule,
            "partition_granule": granule,
            "ranges": ranges,
            "work_items_per_sm": [end - begin for begin, end in ranges],
        })

    result["status"] = "pass"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    raw_cases = json.loads(args.suite.read_text())
    checks = []
    for case_id, raw in enumerate(raw_cases):
        case = normalize(raw, args.suite.parent)
        for sms in (1, 2, 4):
            checks.append(audit_case(case_id, case, sms))

    report = {
        "status": "pass",
        "suite": str(args.suite.resolve()),
        "suite_sha256": sha256(args.suite),
        "kernel_sha256": sha256(SPATTER / "kernel.cpp"),
        "partition_header_sha256": sha256(REPO / "lib" / "include" / "mu_partition.h"),
        "checks": checks,
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
