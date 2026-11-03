#!/usr/bin/env python3
"""Check destination-line ownership for the parallel-collision Spatter Scatter.

Why this exists
---------------
When `SPATTER_DESTINATION_OWNER_GROUPS` is 0 -- which is every Spatter case in
the hpc-v2 plan with `collision_policy: "parallel"` and a scatter destination
(spatter-1 Scatter, spatter-2 GS, spatter-3 MultiScatter) -- the device splits
work by TASK, not by destination:

    const MuPartition range = mu_contiguous_partition(
        tasks, NIGHTLY_CLUSTERS, block_id, 16u);     // kernels/spatter/kernel.cpp

and then writes `dst = pattern[j] + delta * i`.  That is the same shape as the
defect fixed in 4d89301: the partition is over an INDEX into a schedule, while
the thing that must not be shared is an ADDRESS reached through an indirection.
A task-aligned boundary is not a destination-line-aligned boundary unless the
pattern happens to make it one.

`audit_spatter_partition.py` already audits this path, but only as ranges:

    ranges = prove_ranges(total, sms, 16)
    assigned = {task for begin, end in ranges for task in range(begin, end)}
    assert len(assigned) == total_transfers

It proves every TASK runs exactly once.  It never evaluates a destination, so
it cannot see two SMs writing one 64-byte line.  That is the gap this closes.

What it computes
----------------
For each SM count it builds an owner map over EVERY task, resolves each task to
its real destination through `kernels/spatter/plan.py` (the same address map the
host reference and the device use), and then independently collects, for every
occupied 64-byte destination line, the set of SMs that write it.  It reports:

  cache_line_exclusive    every destination line has exactly one owning SM
  coverage_exactly_once   every task is owned by exactly one SM
  destination_overlap     tasks that write the same destination, and whether
                          the colliding tasks are on different SMs

Nothing is asserted; `--schedule round-robin` reproduces the naive grid-stride
schedule that `mu_contiguous_partition` exists to replace, and must fail, so the
detector is demonstrably able to fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SPATTER = REPO / "kernels" / "spatter"

UNOWNED = 0xFF
SCHEDULES = ("contiguous", "round-robin")


def contiguous_partition(total: int, partitions: int, index: int,
                         granule: int = 1) -> tuple[int, int]:
    """Transcription of mu_contiguous_partition (lib/include/mu_partition.h)."""
    if partitions == 0 or index >= partitions or total == 0:
        return (0, 0)
    granule = granule or 1
    chunks = total // granule + (1 if total % granule else 0)
    per, extra = divmod(chunks, partitions)
    begin_chunk = index * per + (index if index < extra else extra)
    owned = per + (1 if index < extra else 0)
    begin = min(begin_chunk * granule, total)
    end = min((begin_chunk + owned) * granule, total)
    return (begin, end)


def task_owners(tasks: int, sms: int, schedule: str, granule: int) -> bytearray:
    """Owner map over every task index, built from the device schedule."""
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    if sms < 1:
        raise ValueError("SM counts must be positive")
    if sms >= UNOWNED:
        raise ValueError("SM count does not fit the owner map sentinel")
    owner = bytearray([UNOWNED]) * tasks
    if schedule == "contiguous":
        if granule < 1:
            raise ValueError("granule must be positive")
        for sm in range(sms):
            begin, end = contiguous_partition(tasks, sms, sm, granule)
            owner[begin:end] = bytes([sm]) * (end - begin)
    else:
        for sm in range(sms):
            segment = owner[sm:tasks:sms]
            owner[sm:tasks:sms] = bytes([sm]) * len(segment)
    return owner


def audit(destinations, sm_counts, granule: int = 16, payload_bytes: int = 8,
          line_bytes: int = 64, schedule: str = "contiguous",
          label: str | None = None) -> dict:
    """`destinations[task]` is the address index that task writes."""
    destinations = list(destinations)
    tasks = len(destinations)
    if tasks < 1:
        raise ValueError("there must be at least one task")
    if payload_bytes < 1 or line_bytes % payload_bytes:
        raise ValueError("line_bytes must be a multiple of payload_bytes")
    per_line = line_bytes // payload_bytes

    checks = []
    for sms in sm_counts:
        owner = task_owners(tasks, sms, schedule, granule)
        unowned = owner.count(UNOWNED)
        tasks_per_sm = [owner.count(sm) for sm in range(sms)]
        duplicates = tasks - unowned - sum(tasks_per_sm)

        # Walk every task -- not only the ones beside a boundary -- and record
        # which SM writes each destination and each destination line.
        line_owner: dict[int, int] = {}
        split_seen: set[int] = set()
        split_lines = []
        destination_owner: dict[int, tuple[int, int]] = {}
        collisions = []
        cross_sm_collisions = 0
        for task, destination in enumerate(destinations):
            sm = owner[task]
            if sm == UNOWNED:
                continue
            previous = destination_owner.get(destination)
            if previous is None:
                destination_owner[destination] = (sm, task)
            else:
                if previous[0] != sm:
                    cross_sm_collisions += 1
                if len(collisions) < 16:
                    collisions.append({
                        "destination": destination,
                        "first_task": previous[1], "first_sm": previous[0],
                        "second_task": task, "second_sm": sm,
                    })
            line = destination // per_line
            first = line_owner.setdefault(line, sm)
            if first == sm or line in split_seen:
                continue
            split_seen.add(line)
            if len(split_lines) < 16:
                split_lines.append({
                    "shared_line": line,
                    "left_sm": first, "right_sm": sm,
                    "boundary_task": task,
                    "last_left_destination":
                        destinations[task - 1] if task else None,
                    "first_right_destination": destination,
                    "left_byte_offset":
                        (destinations[task - 1] % per_line) * payload_bytes
                        if task else None,
                    "right_byte_offset": (destination % per_line) * payload_bytes,
                })

        exclusive = not split_seen
        covered = unowned == 0 and duplicates == 0
        clean = cross_sm_collisions == 0
        checks.append({
            "sm_count": sms,
            "tasks_per_sm": tasks_per_sm,
            "split_lines": split_lines,
            "split_line_count": len(split_seen),
            "unowned_tasks": unowned,
            "doubly_owned_tasks": duplicates,
            "occupied_destinations": len(destination_owner),
            "destination_collisions": collisions,
            "cross_sm_destination_collisions": cross_sm_collisions,
            "coverage_exactly_once": covered,      # computed, not asserted
            "cache_line_exclusive": exclusive,     # computed, not asserted
            "destination_exclusive": clean,        # computed, not asserted
            "status": "pass" if exclusive and covered and clean else "fail",
        })

    passed = all(check["status"] == "pass" for check in checks)
    return {
        "schema": "radiance-spatter-scatter-task-line-ownership-v1",
        "label": label,
        "tasks": tasks,
        "payload_bytes": payload_bytes,
        "line_bytes": line_bytes,
        "destinations_per_line": per_line,
        "max_destination": max(destinations),
        "schedule": schedule,
        "partition_granule_tasks": granule if schedule == "contiguous" else None,
        "checks": checks,
        "partition_cache_line_safe":
            all(c["cache_line_exclusive"] for c in checks),
        "partition_complete": all(c["coverage_exactly_once"] for c in checks),
        "partition_overlap": sum(c["doubly_owned_tasks"] for c in checks),
        "status": "pass" if passed else "fail",
    }


def destinations_for_case(case: dict) -> list[int]:
    """Resolve every task to its destination with the shared address map."""
    plan = case["_plan"]
    length = case["length"]
    return [plan.write.at(case, iteration, j)
            for iteration in range(case["count"])
            for j in range(length)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--count", type=int,
                    help="override the suite iteration count")
    ap.add_argument("--sm-counts", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--granule", type=int, default=16,
                    help="tasks per partition chunk (device uses 16)")
    ap.add_argument("--payload-bytes", type=int, default=8)
    ap.add_argument("--line-bytes", type=int, default=64)
    ap.add_argument("--schedule", choices=SCHEDULES, default="contiguous",
                    help="contiguous models the device; round-robin "
                         "reproduces the naive grid-stride schedule")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    sys.path.insert(0, str(SPATTER))
    from run import normalize  # noqa: E402

    raw = json.loads(args.suite.read_text())
    entry = dict(raw[args.case])
    if args.count is not None:
        entry["count"] = args.count
    case = normalize(entry, args.suite.parent)
    if case["kind"] not in ("scatter", "gs", "multiscatter"):
        ap.error(f"case {args.case} is a {case['kind']}; this audit covers the "
                 "scatter destination paths")
    if case["collision_policy"] != "parallel":
        ap.error("ordered collision policy uses the destination-owner groups; "
                 "audit it with audit_xrage_destination_line_ownership.py")

    report = audit(destinations_for_case(case), args.sm_counts, args.granule,
                   args.payload_bytes, args.line_bytes, args.schedule,
                   label=f"{args.suite.stem}-case{args.case}-{case['kind']}")
    report.update({
        "suite": str(args.suite.resolve()),
        "suite_sha256": sha256(args.suite),
        "case": args.case,
        "kind": case["kind"],
        "pattern_length": case["length"],
        "iterations": case["count"],
        "kernel_sha256": sha256(SPATTER / "kernel.cpp"),
        "partition_header_sha256": sha256(REPO / "lib" / "include" / "mu_partition.h"),
    })
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        sys.stdout.write(text)

    for check in report["checks"]:
        if check["status"] == "pass":
            continue
        for split in check["split_lines"]:
            print(f"FAIL {check['sm_count']} SM: line {split['shared_line']} is "
                  f"written by SM{split['left_sm']} (dest "
                  f"{split['last_left_destination']}, byte "
                  f"{split['left_byte_offset']}) and SM{split['right_sm']} "
                  f"(dest {split['first_right_destination']}, byte "
                  f"{split['right_byte_offset']})", file=sys.stderr)
        if not check["coverage_exactly_once"]:
            print(f"FAIL {check['sm_count']} SM: {check['unowned_tasks']} "
                  f"unowned and {check['doubly_owned_tasks']} doubly owned "
                  "tasks", file=sys.stderr)
        if not check["destination_exclusive"]:
            print(f"FAIL {check['sm_count']} SM: "
                  f"{check['cross_sm_destination_collisions']} destinations "
                  "are written by more than one SM", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
