#!/usr/bin/env python3
"""Check destination-line ownership for every Spatter/xRAGE Gather output.

Why this exists
---------------
The Gather families (spatter-0 Gather, spatter-4 MultiGather, xrage-gather)
write a DENSE output: `spatter_dense[owner]`, with `owner` the partitioned index
itself.  That is the one structural property the defective Scatter path lacked,
and it is the reason `audit_xrage_blockcyclic_partition.py` -- which checks only
`begin % 8 == 0` and then emits `"cache_line_exclusive": True` as a hardcoded
literal -- happened to be sound on the Gather path it guards.

"Happened to be sound" is not evidence.  Three different schedules are compiled
into `kernels/spatter/kernel.cpp` for this path:

    SPATTER_GATHER_BLOCK_CYCLIC_ELEMENTS   chunk-cyclic, chunk = N elements
    SPATTER_GATHER_BLOCK_CYCLIC_16         chunk-cyclic, chunk = 16 elements
    (neither)                              mu_contiguous_partition(..., 8u)

and their safety depends on the arithmetic of (owners, chunk, SM count), not on
a claim in a manifest.  This script computes it.

What it computes
----------------
Owner slots are dense and 8 bytes wide, so the owner map IS the address map and
`audit_dense_contiguous_line_ownership.py` does the line work: it builds an
owner map over every owner slot and independently collects the writers of every
64-byte line.  This script adds the Gather-specific part:

  transfer_coverage_exactly_once  every (iteration, lane) transfer of the
                                  original Spatter case is executed exactly
                                  once, counted per owner slot from the wrap
                                  schedule rather than assumed

`--schedule round-robin` reproduces the naive grid-stride schedule, which must
fail, so the detector is demonstrably able to fail.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SPATTER = REPO / "kernels" / "spatter"

_DENSE_PATH = Path(__file__).with_name("audit_dense_contiguous_line_ownership.py")
_SPEC = importlib.util.spec_from_file_location("dense_line_ownership", _DENSE_PATH)
DENSE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(DENSE)

PAYLOAD_BYTES = 8
SCHEDULES = DENSE.SCHEDULES


def transfers_per_owner(owners: int, length: int, count: int,
                        wrap: int) -> list[int]:
    """Iterations that land on each owner slot, from the device's wrap loop.

    The device runs `for (i = owner / length; i < count; i += wrap)`, so owner
    slot `o` executes one transfer per iteration congruent to `o / length`.
    """
    if wrap < 1:
        raise ValueError("wrap must be positive")
    counts = []
    for owner in range(owners):
        residue = owner // length
        counts.append(0 if residue >= count else 1 + (count - 1 - residue) // wrap)
    return counts


def audit(owners: int, length: int, count: int, wrap: int, sm_counts,
          schedule: str = "contiguous", granule: int = 8,
          chunk_elements: int = 0, line_bytes: int = 64,
          label: str | None = None) -> dict:
    if owners != length * wrap:
        raise ValueError("owners must equal length * wrap")
    report = DENSE.audit(owners, PAYLOAD_BYTES, sm_counts, schedule, granule,
                         chunk_elements, line_bytes, 0, label)
    report["schema"] = "radiance-spatter-gather-output-line-ownership-v1"
    report["pattern_length"] = length
    report["iterations"] = count
    report["wrap"] = wrap

    per_owner = transfers_per_owner(owners, length, count, wrap)
    executed = sum(per_owner)
    expected = length * count
    report["transfers_expected"] = expected
    report["transfers_executed"] = executed
    exact = executed == expected and all(c > 0 for c in per_owner)
    report["transfer_coverage_exactly_once"] = exact   # computed, not asserted

    for check in report["checks"]:
        sms = check["sm_count"]
        slices = DENSE.slices_for(owners, sms, schedule, granule, chunk_elements)
        per_sm = [0] * sms
        for start, stop, step, sm in slices:
            for owner in range(start, stop, step):
                per_sm[sm] += per_owner[owner]
        check["transfers_per_sm"] = per_sm
        check["transfers_executed"] = sum(per_sm)
        check["transfer_coverage_exactly_once"] = \
            exact and sum(per_sm) == expected
        if not check["transfer_coverage_exactly_once"]:
            check["status"] = "fail"
    if any(c["status"] == "fail" for c in report["checks"]):
        report["status"] = "fail"
    return report


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--count", type=int, help="override the suite count")
    ap.add_argument("--sm-counts", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--schedule", choices=SCHEDULES, default="contiguous",
                    help="contiguous models SPATTER_GATHER_BLOCK_CYCLIC_* off; "
                         "block-cyclic models either block-cyclic build; "
                         "round-robin is the unsafe grid-stride variant")
    ap.add_argument("--granule", type=int, default=8,
                    help="owner slots per partition chunk (device uses 8)")
    ap.add_argument("--chunk-elements", type=int, default=0,
                    help="SPATTER_GATHER_BLOCK_CYCLIC_ELEMENTS (16 for the "
                         "_16 build)")
    ap.add_argument("--line-bytes", type=int, default=64)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    sys.path.insert(0, str(SPATTER))
    from run import normalize  # noqa: E402

    raw = json.loads(args.suite.read_text())
    entry = dict(raw[args.case])
    if args.count is not None:
        entry["count"] = args.count
    case = normalize(entry, args.suite.parent)
    if case["kind"] not in ("gather", "multigather"):
        ap.error(f"case {args.case} is a {case['kind']}; this audit covers the "
                 "dense Gather output")

    length, wrap = case["length"], case["wrap"]
    report = audit(length * wrap, length, case["count"], wrap, args.sm_counts,
                   args.schedule, args.granule, args.chunk_elements,
                   args.line_bytes,
                   label=f"{args.suite.stem}-case{args.case}-{case['kind']}")
    report.update({
        "suite": str(args.suite.resolve()),
        "suite_sha256": sha256(args.suite),
        "case": args.case,
        "kind": case["kind"],
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
            writers = ", ".join(
                f"SM{sm} (byte {split['first_byte_offset_per_sm'][str(sm)]})"
                for sm in split["writer_sms"])
            print(f"FAIL {check['sm_count']} SM: output line {split['line']} "
                  f"(owner slots from {split['first_element']}) is written by "
                  f"{writers}", file=sys.stderr)
        if not check["coverage_exactly_once"]:
            print(f"FAIL {check['sm_count']} SM: {check['unowned_elements']} "
                  f"unowned and {check['doubly_owned_elements']} doubly owned "
                  "owner slots", file=sys.stderr)
        if not check["transfer_coverage_exactly_once"]:
            print(f"FAIL {check['sm_count']} SM: "
                  f"{check['transfers_executed']} transfers executed, "
                  f"{report['transfers_expected']} expected", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
