#!/usr/bin/env python3
"""Compute the per-SM owner map of a dense, directly indexed output array.

Why this exists
---------------
Most Radiance HPC kernels partition a dense output by element index and claim
"disjoint ... 64-byte output lines" in a manifest string.  A manifest string is
an assertion.  `audit_stencil3d_artifact.py`, `audit_gemv_artifact.py` and
friends only compare that string; `audit_stream_blockcyclic_partition.py`
computes element coverage but still emits `"cache_line_exclusive": True` as a
hardcoded literal.  None of them maps an element to an address and asks which
SMs write that address's 64-byte line.

The xRAGE Scatter defect (reports/DEFECT-xrage-4sm-cache-line-2026-11-02.md)
showed what an asserted invariant is worth: the claim held, the addresses did
not, and two SMs wrote bytes 40 and 48 of one line.  That defect needed an
indirection.  This script exists to establish, by computation rather than by
argument, that the kernels *without* an indirection really are safe -- and to
fail if one of them is not.

What it computes
----------------
For each SM count it builds an owner map over EVERY element of the output, not
only the elements next to a partition boundary, then independently walks every
64-byte line and collects the set of SMs that write it.  It reports:

  cache_line_exclusive   every line has exactly one owning SM
  coverage_exactly_once  no element is unowned and none is owned twice

Both are computed from the owner map.  Nothing is asserted.

Schedules
---------
  contiguous    mu_contiguous_partition(elements, sms, block_id, granule)
                -- STREAM, stencil3d, nearn, hotspot3d columns, gemv rows
  block-cyclic  chunk_elements-sized chunks round-robin across SMs
                -- STREAM_BLOCK_CYCLIC_*, spmv-jds row chunks, xRAGE gather
  round-robin   element index % sms.  This is the UNSAFE variant: the naive
                grid-stride schedule that `lib/include/mu_partition.h` says it
                exists to avoid ("keeps neighboring cache lines on one cluster
                instead of interleaving their lanes across every cluster").  It
                must fail on any array wider than one line at 2 or more SMs.  A
                detector that cannot fail proves nothing, so every real run
                should be accompanied by a round-robin run that fails.

`--repeat-stride` covers kernels whose owner map is over one dimension but
whose writes repeat every `stride` elements (hotspot3d owns columns and sweeps
z: the owner map repeats every plane).  The repeat is line-preserving only when
the stride is a whole number of lines, which is checked, not assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


UNOWNED = 0xFF
SCHEDULES = ("contiguous", "block-cyclic", "round-robin")


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


def slices_for(elements: int, sms: int, schedule: str, granule: int,
               chunk_elements: int) -> list[tuple[int, int, int, int]]:
    """Return (start, stop, step, sm) for every range an SM writes.

    This is the schedule, transcribed from the device.  The audit below never
    reads it again; it only reads the owner map these ranges produce.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    if schedule == "contiguous":
        if granule < 1:
            raise ValueError("granule must be positive")
        out = []
        for sm in range(sms):
            begin, end = contiguous_partition(elements, sms, sm, granule)
            if end > begin:
                out.append((begin, end, 1, sm))
        return out
    if schedule == "block-cyclic":
        if chunk_elements < 1:
            raise ValueError("chunk_elements must be positive")
        chunks = (elements + chunk_elements - 1) // chunk_elements
        out = []
        for chunk in range(chunks):
            begin = chunk * chunk_elements
            end = min(begin + chunk_elements, elements)
            out.append((begin, end, 1, chunk % sms))
        return out
    return [(sm, elements, sms, sm) for sm in range(sms) if sm < elements]


def _span(start: int, stop: int, step: int) -> int:
    return len(range(start, stop, step))


def audit(elements: int, element_bytes: int, sm_counts,
          schedule: str = "contiguous", granule: int = 16,
          chunk_elements: int = 0, line_bytes: int = 64,
          repeat_stride: int = 0, label: str | None = None) -> dict:
    if elements < 1:
        raise ValueError("elements must be positive")
    if element_bytes < 1:
        raise ValueError("element_bytes must be positive")
    if line_bytes % element_bytes:
        raise ValueError("line_bytes must be a multiple of element_bytes")
    per_line = line_bytes // element_bytes
    lines = (elements + per_line - 1) // per_line
    if repeat_stride < 0:
        raise ValueError("repeat_stride must not be negative")
    stride_line_aligned = (repeat_stride % per_line == 0) if repeat_stride else True

    checks = []
    for sms in sm_counts:
        if sms < 1:
            raise ValueError("SM counts must be positive")
        if sms >= UNOWNED:
            raise ValueError("SM count does not fit the owner map sentinel")

        # 1. Build the owner map over every element.
        owner = bytearray([UNOWNED]) * elements
        duplicates = 0
        element_counts = [0] * sms
        for start, stop, step, sm in slices_for(elements, sms, schedule,
                                                granule, chunk_elements):
            segment = owner[start:stop:step]
            width = len(segment)
            duplicates += width - segment.count(UNOWNED)
            owner[start:stop:step] = bytes([sm]) * width
            element_counts[sm] += width
        unowned = owner.count(UNOWNED)

        # 2. Independently walk every 64-byte line and collect its writers.
        #    This does not look at the schedule, only at the owner map, so a
        #    line split anywhere -- not only at a boundary -- is caught.
        split_lines = []
        split_line_count = 0
        for line in range(lines):
            begin = line * per_line
            segment = owner[begin:min(begin + per_line, elements)]
            writers = sorted(set(segment) - {UNOWNED})
            if len(writers) < 2:
                continue
            split_line_count += 1
            if len(split_lines) < 16:
                offsets = {}
                for position, sm in enumerate(segment):
                    if sm != UNOWNED:
                        offsets.setdefault(sm, position * element_bytes)
                split_lines.append({
                    "line": line,
                    "first_element": begin,
                    "writer_sms": writers,
                    "first_byte_offset_per_sm":
                        {str(sm): offsets[sm] for sm in writers},
                })

        exclusive = split_line_count == 0 and stride_line_aligned
        covered = unowned == 0 and duplicates == 0
        checks.append({
            "sm_count": sms,
            "elements_per_sm": element_counts,
            "total_elements_owned": sum(element_counts),
            "split_lines": split_lines,
            "split_line_count": split_line_count,
            "unowned_elements": unowned,
            "doubly_owned_elements": duplicates,
            "repeat_stride_line_aligned": stride_line_aligned,
            "coverage_exactly_once": covered,    # computed, not asserted
            "cache_line_exclusive": exclusive,   # computed, not asserted
            "status": "pass" if exclusive and covered else "fail",
        })

    passed = all(check["status"] == "pass" for check in checks)
    return {
        "schema": "radiance-dense-line-ownership-v1",
        "label": label,
        "elements": elements,
        "element_bytes": element_bytes,
        "line_bytes": line_bytes,
        "elements_per_line": per_line,
        "lines": lines,
        "schedule": schedule,
        "granule": granule if schedule == "contiguous" else None,
        "chunk_elements": chunk_elements if schedule == "block-cyclic" else None,
        "repeat_stride": repeat_stride or None,
        "checks": checks,
        "partition_cache_line_safe":
            all(c["cache_line_exclusive"] for c in checks),
        "partition_complete": all(c["coverage_exactly_once"] for c in checks),
        "partition_overlap": sum(c["doubly_owned_elements"] for c in checks),
        "status": "pass" if passed else "fail",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--elements", type=int, required=True)
    ap.add_argument("--element-bytes", type=int, default=4)
    ap.add_argument("--sm-counts", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--schedule", choices=SCHEDULES, default="contiguous")
    ap.add_argument("--granule", type=int, default=16,
                    help="mu_contiguous_partition granule, in elements")
    ap.add_argument("--chunk-elements", type=int, default=0,
                    help="block-cyclic chunk width, in elements")
    ap.add_argument("--line-bytes", type=int, default=64)
    ap.add_argument("--repeat-stride", type=int, default=0,
                    help="elements between repeats of the owner map "
                         "(hotspot3d: one z-plane)")
    ap.add_argument("--label")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    report = audit(args.elements, args.element_bytes, args.sm_counts,
                   args.schedule, args.granule, args.chunk_elements,
                   args.line_bytes, args.repeat_stride, args.label)
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
            print(f"FAIL {check['sm_count']} SM: line {split['line']} "
                  f"(elements from {split['first_element']}) is written by "
                  f"{writers}", file=sys.stderr)
        if not check["coverage_exactly_once"]:
            print(f"FAIL {check['sm_count']} SM: "
                  f"{check['unowned_elements']} unowned and "
                  f"{check['doubly_owned_elements']} doubly owned elements",
                  file=sys.stderr)
        if not check["repeat_stride_line_aligned"]:
            print(f"FAIL {check['sm_count']} SM: repeat stride "
                  f"{report['repeat_stride']} is not a whole number of "
                  f"{report['line_bytes']}-byte lines", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
