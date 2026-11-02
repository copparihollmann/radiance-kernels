#!/usr/bin/env python3
"""Check that every real destination cache line has exactly one owning SM.

Why this exists
---------------
`audit_xrage_blockcyclic_partition.py` emits `"cache_line_exclusive": True` as a
hardcoded literal. Its only real assertions are `begin % 8 == 0` and
`end % 8 == 0`, i.e. alignment of a destination's RANK in the list. For the
dense Gather output, rank alignment does imply address alignment, so that script
is sound on the path it guards.

It is NOT sound for the destination-grouped Scatter path. There the host
compacts writes to the list of *occupied* destinations, and the device split
that compacted list with
`mu_contiguous_partition(SPATTER_GROUP_COUNT, NIGHTLY_CLUSTERS, block_id, 8u)`.
Eight adjacent list entries are not eight adjacent output addresses once the
destination space has holes, so aligning a rank does not align an address.

On the real xRAGE trace that split a 64-byte line across two SMs at 4 SMs:

    occupied destinations 1,191,878
    4 SM group ranks [0, 297976, 595944, 893912]
    SM2/SM3: last-left dest 1,373,645, first-right dest 1,373,646
             -> same line 171,705, byte offsets 40 and 48

1 SM and 2 SM show no split, which is why a 2-SM pass never caught it.

The repair (kernels/spatter/kernel.cpp, `spatter_destination_line_boundary`)
derives ownership from the destination index instead of the rank: each rank
boundary produced by `mu_contiguous_partition` is advanced until it is the first
rank of a new destination line. `--boundary-mode rank` reproduces the pre-repair
device and must still fail; `line-aligned` (the default) models the repaired
device.

This script computes the invariant instead of asserting it. It maps EVERY
occupied destination -- not only the ranks next to a boundary -- to its real
line, and requires each line to have exactly one owning SM and each rank to be
owned exactly once. It exits non-zero when either invariant is violated, so it
works as a regression test for the fix.
"""

from __future__ import annotations

import argparse
import json
import sys
from array import array
from pathlib import Path


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


def line_of(destination: int, payload_bytes: int, line_bytes: int) -> int:
    return destination // (line_bytes // payload_bytes)


def line_aligned_rank(occupied: list[int], rank: int, payload_bytes: int,
                      line_bytes: int) -> int:
    """Transcription of spatter_destination_line_boundary (kernel.cpp).

    Advance a rank boundary until the destination it names starts a new
    destination line. Monotone in `rank`, so partition p's end stays partition
    p+1's begin and the partitions remain contiguous, disjoint and exhaustive.
    """
    total = len(occupied)
    if rank >= total:
        return total
    while rank != 0 and \
            line_of(occupied[rank], payload_bytes, line_bytes) == \
            line_of(occupied[rank - 1], payload_bytes, line_bytes):
        rank += 1
        if rank >= total:
            return total
    return rank


BOUNDARY_MODES = ("line-aligned", "rank")


def partition_bounds(occupied: list[int], sms: int, granule: int,
                     payload_bytes: int, line_bytes: int,
                     boundary_mode: str) -> list[tuple[int, int]]:
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(f"boundary_mode must be one of {BOUNDARY_MODES}")
    total = len(occupied)
    bounds = [contiguous_partition(total, sms, p, granule) for p in range(sms)]
    if boundary_mode == "rank":
        return bounds
    return [(line_aligned_rank(occupied, b, payload_bytes, line_bytes),
             line_aligned_rank(occupied, e, payload_bytes, line_bytes))
            for b, e in bounds]


def audit(destinations: list[int], sm_counts, granule: int,
          payload_bytes: int, line_bytes: int,
          boundary_mode: str = "line-aligned") -> dict:
    occupied = sorted(set(destinations))
    total = len(occupied)
    per_line = line_bytes // payload_bytes
    if per_line < 1:
        raise ValueError("line_bytes must be a multiple of payload_bytes")
    checks = []
    for sms in sm_counts:
        if sms < 1:
            raise ValueError("SM counts must be positive")
        bounds = partition_bounds(occupied, sms, granule, payload_bytes,
                                  line_bytes, boundary_mode)

        # Walk every occupied destination and record the SM that owns it.
        # Nothing here trusts that a boundary is the only place a line can
        # straddle two SMs; the owner map is built from the partition itself.
        owner = [-1] * total
        duplicates = 0
        for p, (begin, end) in enumerate(bounds):
            for rank in range(begin, end):
                if owner[rank] != -1:
                    duplicates += 1
                owner[rank] = p
        unowned = sum(1 for o in owner if o == -1)

        line_owner: dict[int, int] = {}
        split_line_ids: set[int] = set()
        split_lines = []
        for rank, destination in enumerate(occupied):
            sm = owner[rank]
            if sm == -1:
                continue
            line = destination // per_line
            first = line_owner.setdefault(line, sm)
            if first == sm or line in split_line_ids:
                continue
            split_line_ids.add(line)
            if len(split_lines) < 16:
                split_lines.append({
                    "left_sm": first, "right_sm": sm,
                    "boundary_rank": rank,
                    "last_left_destination": occupied[rank - 1] if rank else None,
                    "first_right_destination": destination,
                    "shared_line": line,
                    "left_byte_offset":
                        (occupied[rank - 1] % per_line) * payload_bytes if rank else None,
                    "right_byte_offset": (destination % per_line) * payload_bytes,
                })
        split_count = len(split_line_ids)
        exclusive = split_count == 0
        covered = unowned == 0 and duplicates == 0
        checks.append({
            "sm_count": sms,
            "group_ranks": [b for b, _ in bounds],
            "group_ends": [e for _, e in bounds],
            "owned_groups": [e - b for b, e in bounds],
            "split_lines": split_lines,
            "split_line_count": split_count,
            "unowned_groups": unowned,
            "doubly_owned_groups": duplicates,
            "coverage_exactly_once": covered,       # computed, not asserted
            "cache_line_exclusive": exclusive,      # computed, not asserted
            "status": "pass" if exclusive and covered else "fail",
        })
    return {
        "schema": "radiance-xrage-destination-line-ownership-v2",
        "transfers": len(destinations),
        "occupied_destinations": total,
        "max_destination": max(occupied) if occupied else 0,
        "payload_bytes": payload_bytes,
        "line_bytes": line_bytes,
        "partition_granule_entries": granule,
        "boundary_mode": boundary_mode,
        "checks": checks,
        "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--destinations", type=Path, required=True,
                    help="little-endian u32 destination index per transfer")
    ap.add_argument("--sm-counts", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--granule", type=int, default=8,
                    help="list entries per partition chunk (device uses 8)")
    ap.add_argument("--payload-bytes", type=int, default=8)
    ap.add_argument("--line-bytes", type=int, default=64)
    ap.add_argument("--boundary-mode", choices=BOUNDARY_MODES,
                    default="line-aligned",
                    help="line-aligned models the repaired device; rank "
                         "reproduces the pre-repair rank-only partition")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    raw = array("I")
    raw.frombytes(args.destinations.read_bytes())
    if sys.byteorder != "little":
        raw.byteswap()

    report = audit(list(raw), args.sm_counts, args.granule,
                   args.payload_bytes, args.line_bytes, args.boundary_mode)
    report["destinations_path"] = str(args.destinations.resolve())
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        sys.stdout.write(text)

    for check in report["checks"]:
        if check["status"] == "fail":
            for s in check["split_lines"]:
                print(f"FAIL {check['sm_count']} SM: line {s['shared_line']} is written by "
                      f"SM{s['left_sm']} (dest {s['last_left_destination']}, byte "
                      f"{s['left_byte_offset']}) and SM{s['right_sm']} (dest "
                      f"{s['first_right_destination']}, byte {s['right_byte_offset']})",
                      file=sys.stderr)
            if not check["coverage_exactly_once"]:
                print(f"FAIL {check['sm_count']} SM: {check['unowned_groups']} unowned and "
                      f"{check['doubly_owned_groups']} doubly owned destination groups",
                      file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
