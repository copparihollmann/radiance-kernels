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
compacts writes to the list of *occupied* destinations, and the device splits
that compacted list with
`mu_contiguous_partition(SPATTER_GROUP_COUNT, NIGHTLY_CLUSTERS, block_id, 8u)`
(kernels/spatter/kernel.cpp:64,241). Eight adjacent list entries are not eight
adjacent output addresses once the destination space has holes, so aligning a
rank does not align an address.

On the real xRAGE trace this splits a 64-byte line across two SMs at 4 SMs:

    occupied destinations 1,191,878
    4 SM group ranks [0, 297976, 595944, 893912]
    SM2/SM3: last-left dest 1,373,645, first-right dest 1,373,646
             -> same line 171,705, byte offsets 40 and 48

1 SM and 2 SM show no split, which is why a 2-SM pass never caught it.

This script computes the invariant instead of asserting it: it maps every
occupied destination to its real line, and requires each line to have exactly one
owning SM. It exits non-zero when the invariant is violated, so it works as a
regression test for the fix (partition runs of occupied LINE ids, not ranks).
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


def audit(destinations: list[int], sm_counts, granule: int,
          payload_bytes: int, line_bytes: int) -> dict:
    occupied = sorted(set(destinations))
    total = len(occupied)
    checks = []
    for sms in sm_counts:
        bounds = [contiguous_partition(total, sms, p, granule) for p in range(sms)]
        splits = []
        for p in range(1, sms):
            begin = bounds[p][0]
            if begin <= 0 or begin >= total:
                continue
            left, right = occupied[begin - 1], occupied[begin]
            if line_of(left, payload_bytes, line_bytes) == \
               line_of(right, payload_bytes, line_bytes):
                splits.append({
                    "left_sm": p - 1, "right_sm": p, "boundary_rank": begin,
                    "last_left_destination": left, "first_right_destination": right,
                    "shared_line": line_of(left, payload_bytes, line_bytes),
                    "left_byte_offset": (left % (line_bytes // payload_bytes)) * payload_bytes,
                    "right_byte_offset": (right % (line_bytes // payload_bytes)) * payload_bytes,
                })
        checks.append({
            "sm_count": sms,
            "group_ranks": [b for b, _ in bounds],
            "split_lines": splits,
            "cache_line_exclusive": not splits,     # computed, not asserted
            "status": "pass" if not splits else "fail",
        })
    return {
        "schema": "radiance-xrage-destination-line-ownership-v1",
        "transfers": len(destinations),
        "occupied_destinations": total,
        "max_destination": max(occupied) if occupied else 0,
        "payload_bytes": payload_bytes,
        "line_bytes": line_bytes,
        "partition_granule_entries": granule,
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
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    raw = array("I")
    raw.frombytes(args.destinations.read_bytes())
    if sys.byteorder != "little":
        raw.byteswap()

    report = audit(list(raw), args.sm_counts, args.granule,
                   args.payload_bytes, args.line_bytes)
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
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
