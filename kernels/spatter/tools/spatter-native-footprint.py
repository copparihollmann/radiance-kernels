#!/usr/bin/env python3
"""Size every upstream Spatter case at its declared repetition count."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import normalize  # noqa: E402


FIELDS = (
    "suite", "case", "kind", "pattern_length", "count", "transfers",
    "source_elements", "output_elements", "source_bytes", "output_bytes",
    "embedded_bytes", "native_fit", "limit", "largest_single_launch_count",
    "minimum_count_chunks",
)


def original_dimensions(probe: dict, count: int) -> tuple[int, int]:
    kind = probe["kind"]
    if kind == "gs":
        source = max(probe["gather"]) + probe["delta_gather"] * (count - 1) + 1
        output = max(probe["scatter"]) + probe["delta_scatter"] * (count - 1) + 1
    elif kind in ("gather", "multigather"):
        source = max(probe["pattern"]) + probe["delta"] * (count - 1) + 1
        output = probe["length"] * probe["wrap"]
    else:
        source = probe["length"] * probe["wrap"]
        output = max(probe["pattern"]) + probe["delta"] * (count - 1) + 1
    return source, output


def largest_fitting_count(raw: dict, original: int) -> int:
    lo, hi = 1, original
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            normalize({**raw, "count": mid})
        except ValueError:
            hi = mid - 1
        else:
            lo = mid
    return lo


def rows(suite: Path):
    for path in sorted(suite.rglob("*.json")):
        for case_id, raw in enumerate(json.loads(path.read_text())):
            count = raw.get("count", 1024)
            probe = normalize({**raw, "count": 1})
            source, output = original_dimensions(probe, count)
            transfers = count * probe["length"]
            pattern_words = sum(max(1, len(probe[key])) for key in
                                ("pattern", "gather", "scatter"))
            schedule_words = output + 1 + transfers if probe["collision_policy"] == "ordered" else 2
            embedded = 8 * (source + output) + 4 * (pattern_words + schedule_words)
            try:
                mapped = normalize(raw)
                if (mapped["src_length"], mapped["dst_length"]) != (source, output):
                    raise AssertionError(f"dimension mismatch: {path}:{case_id}")
                native, limit = "yes", ""
            except ValueError as error:
                native, limit = "no", str(error)
            capacity = largest_fitting_count(raw, count)
            yield dict(
                suite=str(path.relative_to(suite)), case=case_id,
                kind=probe["kind"], pattern_length=probe["length"], count=count,
                transfers=transfers, source_elements=source, output_elements=output,
                source_bytes=8 * source, output_bytes=8 * output,
                embedded_bytes=embedded, native_fit=native, limit=limit,
                largest_single_launch_count=capacity,
                minimum_count_chunks=(count + capacity - 1) // capacity,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    args = parser.parse_args()
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows(args.suite))


if __name__ == "__main__":
    main()
