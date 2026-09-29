#!/usr/bin/env python3
"""Compare Radiance pattern expansion with pinned upstream Spatter parser."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import parse_pattern  # noqa: E402

REVISION = "ec8923711f8dc21eedff7189f12b02eb06845d2f"
EXTRA_PATTERNS = (("MS1:4:2:3", 8, 0),
                  ("LAPLACIAN:2:2:4", 8, 0),
                  ("0,3,1", 8, 0))
FIELDS = ("pattern", "initial_delta", "pattern_size", "length", "final_delta",
          "indices_sha256", "upstream_revision")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--driver", type=Path, default=ROOT / "golden-runs/upstream-golden")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "evaluation/upstream-pattern-parser-results.csv")
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"upstream revision {revision} differs from pinned {REVISION}")
    patterns = set(EXTRA_PATTERNS)
    for suite in (upstream / "standard-suite").rglob("*.json"):
        data = json.loads(suite.read_text())
        if not isinstance(data, list):
            continue
        for case in data:
            for field, delta_field in (("pattern", "delta"),
                                       ("pattern-gather", "delta-gather"),
                                       ("pattern-scatter", "delta-scatter")):
                value = case.get(field)
                if isinstance(value, str):
                    patterns.add((value, case.get(delta_field, 8),
                                  case.get("pattern-size", 0)))
    records = []
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "upstream.bin"
        for pattern, initial_delta, size in sorted(patterns):
            indices, final_delta = parse_pattern(pattern, initial_delta)
            if size:
                indices = indices[:size]
            command = [str(args.driver.resolve()), "--parse-pattern", pattern,
                       str(initial_delta), str(size), str(output)]
            upstream_delta = int(subprocess.check_output(command, text=True).strip())
            expected = b"".join(struct.pack("<I", index) for index in indices)
            actual = output.read_bytes()
            if actual != expected or upstream_delta != final_delta:
                raise ValueError(f"Radiance pattern differs from upstream: {pattern}")
            records.append(dict(pattern=pattern, initial_delta=initial_delta,
                                pattern_size=size, length=len(indices),
                                final_delta=final_delta,
                                indices_sha256=hashlib.sha256(actual).hexdigest(),
                                upstream_revision=REVISION))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as result:
        writer = csv.DictWriter(result, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"matched {len(records)} Radiance patterns against upstream Spatter")


if __name__ == "__main__":
    main()
