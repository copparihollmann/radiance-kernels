#!/usr/bin/env python3
"""Report which upstream Spatter configurations fit the current Muon mapping."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys


KERNEL = Path(__file__).resolve().parents[1]
MAPPER = KERNEL / "run.py"


def load_mapper():
    sys.path.insert(0, str(KERNEL))
    spec = importlib.util.spec_from_file_location("spatter_mapper", MAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MAPPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_count(mapper, raw: dict, cap: int) -> int:
    """Largest prefix count at or below cap that passes normalization."""
    lo, hi = 0, min(raw.get("count", 1024), cap)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        reduced = dict(raw, count=mid)
        try:
            mapper.normalize(reduced)
        except ValueError:
            hi = mid - 1
        else:
            lo = mid
    return lo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="upstream Spatter JSON file or suite directory")
    parser.add_argument("--max-count", type=int, default=1024,
                        help="upper bound for a scaled prefix, not a benchmark-equivalent result")
    args = parser.parse_args()
    if args.max_count < 1:
        parser.error("--max-count must be positive")
    mapper = load_mapper()
    files = sorted(args.suite.rglob("*.json")) if args.suite.is_dir() else [args.suite]
    writer = csv.writer(sys.stdout)
    writer.writerow(("suite", "case", "kind", "original_count", "native", "native_reason",
                     "fitting_prefix_count", "source_elements", "output_elements"))
    for file in files:
        cases = json.loads(file.read_text())
        for index, raw in enumerate(cases):
            kind = str(raw.get("kernel", "Gather")).lower()
            count = raw.get("count", 1024)
            reason = ""
            try:
                mapper.normalize(raw)
                native = True
            except ValueError as error:
                native = False
                reason = str(error)
            prefix = fit_count(mapper, raw, args.max_count)
            mapped = mapper.normalize(dict(raw, count=prefix)) if prefix else None
            writer.writerow((str(file.relative_to(args.suite) if args.suite.is_dir() else file),
                             index, kind, count, native, reason, prefix,
                             mapped["src_length"] if mapped else "",
                             mapped["dst_length"] if mapped else ""))


if __name__ == "__main__":
    main()
