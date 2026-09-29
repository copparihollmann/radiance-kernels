#!/usr/bin/env python3
"""Verify locally retained Spatter golden files and published comparison rows."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("upstream_golden", TOOLS / "spatter-upstream-golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--model-root", type=Path, default=ROOT / "runs/model")
    parser.add_argument("--golden-root", type=Path, default=ROOT / "golden-runs")
    args = parser.parse_args()
    artifact_rows = rows(ROOT / "evaluation/upstream-golden-artifacts.csv")
    expected = {row["path"]: row for row in artifact_rows}
    if len(expected) != len(artifact_rows):
        raise ValueError("duplicate golden artifact path")
    actual = {path.relative_to(args.golden_root).as_posix(): path
              for path in args.golden_root.rglob("*") if path.is_file()}
    if actual.keys() != expected.keys():
        raise ValueError(f"golden artifact set differs: "
                         f"missing={sorted(expected.keys() - actual.keys())}, "
                         f"new={sorted(actual.keys() - expected.keys())}")
    for name, path in actual.items():
        row = expected[name]
        if path.stat().st_size != int(row["bytes"]) or golden.sha256(path) != row["sha256"]:
            raise ValueError(f"golden artifact changed: {name}")

    source_hash = None
    if args.upstream:
        upstream = args.upstream.resolve()
        revision = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        if revision != golden.UPSTREAM_REVISION:
            raise ValueError("upstream Spatter revision changed")
        source_hash = golden.upstream_source_hash(upstream)

    seen = set()
    count = 0
    for table in ("upstream-golden-results.csv", "upstream-golden-smoke-results.csv",
                  "upstream-golden-scaled-results.csv"):
        for row in rows(ROOT / "evaluation" / table):
            name = row["run"]
            if name in seen:
                raise ValueError(f"duplicate golden run: {name}")
            seen.add(name)
            record = json.loads((args.golden_root / name / "result.json").read_text())
            if {key: str(value) for key, value in record.items()} != row:
                raise ValueError(f"golden table differs from record: {name}")
            build = json.loads((args.source_root / name / "result.json").read_text())
            model = json.loads((args.model_root / name / "result.json").read_text())
            if (row["status"] != "passed" or model["status"] != "passed" or
                    row["upstream_revision"] != golden.UPSTREAM_REVISION or
                    row["suite_sha256"] != build["suite_sha256"] or
                    row["radiance_elf_sha256"] != golden.sha256(
                        args.source_root / name / "kernel.soc.elf") or
                    row["radiance_elf_sha256"] != model["elf_sha256"] or
                    row["golden_digest"] != row["radiance_expected_digest"] or
                    row["golden_digest"] != model["output_digest"] or
                    row["golden_digest"] != golden.fnv_file(
                        args.golden_root / name / "output.bin") or
                    (source_hash and row["upstream_source_sha256"] != source_hash)):
                raise ValueError(f"golden check differs from Radiance/model: {name}")
            count += 1
    print(f"verified {count} upstream golden comparisons and "
          f"{len(artifact_rows)} local golden artifact hashes")


if __name__ == "__main__":
    main()
