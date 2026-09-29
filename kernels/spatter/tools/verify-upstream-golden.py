#!/usr/bin/env python3
"""Verify locally retained Spatter golden files and published comparison rows."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
EXPECTED_RUNS = {
    "upstream-golden-results.csv": {
        *(f"rebuild-gpu-stream-{index}" for index in range(5)),
        "xrage5", "xrage9-ordered", "lulesh-gather", "lulesh-scatter-ordered",
    },
    "upstream-golden-smoke-results.csv": {
        "smoke-1", "smoke-2", "smoke-3", "smoke-4", "ordered-overlap",
    },
    "upstream-golden-scaled-results.csv": {"amg-gpu-scaled-1024"},
}
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
        audited_suite = subprocess.check_output([
            sys.executable, str(TOOLS / "spatter-audit-suite.py"),
            str(upstream / "standard-suite"), "--max-count", "1024",
        ])
        published_suite = (ROOT / "evaluation/standard-suite-coverage.csv").read_bytes()
        if audited_suite != published_suite:
            raise ValueError("standard-suite coverage differs from pinned upstream Spatter")

    seen = set()
    count = 0
    for table, expected_runs in EXPECTED_RUNS.items():
        table_rows = rows(ROOT / "evaluation" / table)
        actual_runs = {row["run"] for row in table_rows}
        if actual_runs != expected_runs or len(table_rows) != len(expected_runs):
            raise ValueError(f"{table}: golden comparison coverage differs; "
                             f"missing={sorted(expected_runs - actual_runs)}, "
                             f"extra={sorted(actual_runs - expected_runs)}")
        for row in table_rows:
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
    composition_rows = rows(ROOT / "evaluation/upstream-golden-composition-results.csv")
    if len(composition_rows) != 1:
        raise ValueError("expected one upstream full-size composition comparison")
    row = composition_rows[0]
    name = "chain-gpu-stream-no-fence"
    fused_name = "composed-gpu-stream-current"
    raw = args.golden_root / "composition-fullsize"
    record = json.loads((raw / "result.json").read_text())
    chain_build = json.loads((args.source_root / name / "result.json").read_text())
    chain_model = json.loads((args.model_root / name / "result.json").read_text())
    fused_build = json.loads((args.source_root / fused_name / "result.json").read_text())
    fused_model = json.loads((args.model_root / fused_name / "result.json").read_text())
    digests = {row["golden_digest"], row["chain_expected_digest"],
               row["chain_model_digest"], row["fused_expected_digest"],
               row["fused_model_digest"], chain_build["expected_digest"],
               chain_model["output_digest"], fused_build["expected_digest"],
               fused_model["output_digest"], golden.fnv_file(raw / "output.bin")}
    if (row["run"] != name or row["status"] != "passed" or
            row["upstream_revision"] != golden.UPSTREAM_REVISION or
            row["driver_script_sha256"] != golden.sha256(
                TOOLS / "spatter-upstream-composition-golden.py") or
            (source_hash and row["upstream_source_sha256"] != source_hash) or
            {key: str(value) for key, value in record.items()} != row or
            chain_model["status"] != "passed" or fused_model["status"] != "passed" or
            len(digests) != 1 or
            row["intermediate_digest"] != golden.fnv_file(raw / "intermediate.bin") or
            row["original_suite_sha256"] != golden.sha256(
                ROOT / "inputs/standard-suite/basic-tests/gpu-stream.json") or
            row["fused_suite_sha256"] != golden.sha256(
                ROOT / "inputs/composed-gpu-stream.json") or
            row["chain_elf_sha256"] != golden.sha256(
                args.source_root / name / "kernel.soc.elf") or
            row["chain_elf_sha256"] != chain_model["elf_sha256"] or
            row["fused_elf_sha256"] != golden.sha256(
                args.source_root / fused_name / "kernel.soc.elf") or
            row["fused_elf_sha256"] != fused_model["elf_sha256"] or
            int(row["count"]) != chain_build["count"] or
            int(row["pattern_length"]) != chain_build["pattern_length"] or
            int(row["intermediate_elements"]) != chain_build["intermediate_elements"]):
        raise ValueError("upstream composition golden differs from Radiance runs")
    for filename, bytes_field, hash_field in (
            ("source.bin", "source_bytes", "source_sha256"),
            ("intermediate.bin", "intermediate_bytes", "intermediate_sha256"),
            ("output.bin", "golden_output_bytes", "golden_output_sha256")):
        path = raw / filename
        if path.stat().st_size != int(row[bytes_field]) or golden.sha256(path) != row[hash_field]:
            raise ValueError(f"upstream composition {filename} differs from manifest")
    for filename, hash_field in (("gather-pattern.bin", "gather_pattern_sha256"),
                                 ("scatter-pattern.bin", "scatter_pattern_sha256")):
        if golden.sha256(raw / filename) != row[hash_field]:
            raise ValueError(f"upstream composition {filename} differs from manifest")
    count += 1
    coverage = "; pinned standard-suite coverage matched" if args.upstream else ""
    print(f"verified {count} upstream golden comparisons and "
          f"{len(artifact_rows)} local golden artifact hashes{coverage}")


if __name__ == "__main__":
    main()
