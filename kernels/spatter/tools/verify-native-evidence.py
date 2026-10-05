#!/usr/bin/env python3
"""Verify retained native Spatter inputs, golden arrays, and source snapshots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
UPSTREAM_REVISION = "ec8923711f8dc21eedff7189f12b02eb06845d2f"
SOURCE_FILES = ("run.py", "plan.py", "Makefile", "kernel.cpp", "spatter_ops.hpp",
                "host.cpp", "emit_symbols.py", "cyclotron-no-trace.patch")
sys.path.insert(0, str(ROOT))
from run import FNV_OFFSET, FNV_PRIME, MASK64  # noqa: E402


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_digest(path: Path) -> str:
    import struct
    digest = FNV_OFFSET
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 65536), b""):
            assert len(block) % 8 == 0
            for (value,) in struct.iter_unpack("<Q", block):
                for word in (value & 0xFFFFFFFF, value >> 32):
                    digest = ((digest ^ word) * FNV_PRIME) & MASK64
    return f"{digest:016x}"


def verify_suite(upstream: Path) -> None:
    local = ROOT / "inputs/standard-suite"
    manifest = rows(local / "manifest.csv")
    assert len(manifest) == 13
    assert {r["input"] for r in manifest} == {
        p.relative_to(local).as_posix() for p in local.rglob("*.json")}
    for record in manifest:
        name = record["input"]
        path = local / name
        assert record["upstream_revision"] == UPSTREAM_REVISION
        assert path.read_bytes() == (upstream / "standard-suite" / name).read_bytes()
        assert path.stat().st_size == int(record["bytes"])
        assert sha256(path) == record["sha256"]
    for name in ("LICENSE", "COPYING"):
        assert (local / name).read_bytes() == (upstream / name).read_bytes()
    footprint = subprocess.check_output([
        "python3", str(ROOT / "tools/spatter-native-footprint.py"), str(local)])
    assert footprint == (ROOT / "evaluation/native-workload-footprint.csv").read_bytes()
    assert len(rows(ROOT / "evaluation/native-workload-footprint.csv")) == 114


def verify_golden(table: Path, build_root: Path, golden_root: Path) -> int:
    records = rows(table)
    for record in records:
        name = record["run"]
        build = json.loads((build_root / name / "result.json").read_text())
        golden = golden_root / name
        assert record["status"] == "passed"
        assert record["upstream_revision"] == UPSTREAM_REVISION
        assert record["golden_digest"] == record["radiance_expected_digest"]
        assert build["expected_digest"] == record["golden_digest"]
        assert str(build["count"]) == record["count"]
        assert record["suite_sha256"] == build["suite_sha256"]
        assert sha256(build_root / name / "kernel.soc.elf") == record["radiance_elf_sha256"]
        assert sha256(golden / "source.bin") == record["source_sha256"]
        assert sha256(golden / "output.bin") == record["golden_output_sha256"]
        assert output_digest(golden / "output.bin") == record["golden_digest"]
        assert (golden / "source.bin").stat().st_size == int(record["source_bytes"])
        assert (golden / "output.bin").stat().st_size == int(record["golden_output_bytes"])
    return len(records)


def verify_snapshots(inputs: Path) -> None:
    manifest = rows(REPO / "kernels/evaluation/firesim/source-snapshot-manifest.csv")
    assert len(manifest) == 5
    for record in manifest:
        directory = inputs / "source-snapshots" / record["source_hash"]
        digest = hashlib.sha256()
        total = 0
        for name in SOURCE_FILES:
            data = (directory / name).read_bytes()
            digest.update(name.encode())
            digest.update(data)
            total += len(data)
        assert digest.hexdigest() == record["source_hash"]
        assert total == int(record["total_bytes"])
        assert len(SOURCE_FILES) == int(record["file_count"])


def verify_full_amg(inputs: Path) -> None:
    plan_path = ROOT / "evaluation/amg-gpu-chunk-plan.json"
    plan = json.loads(plan_path.read_text())
    result = json.loads((ROOT / "evaluation/amg-gpu-full-upstream.json").read_text())
    raw = inputs / "amg-gpu-full-golden"
    assert result["upstream_revision"] == UPSTREAM_REVISION
    assert result["status"] == "passed"
    assert result["suite_sha256"] == plan["suite_sha256"]
    assert result["plan_sha256"] == sha256(plan_path)
    assert result["complete_output_digest"] == plan["expected_full_digest"]
    assert result["complete_output_digest"] == plan["chunks"][-1]["expected_digest"]
    assert result["source_bytes"] == plan["original_source_bytes"]
    assert result["original_count"] == plan["original_count"]
    for name, field in (("source.bin", "source_sha256"),
                        ("output.bin", "output_sha256")):
        assert sha256(raw / name) == result[field]
    assert sha256(inputs / "amg-gpu-native-golden/upstream-golden") == result["upstream_driver_sha256"]
    assert output_digest(raw / "output.bin") == result["complete_output_digest"]
    assert (raw / "output.bin").stat().st_size == result["output_bytes"]
    for artifact in rows(ROOT / "evaluation/amg-gpu-full-artifacts.csv"):
        path = raw / artifact["path"]
        assert path.stat().st_size == int(artifact["bytes"])
        assert sha256(path) == artifact["sha256"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--inputs-root", type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    inputs = args.inputs_root.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    assert revision == UPSTREAM_REVISION
    verify_suite(upstream)
    count = 0
    for table, build, golden in (
        ("native-upstream-golden.csv", "native", "native-golden"),
        ("chunk-smoke-upstream.csv", "chunk-smoke", "chunk-smoke-golden"),
        ("amg-gpu-chunks-upstream.csv", "amg-gpu-native-chunks", "amg-gpu-native-golden"),
    ):
        count += verify_golden(ROOT / "evaluation" / table, inputs / build, inputs / golden)
    verify_snapshots(inputs)
    verify_full_amg(inputs)
    print(f"verified 13 exact upstream decks, 114 footprints, {count} golden cases, "
          "5 source snapshots, and the original-count GPU AMG output")


if __name__ == "__main__":
    main()
