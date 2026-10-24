#!/usr/bin/env python3
"""Assemble a hash-pinned model/campaign manifest from existing ELF builds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[3]
PASSTHROUGH = (
    "case", "collision_policy", "muon_warps_per_core",
    "source_pad_cache_lines", "output_pad_cache_lines",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def operation(result: dict) -> str | None:
    kind = result.get("kind")
    if kind == "scatter":
        return f"scatter-{result.get('collision_policy', 'parallel')}"
    return kind


def assemble(root: Path, family: str) -> dict:
    rows = []
    keys = set()
    source_revisions = set()
    source_hashes = set()
    for result_path in sorted(root.rglob("result.json")):
        elf = result_path.parent / "kernel.soc.elf"
        if not elf.is_file():
            continue
        result = json.loads(result_path.read_text())
        if result.get("status") != "built":
            raise ValueError(f"build is not successful: {result_path}")
        name = result_path.parent.name
        key = (name, result["loader"], int(result["sm_count"]))
        if key in keys:
            raise ValueError(f"duplicate workload identity: {key}")
        keys.add(key)
        row = {
            "name": name,
            "family": family,
            "sm_count": int(result["sm_count"]),
            "loader": result["loader"],
            "elf": str(elf.resolve()),
            "elf_sha256": sha256(elf),
            "bytes": elf.stat().st_size,
            "correctness": result.get("correctness"),
            "status": "built",
        }
        op = operation(result)
        if op is not None:
            row["operation"] = op
        for key in PASSTHROUGH:
            if key in result:
                row[key] = result[key]
        rows.append(row)
        if result.get("kernel_revision"):
            source_revisions.add(result["kernel_revision"])
        if result.get("kernel_source_sha256"):
            source_hashes.add(result["kernel_source_sha256"])
    if not rows:
        raise ValueError(f"no completed ELF builds found under {root}")
    revision = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain=v1"], text=True,
    ))
    return {
        "schema": "radiance-fpga-1a-existing-builds-v1",
        "current_source_revision": revision,
        "current_source_dirty": dirty,
        "artifact_base_revisions": sorted(source_revisions),
        "artifact_source_hashes": sorted(source_hashes),
        "no_vpu": True,
        "hardware_rebuilt": False,
        "sms": sorted({row["sm_count"] for row in rows}),
        "loaders": sorted({row["loader"] for row in rows}),
        "workloads": sorted(rows, key=lambda row: row["name"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", required=True,
                        choices=("stream", "spatter", "xrage", "fp8_gemm"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = assemble(args.root.resolve(), args.family)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"recorded {len(manifest['workloads'])} builds in {args.out}")


if __name__ == "__main__":
    main()
