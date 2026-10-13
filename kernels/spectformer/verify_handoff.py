#!/usr/bin/env python3
"""Verify every file in a copied SpectFormer device handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path) -> int:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != "spectformer.fpga.handoff.v1":
        raise ValueError("unknown handoff schema")
    checked = 0
    seen = set()
    for case in manifest["cases"]:
        name = case["name"]
        if name in seen or Path(name).name != name:
            raise ValueError("duplicate or unsafe case name")
        seen.add(name)
        case_dir = root / name
        if json.loads((case_dir / "load-map.json").read_text()) != case:
            raise ValueError(f"{name}: load map differs from manifest")
        for filename, record in case["files"].items():
            if Path(filename).name != filename:
                raise ValueError(f"{name}: unsafe filename")
            path = case_dir / filename
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise ValueError(f"{name}/{filename}: size or hash mismatch")
            checked += 1
        if case["output"]["bytes"] != case["output"]["elements"] * 4:
            raise ValueError(f"{name}: invalid FP32 output length")
        if "reference.bin" in case["files"] and (
                case["files"]["reference.bin"]["bytes"] != case["output"]["bytes"]):
            raise ValueError(f"{name}: reference length differs from output")
        loads = case["gmem_loads"]
        intervals = sorted((int(item["gpu_address"], 16),
                            int(item["gpu_address"], 16) + item["bytes"])
                           for item in loads)
        if any(upper > next_lower for (_, upper), (next_lower, _) in
               zip(intervals, intervals[1:])):
            raise ValueError(f"{name}: overlapping memory segments")
        for item in loads:
            if case["files"][item["file"]]["bytes"] != item["bytes"]:
                raise ValueError(f"{name}: memory load length differs from file")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    checked = verify(args.root)
    print(f"verified {checked} files across {len(json.loads((args.root / 'manifest.json').read_text())['cases'])} cases")


if __name__ == "__main__":
    main()
