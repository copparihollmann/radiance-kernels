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
        if case["simulator_status"] == "passed":
            raw_path = case_dir / "simulator-result.json"
            raw = json.loads(raw_path.read_text())
            if "simulator-verification.json" in case["files"]:
                verification = json.loads((case_dir / "simulator-verification.json").read_text())
                if (verification.get("status") != "passed" or
                        verification.get("reference_checked") is not True or
                        verification.get("all_finite") is not True or
                        verification.get("out_of_tolerance") != 0 or
                        verification.get("run_result_sha256") != sha256(raw_path) or
                        raw.get("status") not in ("passed", "completed_unverified") or
                        raw.get("reference_checked") is not False or
                        raw.get("output_kind") != case["output_kind"] or
                        verification.get("scope") != case["output_kind"] or
                        verification.get("output_count") != case["output"]["elements"] or
                        any(verification.get(key) != raw.get(key) for key in
                            ("elf_sha256", "weights_sha256", "input_sha256")) or
                        verification.get("reference_sha256") !=
                        case["files"]["reference.bin"]["sha256"]):
                    raise ValueError(f"{name}: invalid checked simulator pass")
            else:
                comparison = raw.get("output_comparison")
                if (raw.get("status") != "passed" or
                        raw.get("reference_checked") is not True or
                        not isinstance(comparison, dict) or
                        comparison.get("all_close") is not True):
                    raise ValueError(f"{name}: simulator pass lacks a reference check")
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
