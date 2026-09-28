#!/usr/bin/env python3
"""Validate paired current-build RTL/model runs and write a result CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


TOOLS = Path(__file__).resolve().parent


def load_summary():
    spec = importlib.util.spec_from_file_location("spatter_summary", TOOLS / "spatter-summarize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def inspect_run(build_root: Path, rtl_root: Path, model_root: Path,
                name: str, summary) -> list[dict]:
    locations = [build_root / name, rtl_root / name, model_root / name]
    builds, rtl, model = [json.loads((path / "result.json").read_text())
                          for path in locations]
    expected_status = "exploratory" if builds["destination_overlap"] else "passed"
    for path, result in zip(locations[1:], (rtl, model)):
        if result["status"] != expected_status:
            raise ValueError(f"{path}: status {result['status']}, expected {expected_status}")
        if result["suite_sha256"] != builds["suite_sha256"] or result["case"] != builds["case"]:
            raise ValueError(f"{path}: source suite or case differs")
        if result["kernel_source_sha256"] != builds["kernel_source_sha256"]:
            raise ValueError(f"{path}: kernel source differs")
        if result.get("address_plan") != builds["address_plan"]:
            raise ValueError(f"{path}: address plan differs")
    elf_hash = digest(locations[0] / "kernel.soc.elf")
    for path, result in zip(locations[1:], (rtl, model)):
        if result["elf_sha256"] != elf_hash or digest(path / "kernel.soc.elf") != elf_hash:
            raise ValueError(f"{path}: ELF hash differs from the build")
    log = (locations[2] / "cyclotron.log").read_text(errors="replace")
    match = re.search(r"SPATTER_CHECK digest=([0-9a-f]{16}) expected=(\w+) "
                      r"guards_intact=(\w+) nonzero_words=(\d+)", log)
    if not match or match.group(3) != "true" or int(match.group(4)) == 0:
        raise ValueError(f"{locations[2]}: incomplete GPU output readback")
    if not builds["destination_overlap"] and match.group(1) != builds["expected_digest"]:
        raise ValueError(f"{locations[2]}: complete output digest differs")
    return [summary.row_for(path, None) for path in locations[1:]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = load_summary()
    rows = []
    for name in args.names:
        rows.extend(inspect_run(args.build_root, args.rtl_root, args.model_root,
                                name, summary))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(f"validated {len(args.names)} paired RTL/Cyclotron cases: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
