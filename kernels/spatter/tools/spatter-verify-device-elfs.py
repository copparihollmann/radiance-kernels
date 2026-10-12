#!/usr/bin/env python3
"""Confirm the sampled-host rebuild leaves every RV32 device segment unchanged."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import subprocess


SECTION = re.compile(r"\.rv32\.seg(\d+)\s+PROGBITS\s+[0-9a-f]+\s+([0-9a-f]+)\s+([0-9a-f]+)")


def device_segments(elf: Path) -> dict[int, tuple[int, str]]:
    output = subprocess.check_output(["readelf", "-SW", str(elf)], text=True)
    data = elf.read_bytes()
    segments = {}
    for match in SECTION.finditer(output):
        index, offset, size = (int(value, 16) for value in match.groups())
        if index in segments:
            raise ValueError(f"duplicate RV32 segment {index} in {elf}")
        blob = data[offset:offset + size]
        if len(blob) != size:
            raise ValueError(f"truncated RV32 segment {index} in {elf}")
        segments[index] = size, hashlib.sha256(blob).hexdigest()
    if not segments:
        raise ValueError(f"no embedded RV32 segments in {elf}")
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="names present in both build roots")
    parser.add_argument("--original", type=Path, default=Path("spatter-vcs-runs"))
    parser.add_argument("--sampled", type=Path, default=Path("spatter-sampled-builds"))
    parser.add_argument("--original-name", help="original case name for a renamed single build")
    args = parser.parse_args()
    if args.original_name and len(args.names) != 1:
        parser.error("--original-name requires exactly one sampled build")
    for name in args.names:
        original_name = args.original_name or name
        original = device_segments(args.original / original_name / "kernel.soc.elf")
        sampled = device_segments(args.sampled / name / "kernel.soc.elf")
        if original != sampled:
            raise ValueError(f"{name}: RV32 device segments changed")
        print(f"{name}: {len(original)} identical RV32 segments")


if __name__ == "__main__":
    main()
