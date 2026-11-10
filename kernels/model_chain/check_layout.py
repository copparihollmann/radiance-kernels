#!/usr/bin/env python3
"""Check that writable stage buffers and launch arguments own cache lines."""

import argparse
import re
import subprocess
from pathlib import Path


def symbols(elf: Path, nm: Path) -> dict[str, tuple[int, int]]:
    listing = subprocess.check_output([str(nm), "-S", str(elf)], text=True)
    found = {}
    for line in listing.splitlines():
        match = re.match(r"([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+\w\s+(\S+)", line)
        if match:
            found[match.group(3)] = (int(match.group(1), 16),
                                     int(match.group(2), 16))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=Path)
    parser.add_argument("--nm", type=Path, required=True)
    parser.add_argument("--generated", action="store_true",
                        help="check every generated tensor and parameter buffer")
    args = parser.parse_args()
    entries = symbols(args.elf, args.nm)
    if args.generated:
        buffers = {name: position for name, position in entries.items()
                   if name.startswith("v_")}
        if not buffers:
            raise SystemExit("no generated buffers found")
        for name, (address, length) in buffers.items():
            if address % 64 or length % 64:
                raise SystemExit(f"{name} is not a whole number of cache lines")
        print(f"{len(buffers)} generated buffers occupy disjoint whole cache lines")
        return
    for name in ("normalized_raw", "projected_raw", "output_raw"):
        address, length = entries[name]
        if address % 64 or length % 64:
            raise SystemExit(f"{name} is not a whole number of cache lines")
    output, size = entries["output_raw"]
    launch, _ = entries["_ZL4args"]
    if launch % 64:
        raise SystemExit("launch arguments are not cache-line aligned")
    if (output + size - 1) // 64 == launch // 64:
        raise SystemExit("launch arguments overlap the output's cache line")
    print(f"output=[{output:#x},{output + size:#x}) args={launch:#x}: disjoint cache lines")


if __name__ == "__main__":
    main()
