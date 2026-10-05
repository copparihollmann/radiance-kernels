#!/usr/bin/env python3
"""Put an exact archived RV32 image into a rebuilt RV64 diagnostic host ELF."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


SECTION = re.compile(r"\.rv32\.seg(\d+)\s+PROGBITS\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)")
LOAD = re.compile(r"^\s+LOAD\s+0x[0-9a-f]+\s+(0x[0-9a-f]+)\s+0x[0-9a-f]+\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)", re.M)


def device_image(path: Path) -> tuple[dict[int, tuple[int, int, str]], dict[int, tuple[int, int]]]:
    sections = subprocess.check_output(["readelf", "-SW", str(path)], text=True)
    headers = subprocess.check_output(["readelf", "-lW", str(path)], text=True)
    data = path.read_bytes()
    image = {}
    for match in SECTION.finditer(sections):
        number, address, offset, size = (int(value, 16) for value in match.groups())
        blob = data[offset:offset + size]
        if len(blob) != size or number in image:
            raise ValueError(f"invalid RV32 section {number} in {path}")
        image[number] = (address, size, hashlib.sha256(blob).hexdigest())
    if not image:
        raise ValueError(f"no RV32 sections in {path}")
    loads = {int(address, 16): (int(filesz, 16), int(memsz, 16))
             for address, filesz, memsz in LOAD.findall(headers)
             if int(address, 16) >= 0x100000000}
    if {part[0] for part in image.values()} != set(loads):
        raise ValueError(f"RV32 sections and LOAD headers disagree in {path}")
    return image, loads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path, help="archived failing ELF")
    parser.add_argument("rebuilt", type=Path, help="ELF with the new RV64 host")
    parser.add_argument("output", type=Path, help="new ELF to write")
    parser.add_argument("--objcopy", default="riscv64-unknown-elf-objcopy")
    args = parser.parse_args()
    original, original_loads = device_image(args.original)
    rebuilt, _ = device_image(args.rebuilt)
    if set(original) != set(rebuilt):
        raise ValueError("the two ELFs have different RV32 segment counts")
    if any(original[number][0] != rebuilt[number][0] for number in original):
        raise ValueError("RV32 virtual addresses changed in the rebuilt ELF")
    if args.output.resolve() in (args.original.resolve(), args.rebuilt.resolve()):
        raise ValueError("output must differ from both input ELFs")
    shutil.copy2(args.rebuilt, args.output)
    with tempfile.TemporaryDirectory() as temp:
        for number in sorted(original):
            section = f".rv32.seg{number}"
            blob = Path(temp) / f"seg{number}.bin"
            subprocess.run([args.objcopy, "--dump-section", f"{section}={blob}",
                            str(args.original)], check=True)
            subprocess.run([args.objcopy, "--update-section", f"{section}={blob}",
                            str(args.output)], check=True)
    actual, actual_loads = device_image(args.output)
    if actual != original or actual_loads != original_loads:
        raise ValueError("spliced RV32 image differs from the archived ELF")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"{args.output}: {len(actual)} exact RV32 LOAD segments; SHA-256 {digest}")


if __name__ == "__main__":
    main()
