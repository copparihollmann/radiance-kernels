#!/usr/bin/env python3
"""Export linked RV32 Hotspot3D addresses for the RV64 checker."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: emit_symbols.py NM ELF OUTPUT")
    nm, elf, output = sys.argv[1:]
    symbols: dict[str, int] = {}
    for line in subprocess.check_output([nm, "-n", elf], text=True).splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2].startswith("hotspot3d_"):
            symbols[fields[2]] = int(fields[0], 16)
    required = (
        "hotspot3d_output", "hotspot3d_expected", "hotspot3d_owners",
        "hotspot3d_guard_before", "hotspot3d_guard_middle_a",
        "hotspot3d_guard_middle_b", "hotspot3d_guard_after",
    )
    missing = [name for name in required if name not in symbols]
    if missing:
        raise SystemExit(f"missing symbols in {elf}: {', '.join(missing)}")
    lines = ["#pragma once"]
    for name in required:
        macro = name.upper() + "_ADDR"
        lines.append(f"#define {macro} 0x{symbols[name]:08x}u")
    Path(output).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
