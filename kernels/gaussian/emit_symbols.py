#!/usr/bin/env python3
"""Export linked RV32 Gaussian addresses for the RV64 checker."""

from pathlib import Path
import subprocess
import sys


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: emit_symbols.py NM ELF OUTPUT")
    nm, elf, output = sys.argv[1:]
    symbols = {}
    for line in subprocess.check_output([nm, "-n", elf], text=True).splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2].startswith("gaussian_"):
            symbols[fields[2]] = int(fields[0], 16)
    required = (
        "gaussian_a", "gaussian_b", "gaussian_m",
        "gaussian_expected_a", "gaussian_expected_b", "gaussian_expected_m",
        "gaussian_model_a", "gaussian_model_b", "gaussian_model_m",
        "gaussian_owners", "gaussian_guard_before", "gaussian_guard_middle",
        "gaussian_guard_model", "gaussian_guard_after",
    )
    missing = [name for name in required if name not in symbols]
    if missing:
        raise SystemExit(f"missing symbols in {elf}: {', '.join(missing)}")
    lines = ["#pragma once"]
    for name in required:
        lines.append(f"#define {name.upper()}_ADDR 0x{symbols[name]:08x}u")
    Path(output).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
