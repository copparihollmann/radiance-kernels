#!/usr/bin/env python3
"""Export linked RV32 LBM addresses for the RV64 checker."""

import subprocess
import sys
from pathlib import Path


if len(sys.argv) != 4:
    raise SystemExit("usage: emit_symbols.py LLVM_NM ELF OUTPUT")

nm, elf, output = sys.argv[1:4]
symbols: dict[str, int] = {}
for line in subprocess.check_output([nm, "-n", elf], text=True).splitlines():
    fields = line.split()
    if len(fields) == 3 and fields[2].startswith("lbm_"):
        symbols[fields[2]] = int(fields[0], 16)

required = (
    "lbm_output_raw",
    "lbm_expected_populations",
    "lbm_gold_velocity",
    "lbm_guard_before",
    "lbm_guard_after",
)
for name in required:
    if name not in symbols:
        raise SystemExit(f"missing {name} in {elf}")

Path(output).write_text(
    "#pragma once\n"
    f"#define LBM_OUTPUT_ADDR 0x{symbols['lbm_output_raw']:08x}u\n"
    f"#define LBM_EXPECTED_POPULATIONS_ADDR "
    f"0x{symbols['lbm_expected_populations']:08x}u\n"
    f"#define LBM_GOLD_VELOCITY_ADDR 0x{symbols['lbm_gold_velocity']:08x}u\n"
    f"#define LBM_GUARD_BEFORE_ADDR 0x{symbols['lbm_guard_before']:08x}u\n"
    f"#define LBM_GUARD_AFTER_ADDR 0x{symbols['lbm_guard_after']:08x}u\n"
)
