#!/usr/bin/env python3
"""Export linked RV32 GEMV addresses for the RV64 host."""

import subprocess
import sys
from pathlib import Path


nm, elf, output = sys.argv[1:4]
symbols = {}
for line in subprocess.check_output([nm, "-n", elf], text=True).splitlines():
    fields = line.split()
    if len(fields) == 3 and fields[2].startswith("gemv_"):
        symbols[fields[2]] = int(fields[0], 16)
for name in ("gemv_y", "gemv_guard_before", "gemv_guard_after"):
    if name not in symbols:
        raise SystemExit(f"missing {name} in {elf}")
Path(output).write_text(
    "#pragma once\n"
    f"#define GEMV_OUTPUT_ADDR 0x{symbols['gemv_y']:08x}u\n"
    f"#define GEMV_GUARD_BEFORE_ADDR 0x{symbols['gemv_guard_before']:08x}u\n"
    f"#define GEMV_GUARD_AFTER_ADDR 0x{symbols['gemv_guard_after']:08x}u\n"
)
