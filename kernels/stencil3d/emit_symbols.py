#!/usr/bin/env python3
"""Export linked RV32 stencil addresses for the RV64 checker."""

import subprocess
import sys
from pathlib import Path

nm, elf, output = sys.argv[1:4]
symbols = {}
for line in subprocess.check_output([nm, "-n", elf], text=True).splitlines():
    fields = line.split()
    if len(fields) == 3 and fields[2].startswith("stencil3d_"):
        symbols[fields[2]] = int(fields[0], 16)
required = (
    "stencil3d_output", "stencil3d_expected",
    "stencil3d_guard_before", "stencil3d_guard_after",
)
for name in required:
    if name not in symbols:
        raise SystemExit(f"missing {name} in {elf}")
Path(output).write_text(
    "#pragma once\n"
    f"#define STENCIL3D_OUTPUT_ADDR 0x{symbols['stencil3d_output']:08x}u\n"
    f"#define STENCIL3D_EXPECTED_ADDR 0x{symbols['stencil3d_expected']:08x}u\n"
    f"#define STENCIL3D_GUARD_BEFORE_ADDR 0x{symbols['stencil3d_guard_before']:08x}u\n"
    f"#define STENCIL3D_GUARD_AFTER_ADDR 0x{symbols['stencil3d_guard_after']:08x}u\n"
)
