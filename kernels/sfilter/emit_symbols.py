#!/usr/bin/env python3
"""Export linked RV32 sfilter addresses for the RV64 checker."""

import subprocess
import sys
from pathlib import Path


nm, elf, output = sys.argv[1:4]
symbols: dict[str, int] = {}
for line in subprocess.check_output([nm, "-n", elf], text=True).splitlines():
    fields = line.split()
    if len(fields) == 3 and fields[2].startswith("sfilter_"):
        symbols[fields[2]] = int(fields[0], 16)

required = (
    "sfilter_output",
    "sfilter_expected",
    "sfilter_guard_before",
    "sfilter_guard_after",
)
for name in required:
    if name not in symbols:
        raise SystemExit(f"missing {name} in {elf}")

Path(output).write_text(
    "#pragma once\n"
    f"#define SFILTER_OUTPUT_ADDR 0x{symbols['sfilter_output']:08x}u\n"
    f"#define SFILTER_EXPECTED_ADDR 0x{symbols['sfilter_expected']:08x}u\n"
    f"#define SFILTER_GUARD_BEFORE_ADDR 0x{symbols['sfilter_guard_before']:08x}u\n"
    f"#define SFILTER_GUARD_AFTER_ADDR 0x{symbols['sfilter_guard_after']:08x}u\n"
)
