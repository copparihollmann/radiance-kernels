#!/usr/bin/env python3
"""Export linked RV32 B+Tree addresses for the RV64 checker."""

import subprocess
import sys
from pathlib import Path


nm, elf, output = sys.argv[1:4]
symbols = {}
for line in subprocess.check_output([nm, "-n", elf], text=True).splitlines():
    fields = line.split()
    if len(fields) == 3 and fields[2].startswith("btree_"):
        symbols[fields[2]] = int(fields[0], 16)

required = (
    "btree_point_expected", "btree_range_expected",
    "btree_guard_before", "btree_point_outputs", "btree_guard_middle",
    "btree_range_outputs", "btree_guard_after",
)
missing = [name for name in required if name not in symbols]
if missing:
    raise SystemExit(f"missing symbols in {elf}: {', '.join(missing)}")

defines = ["#pragma once"]
for name in required:
    macro = name.upper() + "_ADDR"
    defines.append(f"#define {macro} 0x{symbols[name]:08x}u")
Path(output).write_text("\n".join(defines) + "\n")
