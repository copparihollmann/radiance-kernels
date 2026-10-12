#!/usr/bin/env python3
"""Export linked RV32 data addresses for the RV64 validation host."""

import subprocess
import sys
from pathlib import Path

nm, elf, output = sys.argv[1:4]
listing = subprocess.check_output([nm, "-n", elf], text=True)
symbols = {}
for line in listing.splitlines():
    fields = line.split()
    if len(fields) == 3 and fields[2].startswith("spatter_"):
        symbols[fields[2]] = int(fields[0], 16)
needed = (
    "spatter_guard_before", "spatter_guard_after", "spatter_dense",
    "spatter_sparse", "spatter_sparse_scatter",
)
config = (Path(output).parent / "config.h").read_text()
kind = int(next(line.split()[-1] for line in config.splitlines() if line.startswith("#define SPATTER_KIND ")))
name = "spatter_dense" if kind in (0, 3) else ("spatter_sparse_scatter" if kind == 2 else "spatter_sparse")
for key in (name, "spatter_guard_before", "spatter_guard_after"):
    if key not in symbols:
        raise SystemExit(f"missing {key} in {elf}")
Path(output).write_text(
    "#pragma once\n"
    f"#define SPATTER_OUTPUT_ADDR 0x{symbols[name]:08x}u\n"
    f"#define SPATTER_GUARD_BEFORE_ADDR 0x{symbols['spatter_guard_before']:08x}u\n"
    f"#define SPATTER_GUARD_AFTER_ADDR 0x{symbols['spatter_guard_after']:08x}u\n"
)
