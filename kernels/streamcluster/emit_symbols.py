#!/usr/bin/env python3
"""Export linked RV32 StreamCluster addresses for the RV64 checker."""

import subprocess
import sys
from pathlib import Path


nm, elf, output = sys.argv[1:4]
symbols: dict[str, int] = {}
for line in subprocess.check_output([nm, "-n", elf], text=True).splitlines():
    fields = line.split()
    if len(fields) == 3 and fields[2].startswith("streamcluster_"):
        symbols[fields[2]] = int(fields[0], 16)

required = (
    "streamcluster_work_guard_before",
    "streamcluster_work_output",
    "streamcluster_work_guard_after",
    "streamcluster_work_expected",
    "streamcluster_switch_guard_before",
    "streamcluster_switch_output",
    "streamcluster_switch_guard_after",
    "streamcluster_switch_expected",
)
for name in required:
    if name not in symbols:
        raise SystemExit(f"missing {name} in {elf}")

lines = ["#pragma once"]
for name in required:
    macro = name.upper() + "_ADDR"
    lines.append(f"#define {macro} 0x{symbols[name]:08x}u")
Path(output).write_text("\n".join(lines) + "\n")
