#!/usr/bin/env python3
"""Export linked RV32 STREAM Dot addresses for the RV64 checker."""

import subprocess
import sys
from pathlib import Path

nm, elf, output = sys.argv[1:4]
symbols = {}
for line in subprocess.check_output([nm, "-n", elf], text=True).splitlines():
    fields = line.split()
    if len(fields) == 3 and fields[2].startswith("stream_dot_"):
        symbols[fields[2]] = int(fields[0], 16)
required = ("stream_dot_partial", "stream_dot_guard_before",
            "stream_dot_guard_after")
for name in required:
    if name not in symbols:
        raise SystemExit(f"missing {name} in {elf}")
Path(output).write_text(
    "#pragma once\n"
    f"#define STREAM_DOT_PARTIAL_ADDR 0x{symbols['stream_dot_partial']:08x}u\n"
    f"#define STREAM_DOT_GUARD_BEFORE_ADDR 0x{symbols['stream_dot_guard_before']:08x}u\n"
    f"#define STREAM_DOT_GUARD_AFTER_ADDR 0x{symbols['stream_dot_guard_after']:08x}u\n"
)
