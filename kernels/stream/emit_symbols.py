#!/usr/bin/env python3
"""Export linked RV32 STREAM addresses for the RV64 host."""

import subprocess
import sys
from pathlib import Path

nm, elf, output = sys.argv[1:4]
symbols = {}
for line in subprocess.check_output([nm, "-n", elf], text=True).splitlines():
    fields = line.split()
    if len(fields) == 3 and fields[2].startswith("stream_"):
        symbols[fields[2]] = int(fields[0], 16)
config = (Path(output).parent / "config.h").read_text().splitlines()
kind = int(next(line.split()[-1] for line in config
                if line.startswith("#define STREAM_KIND ")))
target = ("stream_c", "stream_b", "stream_c", "stream_a")[kind]
for name in (target, "stream_guard_before", "stream_guard_after"):
    if name not in symbols:
        raise SystemExit(f"missing {name} in {elf}")
Path(output).write_text(
    "#pragma once\n"
    f"#define STREAM_OUTPUT_ADDR 0x{symbols[target]:08x}u\n"
    f"#define STREAM_GUARD_BEFORE_ADDR 0x{symbols['stream_guard_before']:08x}u\n"
    f"#define STREAM_GUARD_AFTER_ADDR 0x{symbols['stream_guard_after']:08x}u\n"
)
