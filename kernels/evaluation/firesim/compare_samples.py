#!/usr/bin/env python3
"""Compare a diagnostic U250 sample dump with an upstream serial output array."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


SAMPLE = re.compile(r"^(\d+) (\d+) (0x[0-9a-f]{8}) (0x[0-9a-f]{8})$")
MASK64 = (1 << 64) - 1


def digest(values: list[int]) -> str:
    result = 0xCBF29CE484222325
    for value in values:
        for word in (value & 0xFFFFFFFF, value >> 32):
            result = ((result ^ word) * 0x100000001B3) & MASK64
    return f"{result:016x}"


def compare(uart: Path, golden: Path) -> dict:
    output = golden.read_bytes()
    if not output or len(output) % 8:
        raise ValueError("golden output must contain nonempty 64-bit words")
    lines = uart.read_text(errors="replace").replace("\r", "").splitlines()
    diagnostic = [line for line in lines if line.startswith("SPATTER_DIAG guards=")]
    if len(diagnostic) != 1:
        raise ValueError(f"expected one diagnostic header, found {len(diagnostic)}")
    sample_lines = [SAMPLE.fullmatch(line) for line in lines]
    samples = [match for match in sample_lines if match is not None]
    if len(samples) != 64:
        raise ValueError(f"expected 64 sample records, found {len(samples)}")
    n = len(output) // 8
    actual, expected, mismatches = [], [], []
    for index, match in enumerate(samples):
        sample_id, position = int(match[1]), int(match[2])
        if sample_id != index or position != index * (n - 1) // 63:
            raise ValueError(f"unexpected sample index or position at row {index}")
        got = int(match[3], 16) | (int(match[4], 16) << 32)
        want = struct.unpack_from("<Q", output, position * 8)[0]
        actual.append(got)
        expected.append(want)
        if got != want:
            mismatches.append({"sample": index, "position": position,
                               "actual": f"{got:016x}", "expected": f"{want:016x}"})
    return {
        "uart": str(uart.resolve()),
        "uart_sha256": hashlib.sha256(uart.read_bytes()).hexdigest(),
        "golden": str(golden.resolve()),
        "golden_sha256": hashlib.sha256(output).hexdigest(),
        "output_elements": n,
        "guard_status": diagnostic[0].split(" guards=", 1)[1].split()[0],
        "actual_sample_digest": digest(actual),
        "expected_sample_digest": digest(expected),
        "matched_samples": 64 - len(mismatches),
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uart", type=Path)
    parser.add_argument("golden", type=Path)
    parser.add_argument("--output", type=Path, help="write complete comparison as JSON")
    args = parser.parse_args()
    result = compare(args.uart, args.golden)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['matched_samples']}/64 samples match; guards={result['guard_status']}; "
          f"actual digest={result['actual_sample_digest']}; "
          f"expected digest={result['expected_sample_digest']}")
    for mismatch in result["mismatches"][:8]:
        print(mismatch)


if __name__ == "__main__":
    main()
