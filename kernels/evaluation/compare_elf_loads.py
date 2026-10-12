#!/usr/bin/env python3
"""Compare initialized PT_LOAD bytes of two fused RV64/RV32 kernel ELFs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct


PT_LOAD = 1
PROGRAM_HEADER = struct.Struct("<IIQQQQQQ")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_segments(data: bytes) -> list[dict]:
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected a little-endian ELF64 file")
    offset = struct.unpack_from("<Q", data, 32)[0]
    entry_size, count = struct.unpack_from("<HH", data, 54)
    if entry_size < PROGRAM_HEADER.size:
        raise ValueError("program header is too small")
    segments = []
    for index in range(count):
        (kind, flags, file_offset, virtual, _physical, file_size,
         memory_size, alignment) = PROGRAM_HEADER.unpack_from(
            data, offset + index * entry_size)
        if kind != PT_LOAD:
            continue
        if file_size > memory_size or file_offset + file_size > len(data):
            raise ValueError("invalid PT_LOAD size")
        initialized = (data[file_offset:file_offset + file_size] +
                       bytes(memory_size - file_size))
        segments.append({"address": virtual, "flags": flags,
                         "alignment": alignment, "file_size": file_size,
                         "memory_size": memory_size, "initialized": initialized})
    return sorted(segments, key=lambda segment: segment["address"])


def compare(old: Path, new: Path) -> dict:
    old_data, new_data = old.read_bytes(), new.read_bytes()
    left, right = load_segments(old_data), load_segments(new_data)
    if len(left) != len(right):
        raise ValueError("PT_LOAD segment counts differ")
    rows = []
    compatible = True
    for index, (a, b) in enumerate(zip(left, right)):
        for field in ("address", "flags", "alignment"):
            if a[field] != b[field]:
                raise ValueError(f"segment {index}: {field} differs")
        common = min(a["memory_size"], b["memory_size"])
        old_common = a["initialized"][:common]
        new_common = b["initialized"][:common]
        first_difference = next((position for position, (u, v) in
                                 enumerate(zip(old_common, new_common)) if u != v), None)
        extra = (a if a["memory_size"] > b["memory_size"] else b)[
            "initialized"][common:]
        extra_all_zero = not any(extra)
        compatible &= first_difference is None and extra_all_zero
        if index + 1 < len(left):
            next_address = left[index + 1]["address"]
            if a["address"] + max(a["memory_size"], b["memory_size"]) > next_address:
                raise ValueError(f"segment {index}: extension overlaps next segment")
        rows.append({
            "address": f"0x{a['address']:x}", "flags": a["flags"],
            "old_file_bytes": a["file_size"], "new_file_bytes": b["file_size"],
            "old_memory_bytes": a["memory_size"],
            "new_memory_bytes": b["memory_size"],
            "old_common_sha256": digest(old_common),
            "new_common_sha256": digest(new_common),
            "first_difference": first_difference,
            "extra_bytes": len(extra), "extra_all_zero": extra_all_zero,
        })
    return {"old_elf": str(old), "new_elf": str(new),
            "old_sha256": digest(old_data), "new_sha256": digest(new_data),
            "load_segments": rows,
            "compatible_initialized_loads": compatible,
            "total_extra_zero_bytes": sum(row["extra_bytes"] for row in rows
                                          if row["extra_all_zero"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.old, args.new)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"compared {len(result['load_segments'])} load segments; "
          f"compatible={result['compatible_initialized_loads']}; "
          f"{result['total_extra_zero_bytes']} added zero bytes")
    if not result["compatible_initialized_loads"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
