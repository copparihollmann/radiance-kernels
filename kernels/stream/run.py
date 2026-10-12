#!/usr/bin/env python3
"""Build one float32 STREAM Copy, Scale, Add, or Triad Radiance kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
KINDS = {"copy": 0, "scale": 1, "add": 2, "triad": 3}
OUTPUT = {"copy": "c", "scale": "b", "add": "c", "triad": "a"}
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def input_value(name: str, i: int) -> float:
    if name == "a":
        return float(i % 1024 + 1)
    if name == "b":
        return float((3 * i) % 1024 + 2)
    return float((5 * i) % 1024 + 3)


def result_value(kind: str, i: int) -> float:
    a, b, c = (input_value(name, i) for name in "abc")
    return {"copy": a, "scale": 2.0 * c, "add": a + b,
            "triad": b + 2.0 * c}[kind]


def float_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def digest(words) -> int:
    value = FNV_OFFSET
    for word in words:
        value = ((value ^ word) * FNV_PRIME) & MASK64
    return value


def write_input(path: Path, name: str, elements: int) -> None:
    with path.open("wb") as stream:
        for start in range(0, elements, 32768):
            words = [input_value(name, i)
                     for i in range(start, min(start + 32768, elements))]
            stream.write(struct.pack(f"<{len(words)}f", *words))


def prepare(kind: str, elements: int, out: Path,
            full_host_check: bool = False) -> dict:
    if elements < 2 or elements & 1:
        raise ValueError("elements must be an even integer of at least 2")
    if elements * 12 + 0x100000 > 0x70000000:
        raise ValueError("STREAM arrays exceed the GPU DRAM address window")
    GENERATED.mkdir(exist_ok=True)
    output = OUTPUT[kind]
    for name in "abc":
        if name != output:
            write_input(GENERATED / f"{name}.bin", name, elements)
        else:
            (GENERATED / f"{name}.bin").write_bytes(b"")
    values = [float_word(result_value(kind, i)) for i in range(elements)]
    expected = digest(values)
    samples = min(64, elements) if elements > 1024 and not full_host_check else 0
    positions = [i * (elements - 1) // (samples - 1) for i in range(samples)] if samples else []
    expected_sample = digest(values[i] for i in positions) if samples else None
    config = [
        "#pragma once",
        f"#define STREAM_KIND {KINDS[kind]}",
        f"#define STREAM_ELEMENTS {elements}u",
        f"#define STREAM_READBACK_SAMPLES {samples}u",
        f"#define STREAM_EXPECTED_READBACK_DIGEST 0x{(expected_sample or expected):016x}ULL",
    ]
    (GENERATED / "config.h").write_text("\n".join(config) + "\n")
    asm = ['.section .data,"aw",@progbits']
    for name in "abc":
        if name == output:
            continue
        asm += [".balign 64", f".globl stream_{name}", f"stream_{name}:",
                f'.incbin "generated/{name}.bin"']
    asm += [
        '.section .bss,"aw",@nobits', '.balign 64',
        '.globl stream_guard_before', 'stream_guard_before:', '.zero 64',
        f'.globl stream_{output}', f'stream_{output}:', f'.zero {elements * 4}',
        '.globl stream_guard_after', 'stream_guard_after:', '.zero 64',
    ]
    (ROOT / "data.S").write_text("\n".join(asm) + "\n")
    source = hashlib.sha256()
    build_rules = ROOT.parent / "addr_hash_stamp.mk"
    source.update(build_rules.name.encode())
    source.update(build_rules.read_bytes())
    for name in ("run.py", "Makefile", "kernel.cpp", "host.cpp", "emit_symbols.py"):
        source.update((ROOT / name).read_bytes())
    case = {"kind": f"stream-{kind}", "stream_operation": kind,
            "pattern_length": elements, "stream_elements": elements,
            "count": 1, "output_elements": elements // 2,
            "logical_payload_bytes": elements * (12 if kind in ("add", "triad") else 8),
            "expected_digest": f"{expected:016x}",
            "expected_sample_digest": f"{expected_sample:016x}" if samples else None,
            "readback_samples": samples, "destination_overlap": False,
            "collision_policy": "parallel",
            "address_plan": {"read": "dense", "write": "dense", "operation": kind},
            "suite_sha256": hashlib.sha256(f"stream:{kind}:{elements}".encode()).hexdigest(),
            "case": KINDS[kind], "kernel_source_sha256": source.hexdigest(),
            "correctness": "sample-checked" if samples else "digest-checked",
            "host_full_readback": full_host_check or not samples,
            "status": "prepared", "simulator": None,
            "element_format": "float32", "output_element_count": elements,
            "checker_word_pairs": elements // 2}
    (GENERATED / "manifest.json").write_text(json.dumps(case, indent=2) + "\n")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--elements", type=int, default=65536)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--full-host-check", action="store_true",
                        help="digest every FP32 output word on the RV64 host")
    parser.add_argument("--host-timing", action="store_true",
                        help="print RV64 release-to-all-finished cycle interval")
    args = parser.parse_args()
    out = args.out.resolve()
    case = prepare(args.kind, args.elements, out, args.full_host_check)
    case["host_timing_requested"] = args.host_timing
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    if args.prepare_only:
        return
    build_env = os.environ.copy()
    if args.host_timing:
        build_env["EXTRA_HOST_CXXFLAGS"] = (
            build_env.get("EXTRA_HOST_CXXFLAGS", "") +
            " -DRAD_HOST_TIMING=1").strip()
    with (out / "build.log").open("w") as log:
        subprocess.run(["make", "kernel.soc.elf"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, env=build_env)
    shutil.copy2(ROOT / "kernel.soc.elf", out / "kernel.soc.elf")
    case["status"] = "built"
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    print(json.dumps(case, indent=2))


if __name__ == "__main__":
    main()
