#!/usr/bin/env python3
"""Build one float32 STREAM Copy, Scale, Add, or Triad Radiance kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import hpc_presets  # noqa: E402  (needs ROOT.parent on the path)
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


def result_value(kind: str, i: int, scalar: float) -> float:
    a, b, c = (input_value(name, i) for name in "abc")
    return {"copy": a, "scale": scalar * c, "add": a + b,
            "triad": b + scalar * c}[kind]


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
            full_host_check: bool = False,
            preset: str = hpc_presets.DEFAULT_PRESET,
            loader: str = "logical", warps_per_core: int = 4,
            scalar: float = 3.0,
            diagnostic_readback: bool = False,
            block_cyclic_lines: bool = False,
            block_cyclic_chunk_lines: int = 0,
            array_offset_bytes: int = 0) -> dict:
    if elements < 2 or elements & 1:
        raise ValueError("elements must be an even integer of at least 2")
    if elements * 12 + 0x100000 + 2 * array_offset_bytes > 0x70000000:
        raise ValueError("STREAM arrays exceed the GPU DRAM address window")
    # The GPU address hash (radiance AddressHashParams, soc/scramble_gpu_elf.py)
    # sends consecutive 32 B units round-robin to 4 L2 slices, each with its own
    # DRAM channel, and keeps each slice's units dense.  A client-address offset
    # of N bytes therefore becomes an offset of N/4 bytes inside the slice, and
    # it is the SLICE-LOCAL address that the DDR3 map decodes: bank =
    # addr[15:13], rank = addr[17:16] (midas DramCommon.scala:118-133).  A
    # multiple of 128 keeps the shift an exact whole number of 32 B units, so
    # the arrays stay line-aligned in slice-local space; 32 KiB is one bank and
    # 256 KiB is one rank.
    if array_offset_bytes < 0 or array_offset_bytes % 128:
        raise ValueError("array_offset_bytes must be a nonnegative multiple of 128")
    if warps_per_core not in (1, 2, 4, 8):
        raise ValueError("warps_per_core must be 1, 2, 4, or 8")
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError("scalar must be finite and greater than zero")
    sms = hpc_presets.resolve(preset).clusters
    if block_cyclic_lines and block_cyclic_chunk_lines:
        raise ValueError("select only one STREAM block-cyclic schedule")
    if block_cyclic_chunk_lines < 0:
        raise ValueError("block-cyclic chunk lines must be nonnegative")
    GENERATED.mkdir(exist_ok=True)
    output = OUTPUT[kind]
    for name in "abc":
        if name != output:
            write_input(GENERATED / f"{name}.bin", name, elements)
        else:
            (GENERATED / f"{name}.bin").write_bytes(b"")
    values = [float_word(result_value(kind, i, scalar)) for i in range(elements)]
    expected = digest(values)
    samples = min(64, elements) if elements > 1024 and not full_host_check else 0
    positions = [i * (elements - 1) // (samples - 1) for i in range(samples)] if samples else []
    expected_sample = digest(values[i] for i in positions) if samples else None
    config = [
        "#pragma once",
        # The cluster count the DATA was partitioned for.  hpc.mk supplies
        # NIGHTLY_CLUSTERS from the preset, and the kernel static_asserts that
        # the two agree, so a preset that does not match the data fails to build
        # rather than producing wrong output.
        f"#define HPC_SM_COUNT {sms}u",
        f"#define STREAM_KIND {KINDS[kind]}",
        f"#define STREAM_ELEMENTS {elements}u",
        f"#define STREAM_SCALAR {scalar.hex()}f",
        f"#define STREAM_READBACK_SAMPLES {samples}u",
        f"#define STREAM_EXPECTED_READBACK_DIGEST 0x{(expected_sample or expected):016x}ULL",
        f"#define STREAM_BLOCK_CYCLIC_CHUNK_LINES {block_cyclic_chunk_lines}u",
        f"#define STREAM_BLOCK_CYCLIC_LINES {int(block_cyclic_lines)}",
    ]
    (GENERATED / "config.h").write_text("\n".join(config) + "\n")
    # Array k (in link order: the two .data inputs, then the .bss output) is
    # pushed k * array_offset_bytes further along, so every pair of arrays is
    # separated by a different multiple of the offset.  The padding goes BEFORE
    # stream_guard_before so the guard band stays immediately adjacent to the
    # output array and still detects an overrun.
    asm = ['.section .data,"aw",@progbits']
    for index, name in enumerate(n for n in "abc" if n != output):
        if index and array_offset_bytes:
            asm += [f".zero {array_offset_bytes}"]
        asm += [".balign 64", f".globl stream_{name}", f"stream_{name}:",
                f'.incbin "generated/{name}.bin"']
    asm += ['.section .bss,"aw",@nobits']
    if array_offset_bytes:
        asm += [".balign 64", f".zero {array_offset_bytes}"]
    asm += [
        '.balign 64',
        '.globl stream_guard_before', 'stream_guard_before:', '.zero 64',
        f'.globl stream_{output}', f'stream_{output}:', f'.zero {elements * 4}',
        '.globl stream_guard_after', 'stream_guard_after:', '.zero 64',
    ]
    (ROOT / "data.S").write_text("\n".join(asm) + "\n")
    source = hashlib.sha256()
    build_rules = ROOT.parent / "addr_hash_stamp.mk"
    source.update(build_rules.name.encode())
    source.update(build_rules.read_bytes())
    for path in (ROOT.parents[1] / "lib/include/mu_partition.h",
                 ROOT.parents[1] / "lib/include/nightly/device.h",
                 ROOT.parents[1] / "lib/include/nightly/host.h"):
        source.update(path.relative_to(ROOT.parents[1]).as_posix().encode())
        source.update(path.read_bytes())
    for shared in ("hpc.mk", "hpc_presets.py"):
        source.update(shared.encode())
        source.update((ROOT.parent / shared).read_bytes())
    for name in ("run.py", "Makefile", "kernel_impl.hpp", "kernel.cpp",
                 "kernel_ilp2.cpp", "kernel_ilp4.cpp", "host.cpp",
                 "emit_symbols.py"):
        source.update(name.encode())
        source.update((ROOT / name).read_bytes())
    case = {"kind": f"stream-{kind}", "stream_operation": kind,
            **hpc_presets.manifest(preset, warps_per_core),
            "loader": loader, "gpu_address_hash": loader == "direct-hashed",
            "pattern_length": elements, "stream_elements": elements,
            "stream_scalar": scalar,
            "benchmark_contract": "stream-v5.10-elemental-equation"
                                  if scalar == 3.0 else "stream-like",
            "count": 1, "output_elements": elements,
            "logical_payload_bytes": elements * (12 if kind in ("add", "triad") else 8),
            "expected_digest": f"{expected:016x}",
            "expected_sample_digest": f"{expected_sample:016x}" if samples else None,
            "readback_samples": samples, "destination_overlap": False,
            "collision_policy": "parallel",
            "address_plan": {"read": "dense", "write": "dense", "operation": kind},
            "suite_sha256": hashlib.sha256(
                f"stream:{kind}:{elements}:scalar{scalar.hex()}:occ{warps_per_core}:"
                f"preset{preset}:blockcyclic{int(block_cyclic_lines)}:"
                f"chunklines{block_cyclic_chunk_lines}:"
                f"arrayoffset{array_offset_bytes}".encode()
            ).hexdigest(),
            "case": KINDS[kind],
            "kernel_revision": subprocess.check_output(
                ["git", "-C", str(ROOT.parents[1]), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "kernel_source_sha256": source.hexdigest(),
            "correctness": "sample-checked" if samples else "digest-checked",
            "host_full_readback": full_host_check or not samples,
            "status": "prepared", "simulator": None,
            "element_format": "float32", "output_element_count": elements,
            "scalar_only": True, "no_vpu": True, "vpu": False,
            "vector_isa": False, "gemmini": False, "mx": False,
            "stream_block_cyclic_lines": block_cyclic_lines,
            "stream_block_cyclic_chunk_lines": block_cyclic_chunk_lines,
            "stream_array_offset_bytes": array_offset_bytes,
            "stream_array_offset_slice_local_bytes": array_offset_bytes // 4,
            "diagnostic_readback": diagnostic_readback,
            "checker_word_pairs": elements // 2}
    (GENERATED / "manifest.json").write_text(json.dumps(case, indent=2) + "\n")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--elements", type=int, default=65536)
    parser.add_argument("--preset", choices=tuple(hpc_presets.PRESETS),
                        default=hpc_presets.DEFAULT_PRESET,
                        help="target bitstream: fixes cluster count, drain "
                             "protocol and L1 coherence together")
    parser.add_argument("--ilp", type=int, choices=(1, 2, 4), default=1,
                        help="unroll factor; selects the kernel variant to build")
    parser.add_argument("--loader", choices=("logical", "direct-hashed"),
                        default="logical",
                        help="logical GPU loading, or direct loading that bypasses the RTL hash")
    parser.add_argument("--warps-per-core", type=int, choices=(1, 2, 4, 8), default=4,
                        help="Muon occupancy used to trade latency hiding for queue pressure")
    parser.add_argument("--scalar", type=float, default=3.0,
                        help="Scale/Triad scalar (STREAM v5.10 uses 3.0)")
    parser.add_argument("--diagnostic-readback", action="store_true",
                        help="print actual digest and guards when the RV64 check fails")
    parser.add_argument("--block-cyclic-lines", action="store_true",
                        help="round-robin complete 64-byte lines across physical SMs")
    parser.add_argument("--block-cyclic-chunk-lines", type=int, default=0,
                        help="round-robin chunks containing this many complete cache lines")
    parser.add_argument("--array-offset-bytes", type=int, default=0,
                        help="push array k of 3 by k*N client bytes, i.e. k*N/4 bytes in "
                             "slice-local DRAM space after the 4-slice GPU address hash; "
                             "32768 moves one DDR3 bank, 262144 moves one rank")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--full-host-check", action="store_true",
                        help="digest every FP32 output word on the RV64 host")
    parser.add_argument("--host-timing", action="store_true",
                        help="print RV64 release-to-all-finished cycle interval")
    args = parser.parse_args()
    out = args.out.resolve()
    case = prepare(args.kind, args.elements, out, args.full_host_check,
                   args.preset, args.loader, args.warps_per_core, args.scalar,
                   args.diagnostic_readback, args.block_cyclic_lines,
                   args.block_cyclic_chunk_lines,
                   args.array_offset_bytes)
    case["host_timing_requested"] = args.host_timing
    case["ilp"] = args.ilp
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    if args.prepare_only:
        return
    build_env = os.environ.copy()
    build_env["MU_ADDR_HASH"] = "1" if args.loader == "direct-hashed" else "0"
    if args.host_timing:
        build_env["EXTRA_HOST_CXXFLAGS"] = (
            build_env.get("EXTRA_HOST_CXXFLAGS", "") +
            " -DRAD_HOST_TIMING=1").strip()
    if args.diagnostic_readback:
        build_env["EXTRA_HOST_CXXFLAGS"] = (
            build_env.get("EXTRA_HOST_CXXFLAGS", "") +
            " -DSTREAM_DIAGNOSTIC_READBACK=1").strip()
    variant = "kernel" if args.ilp == 1 else f"kernel_ilp{args.ilp}"
    with (out / "build.log").open("w") as log:
        subprocess.run(
            ["make", f"{variant}.soc.elf",
             *hpc_presets.make_args(args.preset, args.warps_per_core)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
            env=build_env)
    shutil.copy2(ROOT / f"{variant}.soc.elf", out / "kernel.soc.elf")
    device_elf = out / "kernel.radiance.elf"
    shutil.copy2(ROOT / f"{variant}.radiance.elf", device_elf)
    case["device_elf_sha256"] = hashlib.sha256(device_elf.read_bytes()).hexdigest()
    case["status"] = "built"
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    print(json.dumps(case, indent=2))


if __name__ == "__main__":
    main()
