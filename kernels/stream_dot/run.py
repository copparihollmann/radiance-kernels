#!/usr/bin/env python3
"""Build a faithful float32 BabelStream-derived Dot reduction for Radiance."""

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
UPSTREAM_REPOSITORY = "UoB-HPC/BabelStream"
UPSTREAM_REVISION = "5cb9445483786f3502b12b8faab8414f8c5873b1"
UPSTREAM_KERNEL = "src/cuda/CUDAStream.cu:dot_kernel"
START_A = 0.1
START_B = 0.2
MU_NUM_CORES = 2
MU_NUM_THREADS = 16
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def digest(words: list[int]) -> int:
    value = FNV_OFFSET
    for word in words:
        value = ((value ^ word) * FNV_PRIME) & MASK64
    return value


def partition(total: int, count: int, index: int, granule: int = 16) -> tuple[int, int]:
    chunks = (total + granule - 1) // granule
    base, extra = divmod(chunks, count)
    begin_chunk = index * base + min(index, extra)
    owned = base + (1 if index < extra else 0)
    return min(total, begin_chunk * granule), min(total, (begin_chunk + owned) * granule)


def expected_partials(elements: int, sms: int, warps_per_core: int) -> list[float]:
    threads = warps_per_core * MU_NUM_CORES * MU_NUM_THREADS
    product = f32(f32(START_A) * f32(START_B))
    partials = []
    for sm in range(sms):
        begin, end = partition(elements, sms, sm)
        scratch = []
        for tid in range(threads):
            acc = f32(0.0)
            for _ in range(begin + tid, end, threads):
                acc = f32(acc + product)
            scratch.append(acc)
        offset = threads // 2
        while offset:
            for tid in range(offset):
                scratch[tid] = f32(scratch[tid] + scratch[tid + offset])
            offset //= 2
        partials.append(scratch[0])
    return partials


def write_constant(path: Path, value: float, elements: int) -> None:
    word = struct.pack("<f", f32(value))
    with path.open("wb") as stream:
        block = word * min(elements, 65536)
        left = elements
        while left:
            count = min(left, 65536)
            stream.write(block[:count * 4])
            left -= count


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(elements: int, out: Path, *, sms: int = 1,
            loader: str = "logical", warps_per_core: int = 8,
            robust_end_drain: bool = False,
            robust_spin_until_reset: bool = False,
            robust_serial_fence: bool = False,
            robust_post_spin: int = 500) -> dict:
    if elements < 256 or elements % 16:
        raise ValueError("elements must be a multiple of 16 and at least 256")
    if sms not in (1, 2, 4):
        raise ValueError("sms must be 1, 2, or 4")
    if loader not in ("logical", "direct-hashed"):
        raise ValueError("loader must be logical or direct-hashed")
    if warps_per_core not in (1, 2, 4, 8):
        raise ValueError("warps_per_core must be 1, 2, 4, or 8")
    if robust_spin_until_reset and not robust_end_drain:
        raise ValueError("robust_spin_until_reset requires robust_end_drain")
    if robust_serial_fence and not robust_end_drain:
        raise ValueError("robust_serial_fence requires robust_end_drain")
    if robust_post_spin < 0:
        raise ValueError("robust_post_spin must be nonnegative")
    if elements * 8 + 0x100000 > 0x70000000:
        raise ValueError("STREAM Dot arrays exceed the GPU DRAM address window")

    GENERATED.mkdir(exist_ok=True)
    write_constant(GENERATED / "a.bin", START_A, elements)
    write_constant(GENERATED / "b.bin", START_B, elements)
    partials = expected_partials(elements, sms, warps_per_core)
    words = [float_word(value) for value in partials]
    expected_digest = digest(words)
    expected_final = f32(0.0)
    for value in partials:
        expected_final = f32(expected_final + value)
    scalar_golden = f32(f32(START_A) * f32(START_B) * elements)
    relative_error = abs(expected_final - scalar_golden) / abs(scalar_golden)
    expected_words = ", ".join(f"0x{word:08x}u" for word in words)
    config = [
        "#pragma once",
        f"#define HPC_SM_COUNT {sms}u",
        f"#define NIGHTLY_CLUSTERS {sms}",
        f"#define NIGHTLY_OCC {warps_per_core}",
        f"#define NIGHTLY_ROBUST_GMEM_END {int(robust_end_drain)}",
        f"#define NIGHTLY_ROBUST_SPIN_UNTIL_RESET {int(robust_spin_until_reset)}",
        f"#define NIGHTLY_ROBUST_SERIAL_FENCE {int(robust_serial_fence)}",
        f"#define NIGHTLY_ROBUST_POST_SPIN {robust_post_spin}u",
        f"#define STREAM_DOT_ELEMENTS {elements}u",
        f"#define STREAM_DOT_EXPECTED_WORDS {{{expected_words}}}",
        f"#define STREAM_DOT_EXPECTED_DIGEST 0x{expected_digest:016x}ULL",
    ]
    (GENERATED / "config.h").write_text("\n".join(config) + "\n")
    asm = [
        '.section .data,"aw",@progbits',
        '.balign 64', '.globl stream_dot_a', 'stream_dot_a:',
        '.incbin "generated/a.bin"',
        '.balign 64', '.globl stream_dot_b', 'stream_dot_b:',
        '.incbin "generated/b.bin"',
        '.section .bss,"aw",@nobits', '.balign 64',
        '.globl stream_dot_guard_before', 'stream_dot_guard_before:', '.zero 64',
        '.globl stream_dot_partial', 'stream_dot_partial:', '.zero 16',
        '.balign 64',
        '.globl stream_dot_guard_after', 'stream_dot_guard_after:', '.zero 64',
    ]
    (ROOT / "data.S").write_text("\n".join(asm) + "\n")

    source_hash = hashlib.sha256()
    for path in (
        ROOT.parent / "addr_hash_stamp.mk",
        ROOT.parents[1] / "lib/include/mu_partition.h",
        ROOT.parents[1] / "lib/include/nightly/device.h",
        ROOT.parents[1] / "lib/include/nightly/host.h",
        ROOT / "run.py", ROOT / "Makefile", ROOT / "kernel.cpp",
        ROOT / "host.cpp", ROOT / "emit_symbols.py",
    ):
        source_hash.update(path.relative_to(ROOT.parents[1]).as_posix().encode())
        source_hash.update(path.read_bytes())
    case = {
        "name": f"stream-dot-babelstream-f32--{elements}--{sms}sm",
        "kind": "stream-dot", "operation": "stream-dot",
        "family": "stream_dot", "sm_count": sms, "loader": loader,
        "gpu_address_hash": loader == "direct-hashed",
        "stream_elements": elements, "output_element_count": sms,
        "element_format": "float32", "input_a": f32(START_A),
        "input_b": f32(START_B), "expected_partial_words": words,
        "expected_digest": f"{expected_digest:016x}",
        "expected_final_word": f"{float_word(expected_final):08x}",
        "scalar_golden_word": f"{float_word(scalar_golden):08x}",
        "relative_error_vs_scalar_golden": relative_error,
        "correctness": "exact-device-order-partials-and-guards",
        "benchmark_contract": "babelstream-derived-dot-f32",
        "reference_repository": UPSTREAM_REPOSITORY,
        "reference_revision": UPSTREAM_REVISION,
        "reference_kernel": UPSTREAM_KERNEL,
        "reduction": "per-sm-strided-thread-partials-plus-shared-tree-host-final-sum",
        "partition": "disjoint-contiguous-64-byte-input-lines",
        "logical_payload_bytes": elements * 8,
        "muon_warps_per_core": warps_per_core,
        "robust_end_drain": robust_end_drain,
        "robust_spin_until_reset": robust_spin_until_reset,
        "robust_serial_fence": robust_serial_fence,
        "robust_post_spin": robust_post_spin,
        "kernel_source_sha256": source_hash.hexdigest(),
        "input_a_sha256": sha256(GENERATED / "a.bin"),
        "input_b_sha256": sha256(GENERATED / "b.bin"),
        "status": "prepared", "simulator": None,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=1048576)
    parser.add_argument("--sms", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--loader", choices=("logical", "direct-hashed"),
                        default="logical")
    parser.add_argument("--warps-per-core", type=int, choices=(1, 2, 4, 8),
                        default=8)
    parser.add_argument("--robust-end-drain", action="store_true")
    parser.add_argument("--robust-spin-until-reset", action="store_true")
    parser.add_argument("--robust-serial-fence", action="store_true")
    parser.add_argument("--robust-post-spin", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    case = prepare(
        args.elements, out, sms=args.sms, loader=args.loader,
        warps_per_core=args.warps_per_core,
        robust_end_drain=args.robust_end_drain,
        robust_spin_until_reset=args.robust_spin_until_reset,
        robust_serial_fence=args.robust_serial_fence,
        robust_post_spin=args.robust_post_spin,
    )
    if args.prepare_only:
        return
    env = os.environ.copy()
    env["MU_ADDR_HASH"] = "1" if args.loader == "direct-hashed" else "0"
    with (out / "build.log").open("w") as log:
        subprocess.run(["make", "kernel.soc.elf"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, env=env)
    for name in ("kernel.soc.elf", "kernel.radiance.elf"):
        shutil.copy2(ROOT / name, out / name)
    elf = out / "kernel.soc.elf"
    case.update({"status": "built", "elf": str(elf),
                 "elf_sha256": sha256(elf), "bytes": elf.stat().st_size,
                 "device_elf": str(out / "kernel.radiance.elf"),
                 "device_elf_sha256": sha256(out / "kernel.radiance.elf")})
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    print(json.dumps(case, indent=2))


if __name__ == "__main__":
    main()
