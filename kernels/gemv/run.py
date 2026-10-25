#!/usr/bin/env python3
"""Build one faithful BF16 GEMV for the Radiance FPGA-1A campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess

from gemv import bf16_bits_to_f32, device_gemv_reference, f32_to_bf16_bits


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
GENERATED = ROOT / "generated"
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def digest_u32(words: list[int]) -> int:
    value = FNV_OFFSET
    for word in words:
        value = ((value ^ word) * FNV_PRIME) & MASK64
    return value


def output_words(values: list[int]) -> list[int]:
    padded = values + ([0] if len(values) & 1 else [])
    return [padded[i] | (padded[i + 1] << 16)
            for i in range(0, len(padded), 2)]


def write_u16(path: Path, values: list[int]) -> None:
    with path.open("wb") as stream:
        for start in range(0, len(values), 65536):
            chunk = values[start:start + 65536]
            stream.write(struct.pack(f"<{len(chunk)}H", *chunk))


def source_hash() -> str:
    digest = hashlib.sha256()
    paths = [
        ROOT / name for name in
        ("run.py", "gemv.py", "Makefile", "kernel.cpp", "host.cpp",
         "emit_symbols.py")
    ] + [
        REPO / "lib/include/mu_partition.h",
        REPO / "lib/include/nightly/device.h",
        REPO / "lib/include/nightly/host.h",
        ROOT.parent / "addr_hash_stamp.mk",
    ]
    for path in paths:
        digest.update(path.relative_to(REPO).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare(args: argparse.Namespace) -> dict:
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("rows and cols must be positive")
    if args.unroll % args.ilp:
        raise ValueError("unroll must be divisible by ilp")
    if args.cols % (16 * args.unroll):
        raise ValueError("cols must be divisible by 16 lanes times unroll")
    if args.rows * args.cols * 2 + args.cols * 2 >= 0x70000000:
        raise ValueError("GEMV payload exceeds the GPU DRAM address window")
    if args.robust_spin_until_reset and not args.robust_end_drain:
        raise ValueError("robust spin requires robust end drain")
    if args.robust_serial_fence and not args.robust_end_drain:
        raise ValueError("robust serial fence requires robust end drain")

    rng = random.Random(args.seed)
    a_bits = [f32_to_bf16_bits(rng.gauss(0.0, 1.0))
              for _ in range(args.rows * args.cols)]
    x_bits = [f32_to_bf16_bits(rng.gauss(0.0, 1.0))
              for _ in range(args.cols)]
    expected = device_gemv_reference(
        a_bits, x_bits, args.rows, args.cols,
        lanes=16, ilp=args.ilp, unroll=args.unroll,
    )
    high_precision = []
    for row in range(args.rows):
        total = math.fsum(
            bf16_bits_to_f32(a_bits[row * args.cols + col])
            * bf16_bits_to_f32(x_bits[col])
            for col in range(args.cols)
        )
        high_precision.append(f32_to_bf16_bits(total))

    GENERATED.mkdir(exist_ok=True)
    write_u16(GENERATED / "a.bin", a_bits)
    write_u16(GENERATED / "x.bin", x_bits)
    words = output_words(expected)
    expected_digest = digest_u32(words)
    config = [
        "#pragma once",
        f"#define HPC_SM_COUNT {args.sms}u",
        f"#define NIGHTLY_CLUSTERS {args.sms}",
        f"#define NIGHTLY_OCC {args.warps_per_core}",
        f"#define NIGHTLY_ROBUST_GMEM_END {int(args.robust_end_drain)}",
        f"#define NIGHTLY_ROBUST_SPIN_UNTIL_RESET {int(args.robust_spin_until_reset)}",
        f"#define NIGHTLY_ROBUST_SERIAL_FENCE {int(args.robust_serial_fence)}",
        f"#define NIGHTLY_ROBUST_POST_SPIN {args.robust_post_spin}u",
        f"#define GEMV_ROWS {args.rows}u",
        f"#define GEMV_COLS {args.cols}u",
        f"#define GEMV_ILP {args.ilp}u",
        f"#define GEMV_UNROLL {args.unroll}u",
        f"#define GEMV_OUTPUT_WORDS {len(words)}u",
        f"#define GEMV_EXPECTED_DIGEST 0x{expected_digest:016x}ULL",
    ]
    (GENERATED / "config.h").write_text("\n".join(config) + "\n")
    asm = [
        '.section .data,"aw",@progbits',
        '.balign 64', '.globl gemv_a', 'gemv_a:',
        '.incbin "generated/a.bin"',
        '.balign 64', '.globl gemv_x', 'gemv_x:',
        '.incbin "generated/x.bin"',
        '.section .bss,"aw",@nobits',
        '.balign 64', '.globl gemv_guard_before', 'gemv_guard_before:', '.zero 64',
        '.globl gemv_y', 'gemv_y:', f'.zero {len(words) * 4}',
        '.globl gemv_guard_after', 'gemv_guard_after:', '.zero 64',
    ]
    (ROOT / "data.S").write_text("\n".join(asm) + "\n")
    result = {
        "schema": "radiance-gemv-bf16-build-v1",
        "kind": "gemv",
        "operation": "gemv",
        "family": "gemv_bf16",
        "sm_count": args.sms,
        "loader": args.loader,
        "gpu_address_hash": args.loader == "direct-hashed",
        "rows": args.rows,
        "cols": args.cols,
        "benchmark_contract": (
            "upstream-radiance-gemv-16x4096"
            if (args.rows, args.cols) == (16, 4096)
            else "fixed-size-bf16-gemv-row-saturation"
        ),
        "seed": args.seed,
        "element_format": "bf16",
        "accumulation": "bf16-fma-ilp-chains-then-16-lane-bf16-tree",
        "ilp": args.ilp,
        "unroll": args.unroll,
        "muon_warps_per_core": args.warps_per_core,
        "output_element_count": args.rows,
        "output_words": len(words),
        "expected_digest": f"{expected_digest:016x}",
        "expected_output_bits": [f"{value:04x}" for value in expected],
        "high_precision_reference_bits": [f"{value:04x}" for value in high_precision],
        "reference_contract": "exact-device-order-and-independent-fp32-dot",
        "partition": "contiguous-row-ranges-per-sm",
        "correctness": "full-output-exact-device-order-digest",
        "no_vpu": True,
        "robust_end_drain": args.robust_end_drain,
        "robust_spin_until_reset": args.robust_spin_until_reset,
        "robust_serial_fence": args.robust_serial_fence,
        "robust_post_spin": args.robust_post_spin,
        "kernel_revision": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "kernel_source_sha256": source_hash(),
        "status": "prepared",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sms", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--loader", choices=("logical", "direct-hashed"),
                        default="logical")
    parser.add_argument("--warps-per-core", type=int, choices=(1, 2, 4, 8),
                        default=8)
    parser.add_argument("--ilp", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--unroll", type=int, choices=(1, 2, 4, 8, 16), default=8)
    parser.add_argument("--robust-end-drain", action="store_true")
    parser.add_argument("--robust-spin-until-reset", action="store_true")
    parser.add_argument("--robust-serial-fence", action="store_true")
    parser.add_argument("--robust-post-spin", type=int, default=20000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    result = prepare(args)
    if args.prepare_only:
        return
    env = os.environ.copy()
    env["MU_ADDR_HASH"] = "1" if args.loader == "direct-hashed" else "0"
    with (args.out / "build.log").open("w") as log:
        subprocess.run(["make", "kernel.soc.elf"], cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copy2(ROOT / "kernel.soc.elf", args.out / "kernel.soc.elf")
    shutil.copy2(ROOT / "kernel.radiance.elf", args.out / "kernel.radiance.elf")
    if (ROOT / "kernel.radiance.dump").is_file():
        shutil.copy2(ROOT / "kernel.radiance.dump", args.out / "kernel.radiance.dump")
    result["device_elf_sha256"] = hashlib.sha256(
        (args.out / "kernel.radiance.elf").read_bytes()
    ).hexdigest()
    result["status"] = "built"
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
