#!/usr/bin/env python3
"""Prepare and build the faithful, physical-SM-partitioned Radiance sfilter."""

from __future__ import annotations

from array import array
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import struct
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"

DEFAULT_SIZE = 48
SEED = 0
MASK = [1.0] * 9
MAX_ULP = 4
FLOATS_PER_CACHE_LINE = 16

REFERENCE_REPOSITORY = "vortexgpgpu/vortex"
REFERENCE_REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"
REFERENCE_KERNEL = "tests/opencl/sfilter/kernel.cl"
REFERENCE_KERNEL_GIT_BLOB = "a0be9d109f3d544114ed2e14802d55f566bef88f"
REFERENCE_KERNEL_SHA256 = (
    "c2a128944702a8b80adb111909adacdaf28539d96c68bc1fa909ecc4226749b5"
)
REFERENCE_MAIN = "tests/opencl/sfilter/main.cc"
REFERENCE_MAIN_GIT_BLOB = "c56d26ffdc72cf7f3da4f5b86cb6c99d1decb087"
REFERENCE_MAIN_SHA256 = (
    "01a7bd1816b44af8cd2d4c5839a35999097cac97c1254fd2dd3efa232ba43783"
)

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def digest(words: Iterable[int]) -> int:
    value = FNV_OFFSET
    for word in words:
        value = ((value ^ word) * FNV_PRIME) & MASK64
    return value


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def current_dataset(size: int) -> list[float]:
    """Reproduce the pre-repair Radiance dataset exactly."""
    rng = random.Random(SEED)
    return [f32(rng.uniform(0.0, 100.0)) for _ in range(size * size)]


def reference_output(source: list[float], size: int) -> list[float]:
    """Evaluate the pinned Vortex equation in ordered FP32 operations."""
    if len(source) != size * size:
        raise ValueError("source length does not match size")
    mask = [f32(value) for value in MASK]
    output = [f32(0.0)] * (size * size)
    for y in range(1, size - 1):
        for x in range(1, size - 1):
            addr = x + y * size
            terms = (
                f32(source[addr - 1 - size] * mask[0]),
                f32(source[addr - size] * mask[1]),
                f32(source[addr + 1 - size] * mask[2]),
                f32(source[addr - 1] * mask[3]),
                f32(source[addr] * mask[4]),
                f32(source[addr + 1] * mask[5]),
                f32(source[addr - 1 + size] * mask[6]),
                f32(source[addr + size] * mask[7]),
                f32(source[addr + 1 + size] * mask[8]),
            )
            total = f32(0.0)
            for term in terms:
                total = f32(total + term)
            output[addr] = total
    return output


def row_boundaries(size: int, sms: int) -> list[int]:
    """Balance complete interior rows over physical SM/threadblock IDs."""
    if sms not in (1, 2, 4):
        raise ValueError("sms must be 1, 2, or 4")
    rows = size - 2
    base, extra = divmod(rows, sms)
    boundaries = [1]
    for sm in range(sms):
        boundaries.append(boundaries[-1] + base + (sm < extra))
    assert boundaries[-1] == size - 1
    return boundaries


def ownership_metadata(size: int, sms: int) -> list[dict]:
    boundaries = row_boundaries(size, sms)
    owners = []
    for sm, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        byte_begin = begin * size * 4
        byte_end = end * size * 4
        owners.append({
            "physical_threadblock_id": sm,
            "row_begin_inclusive": begin,
            "row_end_exclusive": end,
            "interior_pixels": (end - begin) * (size - 2),
            "output_row_byte_begin": byte_begin,
            "output_row_byte_end": byte_end,
            "begin_cache_line_offset": byte_begin % 64,
            "end_cache_line_offset": byte_end % 64,
        })
    return owners


def validate_ownership(size: int, sms: int) -> list[dict]:
    owners = ownership_metadata(size, sms)
    pixels = []
    for owner in owners:
        if owner["begin_cache_line_offset"] != 0 or owner["end_cache_line_offset"] != 0:
            raise ValueError("SM ownership boundary is not cache-line aligned")
        for y in range(owner["row_begin_inclusive"], owner["row_end_exclusive"]):
            pixels.extend(x + y * size for x in range(1, size - 1))
    expected = [x + y * size for y in range(1, size - 1)
                for x in range(1, size - 1)]
    if pixels != expected or len(pixels) != len(set(pixels)):
        raise ValueError("SM ownership skips or duplicates interior pixels")
    return owners


def write_u32(path: Path, words: Iterable[int]) -> None:
    data = array("I", words)
    if data.itemsize != 4:
        raise RuntimeError("native unsigned int is not 32 bits")
    if sys.byteorder != "little":
        data.byteswap()
    path.write_bytes(data.tobytes())


def parse_legacy_words(path: Path, symbol: str) -> list[int]:
    match = re.search(
        rf"(?:const\s+)?(?:__global\s+)?uint32_t\s+{re.escape(symbol)}\[\]\s*=\s*\{{([^}}]*)\}};",
        path.read_text(), re.S,
    )
    if match is None:
        raise ValueError(f"{symbol} not found in {path}")
    return [int(token.strip(), 0)
            for token in match.group(1).replace("\n", " ").split(",")
            if token.strip()]


def source_digest() -> str:
    value = hashlib.sha256()
    paths = (
        ROOT.parent / "addr_hash_stamp.mk",
        ROOT.parents[1] / "lib/include/nightly/device.h",
        ROOT.parents[1] / "lib/include/nightly/host.h",
        ROOT / "Makefile",
        ROOT / "emit_symbols.py",
        ROOT / "functional_smoke.py",
        ROOT / "host.cpp",
        ROOT / "kernel.cpp",
        ROOT / "kernel_impl.hpp",
        ROOT / "sfilter.py",
    )
    for path in paths:
        value.update(path.relative_to(ROOT.parents[1]).as_posix().encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def kernel_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT.parents[1]), "rev-parse", "HEAD"], text=True,
    ).strip()


def prepare(size: int, out: Path, *, sms: int = 1,
            loader: str = "logical", warps_per_core: int = 8,
            robust_end_drain: bool = False,
            robust_spin_until_reset: bool = False,
            robust_serial_fence: bool = False,
            robust_post_spin: int = 500) -> dict:
    if size < 16 or size % FLOATS_PER_CACHE_LINE != 0:
        raise ValueError("size must be at least 16 and a multiple of 16 floats")
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

    source = current_dataset(size)
    expected = reference_output(source, size)
    source_words = [float_word(value) for value in source]
    expected_words = [float_word(value) for value in expected]
    owners = validate_ownership(size, sms)
    boundaries = row_boundaries(size, sms)

    # The tracked 48x48 arrays are the old port's exact dataset contract.
    legacy_match = False
    if size == DEFAULT_SIZE:
        legacy_match = (
            source_words == parse_legacy_words(ROOT / "data", "src_raw") and
            expected_words == parse_legacy_words(
                ROOT / "expected", "expected_sfilter_raw")
        )
        if not legacy_match:
            raise ValueError("generated data no longer matches tracked sfilter contract")

    GENERATED.mkdir(exist_ok=True)
    write_u32(GENERATED / "input.bin", source_words)
    write_u32(GENERATED / "expected.bin", expected_words)
    expected_digest = digest(expected_words)
    boundary_initializer = "{" + ",".join(f"{value}u" for value in boundaries) + "}"
    mask_words = [float_word(value) for value in MASK]
    config = [
        "#pragma once",
        f"#define HPC_SM_COUNT {sms}u",
        f"#define NIGHTLY_CLUSTERS {sms}",
        f"#define NIGHTLY_OCC {warps_per_core}",
        f"#define NIGHTLY_ROBUST_GMEM_END {int(robust_end_drain)}",
        f"#define NIGHTLY_ROBUST_SPIN_UNTIL_RESET {int(robust_spin_until_reset)}",
        f"#define NIGHTLY_ROBUST_SERIAL_FENCE {int(robust_serial_fence)}",
        f"#define NIGHTLY_ROBUST_POST_SPIN {robust_post_spin}u",
        f"#define SFILTER_SIZE {size}u",
        f"#define SFILTER_ELEMENTS {size * size}u",
        f"#define SFILTER_MAX_ULP {MAX_ULP}u",
        f"#define SFILTER_FLOATS_PER_CACHE_LINE {FLOATS_PER_CACHE_LINE}u",
        f"#define SFILTER_SM_ROW_BOUNDARIES {boundary_initializer}",
        f"#define SFILTER_EXPECTED_DIGEST 0x{expected_digest:016x}ULL",
    ]
    config.extend(
        f"#define SFILTER_M{index}_BITS 0x{word:08x}u"
        for index, word in enumerate(mask_words)
    )
    (GENERATED / "config.h").write_text("\n".join(config) + "\n")

    assembly = [
        '.section .data,"aw",@progbits',
        '.balign 64', '.globl sfilter_input', 'sfilter_input:',
        '.incbin "generated/input.bin"',
        '.balign 64', '.globl sfilter_expected', 'sfilter_expected:',
        '.incbin "generated/expected.bin"',
        '.section .bss,"aw",@nobits', '.balign 64',
        '.globl sfilter_guard_before', 'sfilter_guard_before:', '.zero 64',
        '.globl sfilter_output', 'sfilter_output:', f'.zero {size * size * 4}',
        '.balign 64',
        '.globl sfilter_guard_after', 'sfilter_guard_after:', '.zero 64',
    ]
    (ROOT / "data.S").write_text("\n".join(assembly) + "\n")

    ownership_serialized = json.dumps(owners, sort_keys=True).encode()
    manifest = {
        "name": f"sfilter--size-{size}--{sms}sm",
        "family": "sfilter",
        "kind": "2d-3x3-spatial-filter",
        "sm_count": sms,
        "loader": loader,
        "gpu_address_hash": loader == "direct-hashed",
        "size": size,
        "seed": SEED,
        "mask": MASK,
        "output_element_count": size * size,
        "interior_element_count": (size - 2) * (size - 2),
        "boundary_element_count": size * size - (size - 2) * (size - 2),
        "element_format": "float32",
        "max_ulp": MAX_ULP,
        "expected_digest": f"{expected_digest:016x}",
        "correctness": "complete-output-four-ulp-zero-boundaries-guards-and-digest",
        "oracle": "ordered-float32-pinned-vortex-nine-multiply-eight-add-every-output",
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_revision": REFERENCE_REVISION,
        "reference_kernel": REFERENCE_KERNEL,
        "reference_kernel_git_blob": REFERENCE_KERNEL_GIT_BLOB,
        "reference_kernel_sha256": REFERENCE_KERNEL_SHA256,
        "reference_main": REFERENCE_MAIN,
        "reference_main_git_blob": REFERENCE_MAIN_GIT_BLOB,
        "reference_main_sha256": REFERENCE_MAIN_SHA256,
        "reference_tolerance_ulp": MAX_ULP,
        "equation": "dst[x+y*ldc]=sum_left_to_right(src[(x+dx)+(y+dy)*ldc]*m[dy,dx])",
        "accumulation_order": "pinned-vortex-left-to-right-no-contraction",
        "dataset_contract": (
            "radiance-sfilter-48x48-python-random-seed-0-mask-nine-ones-zero-boundary"
            if size == DEFAULT_SIZE else
            f"radiance-sfilter-derived-{size}x{size}-python-random-seed-0-mask-nine-ones-zero-boundary"
        ),
        "benchmark_contract": (
            "radiance-existing-sfilter-48x48-vortex-equation-derived-input"
            if size == DEFAULT_SIZE else
            "radiance-scaled-sfilter-vortex-equation-derived-input"
        ),
        "dataset_contract_preserved": legacy_match,
        "partition": "physical-threadblock-contiguous-whole-output-rows",
        "partition_row_boundaries": boundaries,
        "partition_ownership": owners,
        "partition_ownership_sha256": hashlib.sha256(ownership_serialized).hexdigest(),
        "partition_complete": True,
        "partition_overlap": False,
        "partition_cache_line_safe": True,
        "cache_line_bytes": 64,
        "scalar_only": True,
        "no_vpu": True,
        "vpu": False,
        "vector_isa": False,
        "gemmini": False,
        "mx": False,
        "muon_warps_per_core": warps_per_core,
        "robust_end_drain": robust_end_drain,
        "robust_spin_until_reset": robust_spin_until_reset,
        "robust_serial_fence": robust_serial_fence,
        "robust_post_spin": robust_post_spin,
        "kernel_revision": kernel_revision(),
        "kernel_source_sha256": source_digest(),
        "input_sha256": sha256(GENERATED / "input.bin"),
        "expected_sha256": sha256(GENERATED / "expected.bin"),
        "generator_platform": platform.platform(),
        "status": "prepared",
        "simulator": None,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
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
    manifest = prepare(
        args.size, out, sms=args.sms, loader=args.loader,
        warps_per_core=args.warps_per_core,
        robust_end_drain=args.robust_end_drain,
        robust_spin_until_reset=args.robust_spin_until_reset,
        robust_serial_fence=args.robust_serial_fence,
        robust_post_spin=args.robust_post_spin,
    )
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return

    env = os.environ.copy()
    env["MU_ADDR_HASH"] = "1" if args.loader == "direct-hashed" else "0"
    local_toolchain = (
        ROOT.parents[3] / "toolchains/riscv-tools-prefix/riscv-tools"
    )
    if "RISCV64_TOOLCHAIN_PATH" not in env and local_toolchain.is_dir():
        env["RISCV64_TOOLCHAIN_PATH"] = str(local_toolchain)
    with (out / "build.log").open("w") as log:
        subprocess.run(["make", "kernel.soc.elf"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, env=env)
    elf = out / "kernel.soc.elf"
    device_elf = out / "kernel.radiance.elf"
    shutil.copy2(ROOT / "kernel.soc.elf", elf)
    shutil.copy2(ROOT / "kernel.radiance.elf", device_elf)
    manifest.update({
        "status": "built",
        "elf": str(elf),
        "elf_sha256": sha256(elf),
        "bytes": elf.stat().st_size,
        "device_elf": str(device_elf),
        "device_elf_sha256": sha256(device_elf),
    })
    (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
