#!/usr/bin/env python3
"""Build the Vortex-compatible 3D 27-point averaging stencil for Radiance."""

from __future__ import annotations

from array import array
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
REFERENCE_REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"
REFERENCE_KERNEL = "tests/regression/stencil3d/kernel.cpp"
REFERENCE_SEED = 50
RAND_MAX = 2147483647
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


def reference_input(elements: int) -> list[float]:
    # The pinned Vortex host calls std::srand(50), then float(std::rand()) /
    # RAND_MAX.  Calling the process C library reproduces that Linux reference
    # stream, including the float conversion before division.
    libc = ctypes.CDLL(None)
    libc.srand.argtypes = [ctypes.c_uint]
    libc.rand.restype = ctypes.c_int
    libc.srand(REFERENCE_SEED)
    denominator = f32(float(RAND_MAX))
    return [f32(f32(float(libc.rand())) / denominator) for _ in range(elements)]


def reference_output(values: list[float], size: int) -> list[float]:
    plane = size * size
    output: list[float] = []
    for z in range(size):
        for y in range(size):
            for x in range(size):
                total = f32(0.0)
                for dz in (-1, 0, 1):
                    nz = min(size - 1, max(0, z + dz))
                    for dy in (-1, 0, 1):
                        ny = min(size - 1, max(0, y + dy))
                        for dx in (-1, 0, 1):
                            nx = min(size - 1, max(0, x + dx))
                            total = f32(total + values[nz * plane + ny * size + nx])
                output.append(f32(total / f32(27.0)))
    return output


def write_f32(path: Path, values: list[float]) -> None:
    data = array("f", values)
    if sys.byteorder != "little":
        data.byteswap()
    path.write_bytes(data.tobytes())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(size: int, out: Path, *, sms: int = 1,
            loader: str = "logical", warps_per_core: int = 8,
            max_ulp: int = 6, robust_end_drain: bool = False,
            robust_spin_until_reset: bool = False,
            robust_serial_fence: bool = False,
            robust_post_spin: int = 500) -> dict:
    if size < 2 or size > 256:
        raise ValueError("size must be between 2 and 256")
    if sms not in (1, 2, 4):
        raise ValueError("sms must be 1, 2, or 4")
    if loader not in ("logical", "direct-hashed"):
        raise ValueError("loader must be logical or direct-hashed")
    if warps_per_core not in (1, 2, 4, 8):
        raise ValueError("warps_per_core must be 1, 2, 4, or 8")
    if max_ulp < 0:
        raise ValueError("max_ulp must be nonnegative")
    if robust_spin_until_reset and not robust_end_drain:
        raise ValueError("robust_spin_until_reset requires robust_end_drain")
    if robust_serial_fence and not robust_end_drain:
        raise ValueError("robust_serial_fence requires robust_end_drain")
    if robust_post_spin < 0:
        raise ValueError("robust_post_spin must be nonnegative")
    elements = size ** 3
    if elements * 12 + 0x200000 > 0x70000000:
        raise ValueError("stencil arrays exceed the GPU DRAM address window")

    GENERATED.mkdir(exist_ok=True)
    source = reference_input(elements)
    expected = reference_output(source, size)
    write_f32(GENERATED / "input.bin", source)
    write_f32(GENERATED / "expected.bin", expected)
    words = [float_word(value) for value in expected]
    expected_digest = digest(words)
    config = [
        "#pragma once",
        f"#define HPC_SM_COUNT {sms}u",
        f"#define NIGHTLY_CLUSTERS {sms}",
        f"#define NIGHTLY_OCC {warps_per_core}",
        f"#define NIGHTLY_ROBUST_GMEM_END {int(robust_end_drain)}",
        f"#define NIGHTLY_ROBUST_SPIN_UNTIL_RESET {int(robust_spin_until_reset)}",
        f"#define NIGHTLY_ROBUST_SERIAL_FENCE {int(robust_serial_fence)}",
        f"#define NIGHTLY_ROBUST_POST_SPIN {robust_post_spin}u",
        f"#define STENCIL3D_SIZE {size}u",
        f"#define STENCIL3D_ELEMENTS {elements}u",
        f"#define STENCIL3D_MAX_ULP {max_ulp}u",
    ]
    (GENERATED / "config.h").write_text("\n".join(config) + "\n")
    asm = [
        '.section .data,"aw",@progbits',
        '.balign 64', '.globl stencil3d_input', 'stencil3d_input:',
        '.incbin "generated/input.bin"',
        '.balign 64', '.globl stencil3d_expected', 'stencil3d_expected:',
        '.incbin "generated/expected.bin"',
        '.section .bss,"aw",@nobits', '.balign 64',
        '.globl stencil3d_guard_before', 'stencil3d_guard_before:', '.zero 64',
        '.globl stencil3d_output', 'stencil3d_output:', f'.zero {elements * 4}',
        '.globl stencil3d_guard_after', 'stencil3d_guard_after:', '.zero 64',
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
        "name": f"stencil3d--size-{size}--{sms}sm",
        "family": "stencil3d", "sm_count": sms, "loader": loader,
        "gpu_address_hash": loader == "direct-hashed",
        "size": size, "output_element_count": elements,
        "expected_digest": f"{expected_digest:016x}",
        "element_format": "float32", "max_ulp": max_ulp,
        "correctness": "complete-output-reference-check",
        "reference_repository": "vortexgpgpu/vortex",
        "reference_revision": REFERENCE_REVISION,
        "reference_kernel": REFERENCE_KERNEL,
        "reference_seed": REFERENCE_SEED,
        "benchmark_contract": "vortex-stencil3d-27-point-boundary-replication",
        "partition": "disjoint-contiguous-64-byte-output-lines",
        "muon_warps_per_core": warps_per_core,
        "robust_end_drain": robust_end_drain,
        "robust_spin_until_reset": robust_spin_until_reset,
        "robust_serial_fence": robust_serial_fence,
        "robust_post_spin": robust_post_spin,
        "kernel_source_sha256": source_hash.hexdigest(),
        "input_sha256": sha256(GENERATED / "input.bin"),
        "expected_sha256": sha256(GENERATED / "expected.bin"),
        "status": "prepared", "simulator": None,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--sms", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--loader", choices=("logical", "direct-hashed"),
                        default="logical")
    parser.add_argument("--warps-per-core", type=int, choices=(1, 2, 4, 8),
                        default=8)
    parser.add_argument("--max-ulp", type=int, default=6)
    parser.add_argument("--robust-end-drain", action="store_true")
    parser.add_argument("--robust-spin-until-reset", action="store_true")
    parser.add_argument("--robust-serial-fence", action="store_true")
    parser.add_argument("--robust-post-spin", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    case = prepare(
        args.size, out, sms=args.sms, loader=args.loader,
        warps_per_core=args.warps_per_core, max_ulp=args.max_ulp,
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
    elf = out / "kernel.soc.elf"
    device_elf = out / "kernel.radiance.elf"
    shutil.copy2(ROOT / "kernel.soc.elf", elf)
    shutil.copy2(ROOT / "kernel.radiance.elf", device_elf)
    case.update({"status": "built", "elf": str(elf),
                 "elf_sha256": sha256(elf), "bytes": elf.stat().st_size,
                 "device_elf": str(device_elf),
                 "device_elf_sha256": sha256(device_elf)})
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    print(json.dumps(case, indent=2))


if __name__ == "__main__":
    main()
