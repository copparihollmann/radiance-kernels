#!/usr/bin/env python3
"""Generate and build the pinned Vortex/Rodinia Hotspot3D workloads."""

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
REFERENCE_REPOSITORY = ROOT.parents[3] / "reference" / "vortex"
DEFAULT_RISCV64_TOOLCHAIN = (
    ROOT.parents[3] / "toolchains" / "riscv-tools-prefix" / "riscv-tools"
)
REFERENCE_REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"
REFERENCE_SOURCES = {
    "tests/opencl/hotspot3D/main.cc": {
        "blob": "af729c56b76263cb3869b80787173363f4ff69ae",
        "sha256": "3954ddc7b8a841605c09d21fa8ef1f4cc07b1216fc9c3748a425d7b88907feec",
    },
    "tests/opencl/hotspot3D/kernel.cl": {
        "blob": "ec62ac18e872fdb8cd514f146626b6e4ed590ea4",
        "sha256": "f3823a9850bab4460cbaa3763f55c8d2ad2f42d44e13bc11399a43a0e2ef0e47",
    },
    "tests/opencl/hotspot3D/Makefile": {
        "blob": "8cabdd94d49fa92918e25b8ae90d9d952096be80",
        "sha256": "38c9f1d989f66e6e23c55aaf5760f4a8d9270bdf310874de6da9f1690c2acf64",
    },
}
PROFILES = {
    "canary": {"nx": 16, "ny": 16, "nz": 4, "iterations": 2},
    "meaningful": {"nx": 64, "ny": 64, "nz": 8, "iterations": 4},
}
MAX_PD = 3.0e6
PRECISION = 0.001
SPEC_HEAT_SI = 1.75e6
K_SI = 100
FACTOR_CHIP = 0.5
T_CHIP = 0.0005
CHIP_HEIGHT = 0.016
CHIP_WIDTH = 0.016
AMBIENT = 80.0
REFERENCE_SEED = 7
RAND_MAX = 2147483647
COLUMNS_PER_LINE = 16
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1
GUARDS = {
    "before": 0x483D0000,
    "middle_a": 0x483D1000,
    "middle_b": 0x483D2000,
    "after": 0x483D3000,
}
OUTPUT_POISON = 0x7FC00048
OWNER_POISON = 0xFFFFFFFF
MODEL_STAGE_FLAG = 0x80000000


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fmul(lhs: float, rhs: float) -> float:
    return f32(f32(lhs) * f32(rhs))


def fadd(lhs: float, rhs: float) -> float:
    return f32(f32(lhs) + f32(rhs))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv64(words) -> int:
    value = FNV_OFFSET
    for word in words:
        value = ((value ^ (int(word) & 0xFFFFFFFF)) * FNV_PRIME) & MASK64
    return value


def write_f32(path: Path, values: list[float]) -> None:
    data = array("f", values)
    if sys.byteorder != "little":
        data.byteswap()
    path.write_bytes(data.tobytes())


def reference_objects(reference_repo: Path = REFERENCE_REPOSITORY) -> dict:
    repo = reference_repo.resolve()
    if not (repo / ".git").exists():
        raise ValueError(f"missing pinned Vortex Git repository: {repo}")
    resolved = subprocess.check_output(
        ["git", "rev-parse", f"{REFERENCE_REVISION}^{{commit}}"],
        cwd=repo, text=True,
    ).strip()
    if resolved != REFERENCE_REVISION:
        raise ValueError(f"Vortex revision mismatch: {resolved}")
    verified = {}
    for source, expected in REFERENCE_SOURCES.items():
        blob = subprocess.check_output(
            ["git", "rev-parse", f"{REFERENCE_REVISION}:{source}"],
            cwd=repo, text=True,
        ).strip()
        contents = subprocess.check_output(
            ["git", "show", f"{REFERENCE_REVISION}:{source}"], cwd=repo,
        )
        digest = hashlib.sha256(contents).hexdigest()
        if blob != expected["blob"] or digest != expected["sha256"]:
            raise ValueError(
                f"pinned source mismatch for {source}: blob={blob} sha256={digest}"
            )
        verified[source] = {"git_blob": blob, "sha256": digest}
    return verified


def coefficients(nx: int, ny: int, nz: int) -> dict[str, float]:
    # Match the source's effective C types: FACTOR_CHIP, SPEC_HEAT_SI,
    # MAX_PD, and PRECISION are unsuffixed (double), while geometry and the
    # remaining stencil arithmetic are binary32.
    dx = f32(f32(CHIP_HEIGHT) / nx)
    dy = f32(f32(CHIP_WIDTH) / ny)
    dz = f32(f32(T_CHIP) / nz)
    cap = f32(FACTOR_CHIP * SPEC_HEAT_SI * f32(T_CHIP) * dx * dy)
    rx_den = fmul(fmul(fmul(f32(2.0), f32(K_SI)), f32(T_CHIP)), dx)
    rx = f32(dy / rx_den)
    ry_den = fmul(fmul(fmul(f32(2.0), f32(K_SI)), f32(T_CHIP)), dy)
    ry = f32(dx / ry_den)
    rz_den = fmul(fmul(f32(K_SI), dx), dy)
    rz = f32(dz / rz_den)
    max_slope = f32(MAX_PD / (FACTOR_CHIP * f32(T_CHIP) * SPEC_HEAT_SI))
    dt = f32(PRECISION / max_slope)
    step_div_cap = f32(dt / cap)
    ce = f32(step_div_cap / rx)
    cw = ce
    cn = f32(step_div_cap / ry)
    cs = cn
    ct = f32(step_div_cap / rz)
    cb = ct
    cc = f32(f32(1.0) - fadd(fadd(fmul(f32(2.0), ce),
                                       fmul(f32(2.0), cn)),
                                  fmul(f32(3.0), ct)))
    return {"step_div_cap": step_div_cap, "ce": ce, "cw": cw,
            "cn": cn, "cs": cs, "ct": ct, "cb": cb, "cc": cc}


def reference_inputs(elements: int) -> tuple[list[float], list[float]]:
    libc = ctypes.CDLL(None)
    libc.srand.argtypes = [ctypes.c_uint]
    libc.rand.restype = ctypes.c_int
    libc.srand(REFERENCE_SEED)
    temperature: list[float] = []
    power: list[float] = []
    for _ in range(elements):
        temp_fraction = f32((libc.rand() % 25000) / f32(1000.0))
        temperature.append(fadd(f32(300.0), temp_fraction))
        power.append(f32((libc.rand() % 1000) / f32(1000.0)))
    return power, temperature


def reference_states(power: list[float], initial: list[float], nx: int,
                     ny: int, nz: int, iterations: int,
                     coeff: dict[str, float]) -> list[list[float]]:
    """All full-field oracle states; deliberately does not z-sweep/reuse."""
    plane = nx * ny
    source = list(initial)
    states = [source]
    for _ in range(iterations):
        target = [f32(0.0)] * len(source)
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    center = x + y * nx + z * plane
                    west = center if x == 0 else center - 1
                    east = center if x + 1 == nx else center + 1
                    north = center if y == 0 else center - nx
                    south = center if y + 1 == ny else center + nx
                    bottom = center if z == 0 else center - plane
                    top = center if z + 1 == nz else center + plane
                    value = fmul(coeff["cc"], source[center])
                    value = fadd(value, fmul(coeff["cw"], source[west]))
                    value = fadd(value, fmul(coeff["ce"], source[east]))
                    value = fadd(value, fmul(coeff["cs"], source[south]))
                    value = fadd(value, fmul(coeff["cn"], source[north]))
                    value = fadd(value, fmul(coeff["cb"], source[bottom]))
                    value = fadd(value, fmul(coeff["ct"], source[top]))
                    value = fadd(
                        value, fmul(coeff["step_div_cap"], power[center]))
                    value = fadd(value, fmul(coeff["ct"], f32(AMBIENT)))
                    target[center] = value
        source = target
        states.append(source)
    return states


def reference_output(power: list[float], initial: list[float], nx: int,
                     ny: int, nz: int, iterations: int,
                     coeff: dict[str, float]) -> list[float]:
    return reference_states(
        power, initial, nx, ny, nz, iterations, coeff)[-1]


def partition_boundaries(columns: int, sms: int) -> list[int]:
    if sms not in (1, 2, 4):
        raise ValueError("sms must be 1, 2, or 4")
    if columns <= 0 or columns % COLUMNS_PER_LINE:
        raise ValueError("column count must be a positive multiple of 16")
    chunks = columns // COLUMNS_PER_LINE
    base, extra = divmod(chunks, sms)
    boundaries = [0]
    cursor = 0
    for sm in range(sms):
        cursor += base + int(sm < extra)
        boundaries.append(cursor * COLUMNS_PER_LINE)
    return boundaries


def ownership(columns: int, sms: int) -> tuple[list[int], list[int]]:
    boundaries = partition_boundaries(columns, sms)
    owners = [-1] * columns
    visits = [0] * columns
    for sm, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        for column in range(begin, end):
            owners[column] = sm
            visits[column] += 1
    if visits != [1] * columns:
        raise ValueError("partition does not own each column exactly once")
    for begin in range(0, columns, COLUMNS_PER_LINE):
        if len(set(owners[begin:begin + COLUMNS_PER_LINE])) != 1:
            raise ValueError("partition splits an output cache line")
    return owners, boundaries


def source_digest() -> str:
    digest = hashlib.sha256()
    paths = (
        ROOT.parent / "addr_hash_stamp.mk",
        ROOT.parents[1] / "lib/include/mu_partition.h",
        ROOT.parents[1] / "lib/include/nightly/device.h",
        ROOT.parents[1] / "lib/include/nightly/host.h",
        ROOT / "Makefile", ROOT / "README.md", ROOT / "emit_symbols.py",
        ROOT / "host.cpp", ROOT / "kernel.cpp", ROOT / "model_check.py",
        ROOT / "run.py", ROOT / "test_run.py",
    )
    for path in paths:
        digest.update(path.relative_to(ROOT.parents[1]).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("RISCV64_TOOLCHAIN_PATH") and not env.get("RISCV"):
        compiler = DEFAULT_RISCV64_TOOLCHAIN / "bin/riscv64-unknown-elf-g++"
        if not compiler.is_file():
            raise ValueError(
                "RISCV64_TOOLCHAIN_PATH/RISCV is unset and the private "
                f"toolchain is missing: {compiler}"
            )
        env["RISCV64_TOOLCHAIN_PATH"] = str(DEFAULT_RISCV64_TOOLCHAIN)
    return env


def prepare(profile: str, sms: int, out: Path, *, loader: str = "logical",
            warps_per_core: int = 8) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    if sms not in (1, 2, 4):
        raise ValueError("sms must be 1, 2, or 4")
    if loader not in ("logical", "direct-hashed"):
        raise ValueError("loader must be logical or direct-hashed")
    if warps_per_core not in (1, 2, 4, 8):
        raise ValueError("warps_per_core must be 1, 2, 4, or 8")
    reference = reference_objects()
    dims = PROFILES[profile]
    nx, ny, nz = dims["nx"], dims["ny"], dims["nz"]
    iterations = dims["iterations"]
    columns = nx * ny
    elements = columns * nz
    if elements * 5 * 4 + columns * 4 + 0x200000 > 0x70000000:
        raise ValueError("Hotspot3D arrays exceed the GPU DRAM address window")
    owners, boundaries = ownership(columns, sms)
    coeff = coefficients(nx, ny, nz)
    power, initial = reference_inputs(elements)
    states = reference_states(
        power, initial, nx, ny, nz, iterations, coeff)
    expected = states[-1]
    model_inputs = [value for state in states[:-1] for value in state]
    stage_expected_digests = [
        f"{fnv64(float_word(value) for value in state):016x}"
        for state in states[1:]
    ]
    expected_words = [float_word(value) for value in expected]
    expected_digest = fnv64(expected_words)
    owner_digest = fnv64(owners)
    max_ulp = max(32, 16 * iterations)

    GENERATED.mkdir(exist_ok=True)
    write_f32(GENERATED / "power.bin", power)
    write_f32(GENERATED / "initial.bin", initial)
    write_f32(GENERATED / "expected.bin", expected)
    write_f32(GENERATED / "model_inputs.bin", model_inputs)
    config = [
        "#pragma once",
        f"#define HPC_SM_COUNT {sms}u",
        f"#define NIGHTLY_CLUSTERS {sms}",
        f"#define NIGHTLY_OCC {warps_per_core}",
        f"#define HOTSPOT3D_NX {nx}u",
        f"#define HOTSPOT3D_NY {ny}u",
        f"#define HOTSPOT3D_NZ {nz}u",
        f"#define HOTSPOT3D_COLUMNS {columns}u",
        f"#define HOTSPOT3D_ELEMENTS {elements}u",
        f"#define HOTSPOT3D_ITERATIONS {iterations}u",
        f"#define HOTSPOT3D_COLUMNS_PER_LINE {COLUMNS_PER_LINE}u",
        f"#define HOTSPOT3D_MAX_ULP {max_ulp}u",
        f"#define HOTSPOT3D_MODEL_STAGE_FLAG 0x{MODEL_STAGE_FLAG:08x}u",
        f"#define HOTSPOT3D_STEP_DIV_CAP {coeff['step_div_cap'].hex()}f",
        f"#define HOTSPOT3D_CE {coeff['ce'].hex()}f",
        f"#define HOTSPOT3D_CW {coeff['cw'].hex()}f",
        f"#define HOTSPOT3D_CN {coeff['cn'].hex()}f",
        f"#define HOTSPOT3D_CS {coeff['cs'].hex()}f",
        f"#define HOTSPOT3D_CT {coeff['ct'].hex()}f",
        f"#define HOTSPOT3D_CB {coeff['cb'].hex()}f",
        f"#define HOTSPOT3D_CC {coeff['cc'].hex()}f",
        f"#define HOTSPOT3D_AMBIENT {f32(AMBIENT).hex()}f",
        f"#define HOTSPOT3D_EXPECTED_DIGEST 0x{expected_digest:016x}ULL",
        f"#define HOTSPOT3D_OWNER_DIGEST 0x{owner_digest:016x}ULL",
        f"#define HOTSPOT3D_OUTPUT_POISON 0x{OUTPUT_POISON:08x}u",
        f"#define HOTSPOT3D_OWNER_POISON 0x{OWNER_POISON:08x}u",
    ]
    for name, pattern in GUARDS.items():
        config.append(
            f"#define HOTSPOT3D_GUARD_{name.upper()}_PATTERN "
            f"0x{pattern:08x}u"
        )
    (GENERATED / "config.h").write_text("\n".join(config) + "\n")
    asm = [
        '.section .data,"aw",@progbits',
        '.balign 64', '.globl hotspot3d_power', 'hotspot3d_power:',
        '.incbin "generated/power.bin"',
        '.balign 64', '.globl hotspot3d_initial', 'hotspot3d_initial:',
        '.incbin "generated/initial.bin"',
        '.balign 64', '.globl hotspot3d_expected', 'hotspot3d_expected:',
        '.incbin "generated/expected.bin"',
        '.balign 64', '.globl hotspot3d_model_inputs',
        'hotspot3d_model_inputs:', '.incbin "generated/model_inputs.bin"',
        '.balign 64', '.globl hotspot3d_guard_before',
        'hotspot3d_guard_before:',
        *(f'.long 0x{(GUARDS["before"] ^ index):08x}'
          for index in range(16)),
        '.globl hotspot3d_output', 'hotspot3d_output:',
        f'.zero {elements * 4}',
        '.globl hotspot3d_guard_middle_a', 'hotspot3d_guard_middle_a:',
        *(f'.long 0x{(GUARDS["middle_a"] ^ index):08x}'
          for index in range(16)),
        '.globl hotspot3d_scratch', 'hotspot3d_scratch:',
        f'.zero {elements * 4}',
        '.globl hotspot3d_guard_middle_b', 'hotspot3d_guard_middle_b:',
        *(f'.long 0x{(GUARDS["middle_b"] ^ index):08x}'
          for index in range(16)),
        '.globl hotspot3d_owners', 'hotspot3d_owners:',
        f'.zero {columns * 4}',
        '.globl hotspot3d_guard_after', 'hotspot3d_guard_after:',
        *(f'.long 0x{(GUARDS["after"] ^ index):08x}'
          for index in range(16)),
    ]
    (ROOT / "data.S").write_text("\n".join(asm) + "\n")
    case = {
        "schema": "radiance-hotspot3d-build-result-v1",
        "name": f"hotspot3d--{profile}--{sms}sm",
        "family": "hotspot3d", "profile": profile, "sm_count": sms,
        "loader": loader, "gpu_address_hash": loader == "direct-hashed",
        "nx": nx, "ny": ny, "nz": nz, "iterations": iterations,
        "columns": columns, "output_element_count": elements,
        "updates": elements * iterations,
        "max_ulp": max_ulp,
        "correctness": "complete-output-cpu-oracle-plus-ownership-and-guards",
        "reference_repository": "vortexgpgpu/vortex",
        "reference_checkout": str(REFERENCE_REPOSITORY.resolve()),
        "reference_revision": REFERENCE_REVISION,
        "reference_sources": reference,
        "reference_seed": REFERENCE_SEED,
        "benchmark_contract": "vortex-rodinia-hotspot3d-hotspotOpt1",
        "partition": "disjoint-contiguous-16-column-64-byte-lines",
        "partition_boundaries": boundaries,
        "muon_warps_per_core": warps_per_core,
        "scalar_no_vpu": True,
        "expected_digest": f"{expected_digest:016x}",
        "stage_expected_digests": stage_expected_digests,
        "owner_digest": f"{owner_digest:016x}",
        "coefficient_words": {
            name: f"{float_word(value):08x}" for name, value in coeff.items()
        },
        "kernel_source_sha256": source_digest(),
        "power_sha256": sha256(GENERATED / "power.bin"),
        "initial_sha256": sha256(GENERATED / "initial.bin"),
        "expected_sha256": sha256(GENERATED / "expected.bin"),
        "model_inputs_sha256": sha256(GENERATED / "model_inputs.bin"),
        "status": "prepared", "simulator": None,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="meaningful")
    parser.add_argument("--sms", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--loader", choices=("logical", "direct-hashed"),
                        default="logical")
    parser.add_argument("--warps-per-core", type=int, choices=(1, 2, 4, 8),
                        default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    case = prepare(args.profile, args.sms, out, loader=args.loader,
                   warps_per_core=args.warps_per_core)
    if args.prepare_only:
        return
    env = build_environment()
    env["MU_ADDR_HASH"] = "1" if args.loader == "direct-hashed" else "0"
    with (out / "build.log").open("w") as log:
        subprocess.run(["make", "clean"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, env=env)
        subprocess.run(["make", "kernel.soc.elf"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, env=env)
    elf = out / "kernel.soc.elf"
    device_elf = out / "kernel.radiance.elf"
    shutil.copy2(ROOT / "kernel.soc.elf", elf)
    shutil.copy2(ROOT / "kernel.radiance.elf", device_elf)
    case.update({
        "status": "built", "elf": str(elf), "elf_sha256": sha256(elf),
        "bytes": elf.stat().st_size, "device_elf": str(device_elf),
        "device_elf_sha256": sha256(device_elf),
    })
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    print(json.dumps(case, indent=2))


if __name__ == "__main__":
    main()
