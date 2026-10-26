#!/usr/bin/env python3
"""Generate and build the full pinned Vortex/Rodinia Gaussian workload."""

from __future__ import annotations

from array import array
import argparse
import ctypes
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
GENERATED = ROOT / "generated"
REFERENCE_REPOSITORY = ROOT.parents[3] / "reference" / "vortex"
DEFAULT_RISCV64_TOOLCHAIN = (
    ROOT.parents[3] / "toolchains" / "riscv-tools-prefix" / "riscv-tools"
)
REFERENCE_REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"
REFERENCE_SOURCES = {
    "tests/opencl/guassian/main.cc": {
        "blob": "a06ebbbdcd168090a9ac103135fc71652269fa21",
        "sha256": "be765b18368d2ee0f07fad70faa03cc8af8cf5ed8e225e9bc291ee9a8d6d809a",
    },
    "tests/opencl/guassian/kernel.cl": {
        "blob": "c370e9b28c3d6e5c00b13e6f382f34a684adb602",
        "sha256": "0e23c3bf524c72889dd45abe72ca7a2325688fc996b472274c952fa9b6704303",
    },
    "tests/opencl/guassian/Makefile": {
        "blob": "5cfba02810a3f369fca07722da3df049bb2429e7",
        "sha256": "4c5cf2ffa9b012b4c397ff8e96eee58867081394190c696586f900f3927100fb",
    },
    "tests/opencl/guassian/matrix4.txt": {
        "blob": "abf30b495fcd52529379208ff7faf4f259b777e7",
        "sha256": "dda3ac09727d7dc785b9d0bf96d15dcf7afb758fcfba06d539a91304969d587a",
    },
}
PROFILES = {"canary": 4, "meaningful": 64}
UPSTREAM_GENERATED_SIZE = 32
REFERENCE_SEED = 7
RAND_MAX = 2147483647
ROWS_PER_LINE = 16
MODEL_STAGE_FLAG = 0x80000000
OWNER_POISON = 0xFFFFFFFF
MAX_ULP = 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1
GUARDS = {
    "before": 0x6A550000,
    "middle": 0x6A551000,
    "model": 0x6A552000,
    "after": 0x6A553000,
}


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fadd(a: float, b: float) -> float:
    return f32(f32(a) + f32(b))


def fsub(a: float, b: float) -> float:
    return f32(f32(a) - f32(b))


def fmul(a: float, b: float) -> float:
    return f32(f32(a) * f32(b))


def fdiv(a: float, b: float) -> float:
    return f32(f32(a) / f32(b))


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


def write_f32(path: Path, values) -> None:
    packed = array("f", values)
    if packed.itemsize != 4:
        raise RuntimeError("host float is not binary32")
    if sys.byteorder != "little":
        packed.byteswap()
    path.write_bytes(packed.tobytes())


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
            ["git", "show", f"{REFERENCE_REVISION}:{source}"], cwd=repo)
        digest = hashlib.sha256(contents).hexdigest()
        if blob != expected["blob"] or digest != expected["sha256"]:
            raise ValueError(
                f"pinned source mismatch for {source}: blob={blob} sha256={digest}")
        verified[source] = {"git_blob": blob, "sha256": digest}
    return verified


def fixture(profile: str) -> tuple[list[float], list[float]]:
    size = PROFILES[profile]
    if profile == "canary":
        a = [
            -0.6, -0.5, 0.7, 0.3,
            -0.3, -0.9, 0.3, 0.7,
            -0.4, -0.5, -0.3, -0.8,
             0.0, -0.1, 0.2, 0.9,
        ]
        b = [-0.85, -0.68, 0.24, -0.53]
        return [f32(v) for v in a], [f32(v) for v in b]
    libc = ctypes.CDLL(None)
    libc.srand.argtypes = [ctypes.c_uint]
    libc.rand.restype = ctypes.c_int
    libc.srand(REFERENCE_SEED)
    denominator = f32(float(RAND_MAX))
    a = [fdiv(f32(float(libc.rand())), denominator)
         for _ in range(size * size)]
    b = [fdiv(f32(float(libc.rand())), denominator) for _ in range(size)]
    # This is the pinned host's no-pivot stability rule.
    for row in range(size):
        a[row * size + row] = fadd(a[row * size + row], f32(float(size)))
    return a, b


def fan1(a: list[float], m: list[float], size: int, t: int) -> None:
    pivot = a[t * size + t]
    for row in range(t + 1, size):
        m[row * size + t] = fdiv(a[row * size + t], pivot)


def fan2(a: list[float], b: list[float], m: list[float],
         size: int, t: int) -> None:
    pivot_b = b[t]
    for row in range(t + 1, size):
        factor = m[row * size + t]
        for col in range(t, size):
            offset = row * size + col
            a[offset] = fsub(a[offset], fmul(factor, a[t * size + col]))
        b[row] = fsub(b[row], fmul(factor, pivot_b))


def oracle(original_a: list[float], original_b: list[float], size: int) -> dict:
    """Independent complete binary32 Fan1/Fan2 sequence and all pre-states."""
    a, b = list(original_a), list(original_b)
    m = [f32(0.0)] * (size * size)
    model_a, model_b, model_m = [], [], []
    stage_digests = []
    for t in range(size - 1):
        for phase in (0, 1):
            model_a.extend(a)
            model_b.extend(b)
            model_m.extend(m)
            if phase == 0:
                fan1(a, m, size, t)
            else:
                fan2(a, b, m, size, t)
            stage_digests.append({
                "a": f"{fnv64(map(float_word, a)):016x}",
                "b": f"{fnv64(map(float_word, b)):016x}",
                "m": f"{fnv64(map(float_word, m)):016x}",
            })
    return {"a": a, "b": b, "m": m, "model_a": model_a,
            "model_b": model_b, "model_m": model_m,
            "stage_digests": stage_digests}


def back_substitution(a: list[float], b: list[float], size: int) -> list[float]:
    x = [f32(0.0)] * size
    for row in range(size - 1, -1, -1):
        value = b[row]
        for col in range(row + 1, size):
            value = fsub(value, fmul(a[row * size + col], x[col]))
        x[row] = fdiv(value, a[row * size + row])
    return x


def relative_backward_error(a: list[float], b: list[float],
                            x: list[float], size: int) -> float:
    residual_inf = 0.0
    a_inf = 0.0
    for row in range(size):
        dot = math.fsum(float(a[row * size + col]) * float(x[col])
                        for col in range(size))
        residual_inf = max(residual_inf, abs(dot - float(b[row])))
        a_inf = max(a_inf, math.fsum(abs(float(a[row * size + col]))
                                     for col in range(size)))
    x_inf = max(abs(float(value)) for value in x)
    b_inf = max(abs(float(value)) for value in b)
    denominator = a_inf * x_inf + b_inf
    return residual_inf / denominator if denominator else residual_inf


def owner_stride(size: int) -> int:
    return ((size + ROWS_PER_LINE - 1) // ROWS_PER_LINE) * ROWS_PER_LINE


def stage_owners(size: int, sms: int, stage: int) -> list[int]:
    stride = owner_stride(size)
    t = stage // 2
    return [((row // ROWS_PER_LINE) % sms if row < size and row > t
             else OWNER_POISON) for row in range(stride)]


def validate_ownership(size: int, sms: int) -> dict:
    stride = owner_stride(size)
    stages = 2 * (size - 1)
    all_owners = []
    for stage in range(stages):
        owners = stage_owners(size, sms, stage)
        all_owners.extend(owners)
        t = stage // 2
        visits = [0] * size
        for block in range((size + ROWS_PER_LINE - 1) // ROWS_PER_LINE):
            owner = block % sms
            for row in range(block * ROWS_PER_LINE,
                             min(size, (block + 1) * ROWS_PER_LINE)):
                if row > t:
                    if owners[row] != owner:
                        raise ValueError("owner model mismatch")
                    visits[row] += 1
        if any(visits[row] != (1 if row > t else 0) for row in range(size)):
            raise ValueError("row partition is not exact-once")
        for begin in range(0, stride, ROWS_PER_LINE):
            active = [owners[row] for row in range(begin, begin + ROWS_PER_LINE)
                      if row < size and row > t]
            if active and len(set(active)) != 1:
                raise ValueError("row partition splits a cache line")
    return {"stride": stride, "all": all_owners,
            "digest": f"{fnv64(all_owners):016x}"}


def source_digest() -> str:
    digest = hashlib.sha256()
    paths = (
        ROOT.parent / "addr_hash_stamp.mk",
        ROOT.parents[1] / "lib/include/nightly/device.h",
        ROOT.parents[1] / "lib/include/nightly/host.h",
        ROOT / "Makefile", ROOT / "README.md", ROOT / "actual_checker.h",
        ROOT / "emit_symbols.py", ROOT / "host.cpp", ROOT / "kernel.cpp",
        ROOT / "model_check.py", ROOT / "native_checker_test.cpp",
        ROOT / "run.py", ROOT / "softfloat_f32.h", ROOT / "test_run.py",
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
            raise ValueError(f"private RISC-V toolchain missing: {compiler}")
        env["RISCV64_TOOLCHAIN_PATH"] = str(DEFAULT_RISCV64_TOOLCHAIN)
    return env


def prepare(profile: str, sms: int, out: Path,
            *, loader: str = "logical", warps_per_core: int = 8,
            robust_end_drain: bool = False,
            robust_spin_until_reset: bool = False,
            robust_serial_fence: bool = False,
            robust_post_spin: int = 500) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
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
    references = reference_objects()
    size = PROFILES[profile]
    original_a, original_b = fixture(profile)
    expected = oracle(original_a, original_b, size)
    owners = validate_ownership(size, sms)
    solution = back_substitution(expected["a"], expected["b"], size)
    residual = relative_backward_error(original_a, original_b, solution, size)
    # A conservative first-order no-pivot binary32 bound: 64*n*u.  The
    # diagonally-dominant meaningful input is expected to be far below it.
    residual_tolerance = 64.0 * size * (2.0 ** -24)
    if not math.isfinite(residual) or residual > residual_tolerance:
        raise ValueError(
            f"CPU oracle residual {residual} exceeds {residual_tolerance}")
    tolerance_factor = 64 * size
    if tolerance_factor & (tolerance_factor - 1) or tolerance_factor > (1 << 24):
        raise ValueError("residual tolerance must be an exact power of two")
    residual_tolerance_shift = 24 - (tolerance_factor.bit_length() - 1)
    stages = 2 * (size - 1)
    matrix_words = size * size
    updates = sum((size - 1 - t) +
                  (size - 1 - t) * (size - t) + (size - 1 - t)
                  for t in range(size - 1))

    GENERATED.mkdir(exist_ok=True)
    binaries = {
        "original_a.bin": original_a, "original_b.bin": original_b,
        "expected_a.bin": expected["a"], "expected_b.bin": expected["b"],
        "expected_m.bin": expected["m"],
        "expected_solution.bin": solution,
        "model_a.bin": expected["model_a"],
        "model_b.bin": expected["model_b"], "model_m.bin": expected["model_m"],
    }
    for name, values in binaries.items():
        write_f32(GENERATED / name, values)

    residual_e18 = round(residual * 1e18)
    tolerance_e18 = round(residual_tolerance * 1e18)
    config = [
        "#pragma once", f"#define HPC_SM_COUNT {sms}u",
        f"#define NIGHTLY_CLUSTERS {sms}",
        f"#define NIGHTLY_OCC {warps_per_core}",
        f"#define NIGHTLY_ROBUST_GMEM_END {int(robust_end_drain)}",
        f"#define NIGHTLY_ROBUST_SPIN_UNTIL_RESET {int(robust_spin_until_reset)}",
        f"#define NIGHTLY_ROBUST_SERIAL_FENCE {int(robust_serial_fence)}",
        f"#define NIGHTLY_ROBUST_POST_SPIN {robust_post_spin}u",
        f"#define GAUSSIAN_SIZE {size}u",
        f"#define GAUSSIAN_MATRIX_WORDS {matrix_words}u",
        f"#define GAUSSIAN_STAGES {stages}u",
        f"#define GAUSSIAN_ROWS_PER_LINE {ROWS_PER_LINE}u",
        f"#define GAUSSIAN_OWNER_STRIDE {owners['stride']}u",
        f"#define GAUSSIAN_OWNER_POISON 0x{OWNER_POISON:08x}u",
        f"#define GAUSSIAN_MODEL_STAGE_FLAG 0x{MODEL_STAGE_FLAG:08x}u",
        f"#define GAUSSIAN_MAX_ULP {MAX_ULP}u",
        f"#define GAUSSIAN_OWNER_DIGEST 0x{owners['digest']}ULL",
        f"#define GAUSSIAN_ORIGINAL_A_DIGEST 0x{fnv64(map(float_word, original_a)):016x}ULL",
        f"#define GAUSSIAN_ORIGINAL_B_DIGEST 0x{fnv64(map(float_word, original_b)):016x}ULL",
        f"#define GAUSSIAN_ORACLE_RESIDUAL_E18 {residual_e18}ULL",
        f"#define GAUSSIAN_RESIDUAL_TOLERANCE_E18 {tolerance_e18}ULL",
        f"#define GAUSSIAN_RESIDUAL_TOLERANCE_SHIFT {residual_tolerance_shift}u",
    ]
    for name, pattern in GUARDS.items():
        config.append(f"#define GAUSSIAN_GUARD_{name.upper()}_PATTERN "
                      f"0x{pattern:08x}u")
    (GENERATED / "config.h").write_text("\n".join(config) + "\n")

    asm = ['.section .data,"aw",@progbits', '.balign 64']
    def incbin(symbol: str, filename: str) -> None:
        asm.extend([f'.globl {symbol}', f'{symbol}:',
                    f'.incbin "generated/{filename}"', '.balign 64'])
    incbin("gaussian_original_a", "original_a.bin")
    incbin("gaussian_original_b", "original_b.bin")
    asm.extend(['.globl gaussian_guard_before', 'gaussian_guard_before:'])
    asm.extend(f'.long 0x{GUARDS["before"] ^ i:08x}' for i in range(16))
    asm.append('.balign 64')
    incbin("gaussian_a", "original_a.bin")
    incbin("gaussian_b", "original_b.bin")
    asm.extend(['.globl gaussian_m', 'gaussian_m:', f'.zero {matrix_words * 4}',
                '.balign 64', '.globl gaussian_guard_middle',
                'gaussian_guard_middle:'])
    asm.extend(f'.long 0x{GUARDS["middle"] ^ i:08x}' for i in range(16))
    asm.append('.balign 64')
    incbin("gaussian_expected_a", "expected_a.bin")
    incbin("gaussian_expected_b", "expected_b.bin")
    incbin("gaussian_expected_m", "expected_m.bin")
    asm.extend(['.globl gaussian_guard_model', 'gaussian_guard_model:'])
    asm.extend(f'.long 0x{GUARDS["model"] ^ i:08x}' for i in range(16))
    asm.append('.balign 64')
    incbin("gaussian_model_a", "model_a.bin")
    incbin("gaussian_model_b", "model_b.bin")
    incbin("gaussian_model_m", "model_m.bin")
    asm.extend(['.globl gaussian_owners', 'gaussian_owners:',
                f'.fill {stages * owners["stride"]},4,0xffffffff',
                '.balign 64', '.globl gaussian_guard_after',
                'gaussian_guard_after:'])
    asm.extend(f'.long 0x{GUARDS["after"] ^ i:08x}' for i in range(16))
    (ROOT / "data.S").write_text("\n".join(asm) + "\n")

    case = {
        "schema": "radiance-gaussian-build-result-v1",
        "name": f"gaussian--{profile}--{size}--{sms}sm",
        "family": "gaussian", "profile": profile, "size": size,
        "sm_count": sms, "loader": loader,
        "gpu_address_hash": loader == "direct-hashed", "stages": stages,
        "fan1_stages": size - 1, "fan2_stages": size - 1,
        "useful_scalar_updates": updates,
        "upstream_generated_default_size": UPSTREAM_GENERATED_SIZE,
        "meaningful_vs_upstream_default_cubic_work_ratio":
            (size / UPSTREAM_GENERATED_SIZE) ** 3,
        "phase_order": "Fan1(t),device-wide-host-wait,Fan2(t),device-wide-host-wait",
        "partition": "absolute-16-row-cache-line-block-cyclic",
        "partition_exact_once": True, "partition_cache_line_safe": True,
        "owner_stride": owners["stride"], "owner_digest": owners["digest"],
        "correctness": (
            "actual-A-b-software-binary32-back-substitution-plus-certified-"
            "original-system-residual-and-complete-A-b-m-finiteness-stage-"
            "ownership-original-integrity-and-guards"
        ),
        "oracle": "independent-python-binary32-full-pivot-sequence",
        "solution": (
            "integer-software-binary32-back-substitution-from-actual-"
            "FPGA-transformed-A-b"
        ),
        "oracle_trajectory_role": "exhaustive-diagnostic-not-acceptance-gate",
        "actual_output_acceptance": (
            "outward-fixed-point-exact-product-residual-certificate"
        ),
        "relative_backward_error": residual,
        "relative_backward_error_source": "CPU-oracle-diagnostic-only",
        "residual_norm": "||A*x-b||inf/(||A||inf*||x||inf+||b||inf)",
        "residual_tolerance": residual_tolerance,
        "residual_tolerance_justification": "64*n*binary32-unit-roundoff",
        "residual_tolerance_power2_shift": residual_tolerance_shift,
        "residual_certificate_denominator_lower_bound": "||b||inf",
        "residual_certificate_relation": (
            "certified-upper(||A*x-b||inf) <= tolerance*||b||inf; "
            "strictly implies documented normwise bound"
        ),
        "expected_solution_digest":
            f"{fnv64(map(float_word, solution)):016x}",
        "original_digests": {
            "a": f"{fnv64(map(float_word, original_a)):016x}",
            "b": f"{fnv64(map(float_word, original_b)):016x}",
        },
        "max_ulp": MAX_ULP, "scalar_only": True, "no_vpu": True,
        "vpu": False, "vector_isa": False, "gemmini": False, "mx": False,
        "muon_warps_per_core": warps_per_core,
        "robust_end_drain": robust_end_drain,
        "robust_spin_until_reset": robust_spin_until_reset,
        "robust_serial_fence": robust_serial_fence,
        "robust_post_spin": robust_post_spin,
        "reference_repository": "vortexgpgpu/vortex",
        "reference_checkout": str(REFERENCE_REPOSITORY.resolve()),
        "reference_revision": REFERENCE_REVISION,
        "reference_sources": references, "reference_seed": REFERENCE_SEED,
        "benchmark_contract": "Vortex-Rodinia-guassian-Fan1-Fan2-no-pivot",
        "stage_expected_digests": expected["stage_digests"],
        "final_digests": {
            "a": expected["stage_digests"][-1]["a"],
            "b": expected["stage_digests"][-1]["b"],
            "m": expected["stage_digests"][-1]["m"],
        },
        "kernel_source_sha256": source_digest(),
        "generated_sha256": {name: sha256(GENERATED / name)
                             for name in (*binaries, "config.h")},
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
    parser.add_argument("--robust-end-drain", action="store_true")
    parser.add_argument("--robust-spin-until-reset", action="store_true")
    parser.add_argument("--robust-serial-fence", action="store_true")
    parser.add_argument("--robust-post-spin", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    case = prepare(args.profile, args.sms, out,
                   loader=args.loader, warps_per_core=args.warps_per_core,
                   robust_end_drain=args.robust_end_drain,
                   robust_spin_until_reset=args.robust_spin_until_reset,
                   robust_serial_fence=args.robust_serial_fence,
                   robust_post_spin=args.robust_post_spin)
    if args.prepare_only:
        return
    env = build_environment()
    env["MU_ADDR_HASH"] = "1" if args.loader == "direct-hashed" else "0"
    with (out / "build.log").open("w") as log:
        subprocess.run(["make", "clean"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, env=env)
        subprocess.run(["make", "kernel.soc.elf"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, env=env)
    elf, device = out / "kernel.soc.elf", out / "kernel.radiance.elf"
    shutil.copy2(ROOT / "kernel.soc.elf", elf)
    shutil.copy2(ROOT / "kernel.radiance.elf", device)
    case.update({
        "status": "built", "elf": str(elf), "elf_sha256": sha256(elf),
        "bytes": elf.stat().st_size, "device_elf": str(device),
        "device_elf_sha256": sha256(device),
    })
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    print(json.dumps(case, indent=2))


if __name__ == "__main__":
    main()
