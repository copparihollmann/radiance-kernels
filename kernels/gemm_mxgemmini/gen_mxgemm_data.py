#!/usr/bin/env python3
"""Generate mxgemm.data.<fmt>.m<M>n<N>k<K>.h headers for the standalone MX kernels.

The gemm_mxgemmini kernels #include a data header that carries the operands, the e8m0 block
scales, and the GOLDEN output. Only fp8 m64n64k64 was ever committed, so every other kernel
variant (128x128x512, 256x256x256, ...) could not even build. This regenerates them.

The golden comes from autocomp's `mx_golden`, which implements the HARDWARE accumulation
semantics (16-deep systolic array with the acc_e/acc_m precision schedule, one e8m0 scale per
32-element K group). It is validated bit-exact against BOTH real spike libgemmini AND the
upstream lib/mxgemmini golden model (see autocomp scripts/muon/xcheck_upstream_golden.py).
Do NOT substitute an idealized numpy/torch matmul: it disagrees with silicon on ~90% of
elements.

Operand magnitudes are capped (|a| <= 1) on purpose: with unrestricted fp8 (|a| up to 448) a
long dot product saturates the e4 block accumulators and essentially every golden element
comes out +/-inf -- a golden that any broken kernel "matches".

Usage:
  ./gen_mxgemm_data.py fp8 64 64 128      # one shape
  ./gen_mxgemm_data.py --all-fp8          # every missing fp8 header
"""
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MX_GOLDEN = pathlib.Path(
    "/scratch/agustin/projects/autocomp/scripts/muon/mx_golden/mx_golden"
)
GROUP = 32
FMT_CODE = {"fp8": 0, "fp6": 1, "fp4": 2}

# Shapes the committed kernels #include but that have no header.
MISSING_FP8 = [(64, 64, 128), (64, 64, 512), (128, 128, 128),
               (128, 128, 256), (128, 128, 512), (256, 256, 256)]
MISSING_FP4 = [(64, 64, 64), (64, 64, 128), (128, 128, 128),
               (128, 128, 256), (128, 128, 512), (128, 128, 1024)]


def rand_fp8(rng, n):
    """Random fp8 e4m3 codes with |value| <= 1 (exp field <= bias)."""
    exp = rng.integers(0, 8, size=n, dtype=np.uint8)
    mant = rng.integers(0, 8, size=n, dtype=np.uint8)
    sign = rng.integers(0, 2, size=n, dtype=np.uint8)
    return ((sign << 7) | (exp << 3) | mant).astype(np.uint8)


def emit(f, ctype, name, dims, arr):
    f.write(f"static const {ctype} {name}{dims} = {{\n")
    w = 2 if ctype == "uint8_t" else 4
    rows = ["    { " + ", ".join(f"0x{v:0{w}x}" for v in row) + " }" for row in arr]
    f.write(",\n".join(rows))
    f.write("\n};\n")


def rand_fp4(rng, n):
    """Random fp4 e2m1 nibbles with |value| <= 1.5 (exp field <= 1).

    e2m1: e=0 -> {0, 0.5}; e=1 -> {1.0, 1.5}; e=2 -> {2,3}; e=3 -> {4,6}. Capping e<=1 keeps
    partial sums inside the e4 block accumulators (same reason as the fp8 cap).
    """
    exp = rng.integers(0, 2, size=n, dtype=np.uint8)
    mant = rng.integers(0, 2, size=n, dtype=np.uint8)
    sign = rng.integers(0, 2, size=n, dtype=np.uint8)
    return ((sign << 3) | (exp << 1) | mant).astype(np.uint8)


def pack_nibbles_along_axis0(x):
    """[R, C] nibbles -> [R/2, C] bytes; row r goes to low nibble if r even, else high."""
    R, C = x.shape
    assert R % 2 == 0
    return ((x[1::2, :].astype(np.uint8) << 4) | (x[0::2, :].astype(np.uint8) & 0xF)).astype(np.uint8)


def pack_nibbles_along_axis1(x):
    """[R, C] nibbles -> [R, C/2] bytes; col c goes to low nibble if c even, else high."""
    R, C = x.shape
    assert C % 2 == 0
    return ((x[:, 1::2].astype(np.uint8) << 4) | (x[:, 0::2].astype(np.uint8) & 0xF)).astype(np.uint8)


def gen(fmt, M, N, K):
    assert fmt in ("fp8", "fp4"), "fp6 (LUT-indexed) not supported here yet"
    GK, GN = K // GROUP, N // GROUP
    rng = np.random.default_rng(hash((fmt, M, N, K)) & 0xFFFFFFFF)

    if fmt == "fp8":
        A = rand_fp8(rng, M * K).reshape(M, K)
        B = rand_fp8(rng, K * N).reshape(K, N)
    else:  # fp4: 32x32 PE tiles, nibble-packed (A along M, B along N)
        A_nib = rand_fp4(rng, M * K).reshape(M, K)
        B_nib = rand_fp4(rng, K * N).reshape(K, N)
        A = pack_nibbles_along_axis0(A_nib)  # [M/2][K]
        B = pack_nibbles_along_axis1(B_nib)  # [K][N/2]
    # e8m0 scale codes near 1.0 (0x7f == 2^0)
    SA = rng.integers(0x7B, 0x83, size=(GK, M), dtype=np.uint8)
    SB = rng.integers(0x7B, 0x83, size=(GK, N), dtype=np.uint8)

    tmp = HERE / "_gen" / f"{fmt}_{M}_{N}_{K}"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "A.bin").write_bytes(A.tobytes())
    (tmp / "B.bin").write_bytes(B.tobytes())
    (tmp / "SA.bin").write_bytes(SA.tobytes())
    (tmp / "SB.bin").write_bytes(SB.tobytes())

    def run_golden(out_fmt, cbin, sbin=None):
        env = {"PATH": "/usr/bin:/bin"}
        if out_fmt is not None:
            env["MX_OUT_FMT"] = str(out_fmt)
            if sbin:
                env["MX_SCALES_OUT"] = str(sbin)
        subprocess.run(
            [str(MX_GOLDEN), str(M), str(N), str(K), str(tmp / "A.bin"), str(tmp / "B.bin"),
             str(tmp / "SA.bin"), str(tmp / "SB.bin"), str(cbin), str(FMT_CODE[fmt])],
            check=True, env=env,
        )

    run_golden(None, tmp / "C_bf16.bin")                                  # fullout bf16
    run_golden(0, tmp / "C_q.bin", tmp / "C_scales.bin")                  # requant to fp8
    C_bf16 = np.frombuffer((tmp / "C_bf16.bin").read_bytes(), dtype="<u2").reshape(M, N)
    C_q = np.frombuffer((tmp / "C_q.bin").read_bytes(), dtype=np.uint8)[: M * N].reshape(M, N)
    C_sc = np.frombuffer((tmp / "C_scales.bin").read_bytes(), dtype=np.uint8)[: M * GN]
    C_sc = C_sc.reshape(M, GN).T.copy()  # header wants [GN][M]

    out = HERE / f"mxgemm.data.{fmt}.m{M}n{N}k{K}.h"
    guard = f"MXGEMM_DATA_{fmt.upper()}_M{M}N{N}K{K}_H"
    with open(out, "w") as f:
        f.write(f"// @generated by gen_mxgemm_data.py -- do not edit\n")
        f.write(f"// golden from autocomp mx_golden (hardware semantics; bit-exact vs spike\n")
        f.write(f"// and vs the upstream lib/mxgemmini golden model)\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n#include <stdint.h>\n\n")
        f.write(f"#define MATMUL_M {M}\n#define MATMUL_K {K}\n#define MATMUL_N {N}\n")
        f.write(f"#define MATMUL_GK {GK}\n#define MATMUL_GN {GN}\n\n")
        if fmt == "fp8":
            emit(f, "uint8_t", "A_in", "[MATMUL_M][MATMUL_K]", A)
            emit(f, "uint8_t", "B_in", "[MATMUL_K][MATMUL_N]", B)
        else:
            # sub-byte: A nibble-packed along M, B along N. The kernels alias
            # `A_in = &A_in_hw[0][0]` (see mxgemm.fp4.*.cpp).
            emit(f, "uint8_t", "A_in_hw", "[MATMUL_M / 2][MATMUL_K]", A)
            emit(f, "uint8_t", "B_in", "[MATMUL_K][MATMUL_N / 2]", B)
        emit(f, "uint8_t", "A_scales_row", "[MATMUL_GK][MATMUL_M]", SA)
        emit(f, "uint8_t", "B_scales_col", "[MATMUL_GK][MATMUL_N]", SB)
        emit(f, "uint8_t", "C_out", "[MATMUL_M][MATMUL_N]", C_q)
        emit(f, "uint8_t", "C_scales_row", "[MATMUL_GN][MATMUL_M]", C_sc)
        emit(f, "uint16_t", "C_out_bf16", "[MATMUL_M][MATMUL_N]", C_bf16)
        f.write(f"\n#endif // {guard}\n")
    print(f"wrote {out.name}  (M={M} N={N} K={K} GK={GK} GN={GN})")


if __name__ == "__main__":
    if not MX_GOLDEN.exists():
        raise SystemExit(f"build mx_golden first: {MX_GOLDEN}")
    if sys.argv[1:2] == ["--all-fp8"]:
        for M, N, K in MISSING_FP8:
            gen("fp8", M, N, K)
    elif sys.argv[1:2] == ["--all-fp4"]:
        for M, N, K in MISSING_FP4:
            gen("fp4", M, N, K)
    else:
        fmt, M, N, K = sys.argv[1], *map(int, sys.argv[2:5])
        gen(fmt, M, N, K)
