#!/usr/bin/env python3
"""Generate `data` for autocomp_fused_rmsnorm_qkv  --  RMSNorm fused into the matmul PROLOGUE.

Novel whole-Radiance pattern (prologue fusion, the one NOT previously demonstrated):
  A SIMT prologue reads the activation X[M][K] (bf16) from GMEM, computes RMSNorm per row
    xn = x * rsqrt(mean(x^2)+eps) * gamma
  quantizes xn -> fp8 e4m3, and writes the codes into the exact GMEM buffer that mxgemm reads
  its A operand from.  THEN the fp8 mxgemm runs.  No DRAM round-trip of a separate normalized
  activation tensor: the normalize + quantize happens in-place feeding the accelerator.

Golden = mx_golden( fp8( RMSNorm(X) ), W ), with the SAME fp8_e4m3_to_code() the kernel uses.
Verify is tolerance-based: the per-row rsqrt is one IEEE division, and a boundary-case fp8 code
flip changes one dot-product term slightly -- a relative tolerance absorbs any such 1-ULP-of-A
difference while still catching a broken prologue (which mis-scales whole rows).
"""
import math
import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MX_GOLDEN = pathlib.Path("/scratch/agustin/projects/autocomp/scripts/muon/mx_golden/mx_golden")
GROUP = 32
FP8_CODE = 0

M, N, K = 64, 64, 256
EPS = 1e-5


def bf16_round_trunc(x):
    """float32 -> bf16 (truncate) -> float32, matching the kernel's f_to_bf16/bf16_to_f."""
    u = x.astype(np.float32).view(np.uint32)
    return ((u >> 16) << 16).view(np.float32)


def f_to_bf16(x):
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def round_half_to_even(x):
    fl = math.floor(x)
    frac = x - fl
    if frac < 0.5:
        return int(fl)
    if frac > 0.5:
        return int(fl) + 1
    return int(fl) + 1 if (int(fl) & 1) else int(fl)


def fp8_e4m3_to_code(v):
    """fp8 e4m3 encode, RNE, subnormals flush to 0.

    The exponent E is taken from the float32 exponent field (E = expfield-127) rather than
    floor(log2f) -- identical result for normals, and it lets the KERNEL avoid libm log2f while
    staying bit-identical to this reference.
    """
    vf = np.float32(v)
    if float(vf) == 0.0 or not math.isfinite(float(vf)):
        return 0
    ubits = int(vf.view(np.uint32))
    s = (ubits >> 31) & 1
    av = np.float32(abs(float(vf)))
    expf = (int(av.view(np.uint32)) >> 23) & 0xFF
    bias, emin, emax = 7, -6, 8
    if expf == 0:            # subnormal float32 -> below emin
        return 0
    E = expf - 127
    if E < emin:
        return 0
    if E > emax:
        E_used, mant = emax, 6
    else:
        E_used = E
        base = np.float32(np.uint32((E_used + 127) << 23).view(np.float32))
        delta = np.float32(base / np.float32(8.0))
        k = round_half_to_even(float(np.float32((av - base) / delta)))
        if k >= 8:
            E_used += 1
            k = 0
            if E_used > emax:
                E_used, k = emax, 6
        else:
            hi = 6 if E_used == emax else 7
            k = min(max(k, 0), hi)
        mant = k
    return ((s << 7) | (((E_used + bias) & 0xF) << 3) | (mant & 0x7)) & 0xFF


def rand_fp8(rng, n):
    exp = rng.integers(0, 8, size=n, dtype=np.uint8)
    mant = rng.integers(0, 8, size=n, dtype=np.uint8)
    sign = rng.integers(0, 2, size=n, dtype=np.uint8)
    return ((sign << 7) | (exp << 3) | mant).astype(np.uint8)


def emit(f, ctype, name, dims, arr):
    w = {"uint8_t": 2, "uint16_t": 4, "uint32_t": 8}[ctype]
    rows = ["    " + ", ".join(f"0x{v:0{w}x}" for v in row) for row in arr]
    f.write(f"static const {ctype} {name}{dims} = {{\n")
    f.write(",\n".join(rows))
    f.write("\n};\n")


def main():
    GK, GN = K // GROUP, N // GROUP
    rng = np.random.default_rng(0x5A17)

    # Activation X (bf16) and per-feature RMSNorm weight gamma (bf16, near 1).
    X = bf16_round_trunc(rng.standard_normal((M, K)).astype(np.float32))
    gamma = bf16_round_trunc((0.75 + 0.5 * rng.random(K)).astype(np.float32))

    # --- reference RMSNorm + fp8 quant, matching the kernel's float32 path exactly ---
    A_codes = np.zeros((M, K), dtype=np.uint8)
    for m in range(M):
        row = X[m].astype(np.float32)
        ss = np.float32(0.0)
        for k in range(K):
            ss = np.float32(ss + np.float32(row[k] * row[k]))
        mean = np.float32(ss / np.float32(K))
        rms = np.float32(1.0) / np.float32(math.sqrt(float(np.float32(mean + np.float32(EPS)))))
        for k in range(K):
            xn = np.float32(np.float32(row[k] * rms) * gamma[k])
            A_codes[m, k] = fp8_e4m3_to_code(xn)

    # Weights (fp8) + block scales.  A scales are all 2^0 (the prologue owns the row scaling).
    B_in = rand_fp8(rng, K * N).reshape(K, N)
    SA = np.full((GK, M), 0x7F, dtype=np.uint8)
    SB = rng.integers(0x7B, 0x83, size=(GK, N), dtype=np.uint8)

    tmp = HERE / "_gen"
    tmp.mkdir(exist_ok=True)
    (tmp / "A.bin").write_bytes(A_codes.tobytes())
    (tmp / "B.bin").write_bytes(B_in.tobytes())
    (tmp / "SA.bin").write_bytes(SA.tobytes())
    (tmp / "SB.bin").write_bytes(SB.tobytes())
    subprocess.run(
        [str(MX_GOLDEN), str(M), str(N), str(K), str(tmp / "A.bin"), str(tmp / "B.bin"),
         str(tmp / "SA.bin"), str(tmp / "SB.bin"), str(tmp / "C.bin"), str(FP8_CODE)],
        check=True, env={"PATH": "/usr/bin:/bin"})
    C_bf16 = np.frombuffer((tmp / "C.bin").read_bytes(), dtype="<u2")[: M * N].reshape(M, N)
    gold = C_bf16.astype(np.uint16)

    X_bits = f_to_bf16(X)            # store X as bf16 [M][K]
    g_bits = f_to_bf16(gamma)        # gamma as bf16 [K]

    with open(HERE / "data", "w") as f:
        f.write("// @generated by gen_data.py -- RMSNorm(SIMT prologue) fused into fp8 mxgemm\n")
        f.write("#include <stdint.h>\n")
        f.write(f"#define MATMUL_M {M}\n#define MATMUL_N {N}\n#define MATMUL_K {K}\n")
        f.write(f"#define MATMUL_GK {GK}\n#define MATMUL_GN {GN}\n")
        f.write(f"#define VERIFY_COUNT {M * N}\n")
        f.write(f"#define RMS_EPS {EPS:.8e}f\n")
        emit(f, "uint16_t", "X_bf16", "[MATMUL_M][MATMUL_K]", X_bits)
        emit(f, "uint16_t", "gamma_bf16", "[MATMUL_K]", g_bits.reshape(1, K))
        emit(f, "uint8_t", "B_in", "[MATMUL_K][MATMUL_N]", B_in)
        emit(f, "uint8_t", "A_scales_row", "[MATMUL_GK][MATMUL_M]", SA)
        emit(f, "uint8_t", "B_scales_col", "[MATMUL_GK][MATMUL_N]", SB)
        f.write("__global uint8_t A_store[MATMUL_M * MATMUL_K] = {0};  // fp8 A written by prologue\n")
        f.write("__global uint16_t C_raw[MATMUL_M * MATMUL_N] = {0};\n")
        emit(f, "uint16_t", "gold_raw", "[MATMUL_M * MATMUL_N]", gold)
        f.write("static const uint8_t *A_in = "
                "reinterpret_cast<const uint8_t*>(reinterpret_cast<uint32_t>(&A_store[0]));\n")
        f.write("__global float tls_guard[1024] = {0};\n")
    print(f"wrote data  M={M} N={N} K={K} GK={GK} GN={GN}  nonzero A codes={int((A_codes!=0).sum())}/{M*K}")


if __name__ == "__main__":
    if not MX_GOLDEN.exists():
        raise SystemExit(f"build mx_golden first: {MX_GOLDEN}")
    main()
