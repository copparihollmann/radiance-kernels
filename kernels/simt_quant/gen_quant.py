#!/usr/bin/env python3
"""MX-fp8 (e4m3) block quantization coverage kernel data.

Input: bf16 activation tile X[M][N] (M=64, N=512), stored as fp32 holding
bf16-exact values.  For each contiguous 32-element block along N:
    amax  = max |x| in block
    e     = ceil(log2(amax / FP8_MAX))          (FP8_MAX = 448.0)
    scale = 2^e,  e8m0 code = e + 127 (clamped [0,254])
    out_fp8[i] = quantize_e4m3(x[i] * 2^-e)      (round-to-nearest-even)

Everything below is implemented with the SAME integer/float bit operations the
device kernel uses, so the golden is bit-exact with the kernel output.  Outputs
are packed little-endian: 4 fp8 codes per uint32 word; one e8m0 code per uint32.
"""
import numpy as np

M, N = 64, 512
BLK = 32
NBLK_ROW = N // BLK            # 16
NUM_BLOCKS = M * NBLK_ROW      # 1024
FP8_MAX = 448.0

# ---- bit helpers on python ints (mirror the C exactly) --------------------
def f32_bits(x):
    return int(np.float32(x).view(np.uint32))

def bits_f32(u):
    return float(np.uint32(u & 0xFFFFFFFF).view(np.float32))

def round_bf16(u):
    """round-to-nearest-even fp32 bits -> bf16-truncated fp32 bits."""
    lsb = (u >> 16) & 1
    u2 = (u + 0x7FFF + lsb) & 0xFFFF0000
    return u2 & 0xFFFFFFFF

def block_exp(amax_bits):
    """e = ceil(log2(amax / 448)) via exact integer exponent extraction."""
    E = ((amax_bits >> 23) & 0xFF) - 127
    man = amax_bits & 0x7FFFFF
    # 448 = 1.75 * 2^8 ; m_a > 1.75  <=>  man > 0x600000
    return (E - 8) + (1 if man > 0x600000 else 0)

def round_shift(v, sh):
    """round-half-to-even of v >> sh (sh > 0)."""
    mask = (1 << sh) - 1
    low = v & mask
    half = 1 << (sh - 1)
    q = v >> sh
    if low > half or (low == half and (q & 1)):
        q += 1
    return q

def f32_to_e4m3(xf):
    """fp32 -> e4m3 code (uint8), round-to-nearest-even, saturate to +-448."""
    x = f32_bits(xf)
    sign = (x >> 31) & 1
    absx = x & 0x7FFFFFFF
    s = (sign << 7) & 0xFF
    if absx == 0:
        return s
    axf = bits_f32(absx)
    if axf >= 448.0:
        return s | 0x7E
    e = ((absx >> 23) & 0xFF) - 127
    man = absx & 0x7FFFFF
    sig = (1 << 23) | man            # value = sig * 2^(e-23)
    if e < -6:                       # subnormal target (exp fixed at -6)
        q = round_shift(sig, 14 - e) # value / 2^-9
        if q >= 8:                   # rolled up into min-normal
            out = (1 << 3) | (q - 8)
        else:
            out = q
    else:                            # normal target
        r = round_shift(sig, 20)     # sig/2^20 in [8,16]
        E = e
        if r == 16:
            r = 8
            E += 1
        expf = E + 7
        mant = r - 8
        if expf > 15 or (expf == 15 and mant == 7):
            return s | 0x7E          # saturate
        out = ((expf << 3) | mant) & 0xFF
    return s | out

def pow2_neg_e(e):
    """2^-e as an exact fp32, built from the bit pattern (mirrors device)."""
    return bits_f32(((127 - e) & 0xFFFFFFFF) << 23)

# ---- generate bf16 inputs -------------------------------------------------
rng = np.random.default_rng(1234)
X = rng.standard_normal((M, N)).astype(np.float32)
X = np.tanh(X).astype(np.float32)
# give each 32-block its own magnitude so the per-block scale, subnormal, and
# near-saturation paths all get exercised (block amax spans ~2^-6 .. ~2^6).
mag_exps = rng.integers(-6, 7, size=(M, NBLK_ROW)).astype(np.float32)
mag = np.power(np.float32(2.0), mag_exps).astype(np.float32)      # per block
mag = np.repeat(mag, BLK, axis=1).astype(np.float32)             # broadcast to elems
X = (X * mag).astype(np.float32)
# round every element to bf16 (stored back as fp32)
Xb = np.empty_like(X)
flatX = X.reshape(-1)
flatXb = Xb.reshape(-1)
for i in range(flatX.size):
    flatXb[i] = bits_f32(round_bf16(f32_bits(flatX[i])))

# ---- reference quantization ----------------------------------------------
codes = np.zeros(M * N, dtype=np.uint8)
scale_codes = np.zeros(NUM_BLOCKS, dtype=np.uint8)
for b in range(NUM_BLOCKS):
    base = b * BLK
    blk = flatXb[base:base + BLK]
    amax = np.float32(0.0)
    for v in blk:
        av = np.float32(abs(v))
        if av > amax:
            amax = av
    if float(amax) == 0.0:
        scale_codes[b] = 0
        # codes already 0
        continue
    e = block_exp(f32_bits(amax))
    sc = e + 127
    sc = 0 if sc < 0 else (254 if sc > 254 else sc)
    scale_codes[b] = sc
    p2 = pow2_neg_e(e)
    for j in range(BLK):
        xs = np.float32(np.float32(blk[j]) * np.float32(p2))
        codes[base + j] = f32_to_e4m3(xs)

# pack 4 fp8 codes / uint32 word, little-endian element order
packed = codes.view(np.uint32) if False else None
packed = np.zeros(M * N // 4, dtype=np.uint32)
for w in range(packed.size):
    packed[w] = (int(codes[4 * w + 0])
                 | (int(codes[4 * w + 1]) << 8)
                 | (int(codes[4 * w + 2]) << 16)
                 | (int(codes[4 * w + 3]) << 24))
scales32 = scale_codes.astype(np.uint32)

# ---- emit -----------------------------------------------------------------
def emit_f32(f, name, arr):
    flat = arr.reshape(-1)
    f.write(f"__global float {name}[{flat.size}] = {{\n")
    f.write(",\n".join(f"{v:.9e}f" for v in flat))
    f.write("\n};\n")

def emit_u32(f, name, arr):
    flat = arr.reshape(-1)
    f.write(f"__global uint32_t {name}[{flat.size}] = {{\n")
    f.write(",\n".join(f"0x{int(v)&0xFFFFFFFF:08x}u" for v in flat))
    f.write("\n};\n")

out = "/scratch/agustin/projects/radiance-kernels/kernels/autocomp_simt_quant/data"
with open(out, "w") as f:
    f.write("// generated by gen_quant.py - do not edit\n")
    f.write("// MX-fp8 (e4m3) per-32-block quantization of a bf16 activation tile\n")
    f.write(f"static const uint32_t M = {M};\n")
    f.write(f"static const uint32_t N = {N};\n")
    f.write(f"static const uint32_t BLK = {BLK};\n")
    f.write(f"static const uint32_t NUM_BLOCKS = {NUM_BLOCKS};\n")
    f.write(f"#define PACKED_COUNT {M*N//4}\n")
    f.write(f"#define SCALE_COUNT {NUM_BLOCKS}\n")
    # single contiguous golden: packed fp8 words first, then e8m0 scale words.
    # keeps verify() to one tight loop over two arrays -> fewer distinct
    # registers on the main()-running warps (rename-budget headroom).
    gold_all = np.concatenate([packed, scales32]).astype(np.uint32)
    f.write(f"#define TOTAL_COUNT {gold_all.size}\n")
    emit_f32(f, "X_raw", Xb)
    emit_u32(f, "gold_all", gold_all)

# quick sanity print
nz = int((codes != 0).sum())
print(f"wrote {out}")
print(f"blocks={NUM_BLOCKS} elems={M*N} nonzero_fp8={nz} "
      f"scale_min={int(scale_codes.min())} scale_max={int(scale_codes.max())}")
print("sample codes[:8]=", [int(c) for c in codes[:8]])
print("sample scale[:4]=", [int(c) for c in scale_codes[:4]])
