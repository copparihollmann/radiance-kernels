"""Independent NumPy reference for the block-scaled E4M3 / BF16 variant.

No libgemmini arithmetic functions are called. Mesh products truncate to three
fraction bits, saturate at 448, and flush exponents below -16 (PROD_FLOOR);
the 16-row accumulator precision ramp is (4,4)x8, (4,5)x2, (4,6)x5, (8,7).
Each 16-term result is scaled/accumulated in BF16 by MX. K blocks of 32
are restored to their FP32 powers of two and accumulated sequentially in FP32.
"""
import numpy as np


def bf16(value):
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    with np.errstate(over="ignore"):
        rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & 1)
    rounded = rounded & np.uint32(0xffff0000)
    return rounded.astype(np.uint32).view(np.float32)


def _float_round(value, exponent_bits, fraction_bits):
    value = np.asarray(value, dtype=np.float64)
    maximum = (2.0 - 2.0**-fraction_bits) * 2.0**((1 << (exponent_bits-1))-1)
    minimum_exponent = 2 - (1 << (exponent_bits-1))
    magnitude = np.abs(value)
    _, exponent = np.frexp(magnitude)
    step = np.exp2(np.maximum(exponent-1, minimum_exponent)-fraction_bits)
    rounded = np.rint(magnitude / step) * step
    rounded = np.where(rounded > maximum, np.inf, rounded)
    return np.copysign(rounded, value).astype(np.float32)


def _fp8(value, truncate=False):
    value = np.asarray(value, dtype=np.float32)
    magnitude = np.abs(value).astype(np.float64)
    _, exponent = np.frexp(magnitude)
    step = np.exp2((exponent-1 if truncate else np.maximum(exponent-1, -6))-3)
    scaled = magnitude / step
    rounded = (np.floor(scaled) if truncate else np.rint(scaled)) * step
    if truncate:
        rounded = np.where(exponent-1 < -16, 0.0, rounded)
    return np.copysign(np.minimum(rounded, 448.0), value).astype(np.float32)


def _normalize(value, axis):
    maximum = np.max(np.abs(value), axis=axis, keepdims=True)
    _, exponent = np.frexp(maximum)
    exponent = np.where(maximum == 0, 0, np.clip(exponent-1, -126, 126))
    scale = np.exp2(exponent).astype(np.float32)
    return _fp8(value/scale), scale


def linear(input, weight):
    input = np.asarray(input, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if input.shape[-1] != weight.shape[0] or weight.ndim != 2:
        raise ValueError("expected X[...,K] and W[K,N]")
    if not np.isfinite(input).all() or not np.isfinite(weight).all():
        raise ValueError("MX variant requires finite inputs")
    k, n = weight.shape
    flat = input.reshape(-1, k)
    result = np.zeros((len(flat), n), np.float32)
    ramp = [(4, 4)]*8 + [(4, 5)]*2 + [(4, 6)]*5 + [(8, 7)]
    for block in range(0, k, 32):
        count = min(k-block, 32)
        a = np.zeros((len(flat), 32), np.float32)
        b = np.zeros((32, n), np.float32)
        a[:, :count] = flat[:, block:block+count]
        b[:count] = weight[block:block+count]
        a, a_scale = _normalize(a, axis=1)
        b, b_scale = _normalize(b, axis=0)
        partial = np.zeros_like(result)
        for tile in (0, 16):
            acc = np.zeros_like(result)
            for t, (ebits, mbits) in enumerate(ramp):
                product = _fp8(a[:, tile+t, None] * b[None, tile+t, :], truncate=True)
                lhs = _float_round(acc, ebits, mbits)
                rhs = _float_round(product, ebits, mbits)
                acc = _float_round(lhs.astype(np.float64)+rhs.astype(np.float64), ebits, mbits)
            partial = bf16(partial + bf16(acc))
        result += (partial*a_scale)*b_scale
    return result.reshape((*input.shape[:-1], n))


def add(left, right):
    return bf16(bf16(left) + bf16(right))
