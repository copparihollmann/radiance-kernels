"""Prepare immutable MX query weights without changing their numerical policy.

Direct FP8 stores only selected matrices. INT4 lookup keeps the original packed
checkpoint and adds 68 bytes/column plus one exponent per K32/column group.
Metadata offsets are relative to its base, so the file is identical on RV32 and
the native reference host. No C++ conversion routine is used here.
"""
import struct

import numpy as np


def encode_fp8(value):
    value = np.asarray(value, np.float32)
    if not np.isfinite(value).all():
        raise ValueError('prepared MX weights must be finite')
    codes = np.arange(127, dtype=np.uint8)
    exponent, fraction = codes >> 3, codes & 7
    table = np.where(exponent == 0, fraction.astype(np.float64)*2.0**-9,
                     (1+fraction.astype(np.float64)/8)*np.exp2(exponent.astype(np.int32)-7))
    magnitude = np.minimum(np.abs(value).astype(np.float64), table[-1])
    upper = np.minimum(np.searchsorted(table, magnitude), 126)
    lower = np.maximum(upper-1, 0)
    dl, du = magnitude-table[lower], table[upper]-magnitude
    choose_upper = (du < dl) | ((du == dl) & ((lower & 1) != 0))
    rounded = np.where(choose_upper, upper, lower).astype(np.uint8)
    return rounded | (np.signbit(value).astype(np.uint8) << 7)


def exponents(weight):
    weight = np.asarray(weight, np.float32)
    if weight.ndim != 2 or not min(weight.shape) or not np.isfinite(weight).all():
        raise ValueError('expected a finite nonempty W[K,N]')
    result = []
    for block in range(0, len(weight), 32):
        maximum = np.abs(weight[block:block+32]).max(axis=0)
        _, exponent = np.frexp(maximum)
        result.append(np.where(maximum == 0, 0, np.clip(exponent-1, -126, 126)).astype(np.int8))
    return np.asarray(result)


def direct(weight):
    weight = np.asarray(weight, np.float32)
    scales = exponents(weight)
    packed = np.empty(weight.shape, np.uint8)
    for block, exponent in enumerate(scales):
        inverse = np.exp2(-exponent.astype(np.int32)).astype(np.float32)
        packed[block*32:(block+1)*32] = encode_fp8(weight[block*32:(block+1)*32]*inverse)
    return packed.tobytes(), scales.tobytes()


def int4_lookup(codes, scales):
    codes, scales = np.asarray(codes), np.asarray(scales, np.float32)
    if (codes.ndim != 2 or scales.shape != (codes.shape[1],) or
            np.any(codes < -8) or np.any(codes > 7) or not np.isfinite(scales).all()):
        raise ValueError('expected signed INT4 W[K,N] and one finite scale/column')
    weight = codes.astype(np.float32)*scales
    groups = exponents(weight)
    # -8 has the largest absolute integer code, including signed endpoints.
    maximum = np.abs(np.float32(-8)*scales)
    if not np.isfinite(maximum).all():
        raise ValueError('INT4 scale overflows FP32 dequantization')
    _, e = np.frexp(maximum)
    base = np.where(maximum == 0, 0, np.clip(e-1, -126, 126)).astype(np.int32)
    table = np.zeros((len(scales), 68), np.uint8)
    table[:, 0] = base.astype(np.int8).view(np.uint8)
    table[:, 1] = encode_fp8(np.zeros_like(scales)*scales)
    raw = np.arange(16)
    signed = np.where(raw < 8, raw, raw-16).astype(np.float32)
    dequantized = scales[:, None]*signed
    for band in range(4):
        exponent = base-band
        valid = exponent >= -126
        inverse = np.exp2(-exponent[valid]).astype(np.float32)
        table[valid, 4+16*band:20+16*band] = encode_fp8(dequantized[valid]*inverse[:, None])
    header = struct.pack('<II', 8, 8+table.nbytes)
    # Zero groups use the explicit zero code and exponent zero; nonzero groups
    # must select one of the four representable integer-magnitude bands.
    for block, exponent in enumerate(groups):
        nonzero = np.any(codes[block*32:(block+1)*32] != 0, axis=0) & (scales != 0)
        offset = base-exponent.astype(np.int32)
        if np.any(nonzero & ((offset < 0) | (offset > 3))):
            raise ValueError('INT4 group is outside the prepared exponent bands')
    return header+table.tobytes()+groups.tobytes()


def int8_lookup(codes, scales):
    """One base FP8 table, with exact normal-code exponent adjustment.

At the base exponent of 128*abs(scale), |integer|>=2 is FP8 normal.
Only +/-1 can be subnormal, so two extra codes at base-1 avoid double
rounding when a smaller K32 maximum increases their normalized exponent.
"""
    codes, scales = np.asarray(codes), np.asarray(scales,np.float32)
    if (codes.ndim != 2 or scales.shape != (codes.shape[1],) or
            np.any(codes < -128) or np.any(codes > 127) or not np.isfinite(scales).all()):
        raise ValueError('expected signed INT8 W[K,N] and finite column scales')
    weight = codes.astype(np.float32)*scales
    groups = exponents(weight)
    maximum = np.abs(np.float32(-128)*scales)
    if not np.isfinite(maximum).all():
        raise ValueError('INT8 scale overflows FP32 dequantization')
    _, e = np.frexp(maximum)
    base = np.where(maximum == 0,0,np.clip(e-1,-126,126)).astype(np.int32)
    raw = np.arange(256)
    signed = np.where(raw < 128,raw,raw-256).astype(np.float32)
    inverse = np.exp2(-base).astype(np.float32)
    table = np.zeros((len(scales),260),np.uint8)
    table[:,0] = base.astype(np.int8).view(np.uint8)
    table[:,1] = encode_fp8(np.zeros_like(scales)*scales)
    valid = base > -126
    alternate_inverse = np.exp2(-(base[valid]-1)).astype(np.float32)
    table[valid,2] = encode_fp8(scales[valid]*alternate_inverse)
    table[valid,3] = encode_fp8(-scales[valid]*alternate_inverse)
    table[:,4:] = encode_fp8((scales[:,None]*signed)*inverse[:,None])
    return struct.pack('<II',8,8+table.nbytes)+table.tobytes()+groups.tobytes()
