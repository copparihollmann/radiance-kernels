#!/usr/bin/env python3

from pathlib import Path
import random
import struct
from string import Template


ROWS = 16
COLS = 4096
SEED = 0


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f32_to_bf16_bits(value: float) -> int:
    bits = f32_bits(value)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


def bf16_bits_to_f32(bits: int) -> float:
    return bits_to_f32(bits << 16)


def bf16_add_bits(left: int, right: int) -> int:
    """One device BF16 add, rounded once to BF16."""
    return f32_to_bf16_bits(bf16_bits_to_f32(left) + bf16_bits_to_f32(right))


def bf16_fma_bits(accum: int, left: int, right: int) -> int:
    """One device BF16 fused multiply-add, rounded once to BF16."""
    return f32_to_bf16_bits(
        bf16_bits_to_f32(left) * bf16_bits_to_f32(right)
        + bf16_bits_to_f32(accum)
    )


def device_gemv_reference(
    a_bits: list[int], x_bits: list[int], rows: int, cols: int,
    *, lanes: int = 16, ilp: int = 1, unroll: int = 8,
) -> list[int]:
    """Reproduce the Muon kernel's BF16 accumulation and reduction order.

    The old reference accumulated the whole dot product in FP32 and rounded
    once.  The kernel executes BF16 FMAs in one chain per ILP slot and a BF16
    16-lane tree reduction, so that old reference was not numerically
    equivalent even though the matrix/vector data were identical.
    """
    if rows <= 0 or cols <= 0 or lanes <= 0 or ilp <= 0 or unroll <= 0:
        raise ValueError("GEMV dimensions and scheduling parameters must be positive")
    if unroll % ilp:
        raise ValueError("unroll must be divisible by ilp")
    if cols % (lanes * unroll):
        raise ValueError("cols must contain complete lane/unroll groups")
    if len(a_bits) != rows * cols or len(x_bits) != cols:
        raise ValueError("GEMV input dimensions do not match their payloads")

    output = []
    for row in range(rows):
        lane_sums = []
        row_offset = row * cols
        for lane in range(lanes):
            accum = [0] * ilp
            for base in range(lane, cols, lanes * unroll):
                for group in range(0, unroll, ilp):
                    for chain in range(ilp):
                        index = base + (group + chain) * lanes
                        accum[chain] = bf16_fma_bits(
                            accum[chain], a_bits[row_offset + index], x_bits[index]
                        )
            total = accum[0]
            for chain in range(1, ilp):
                total = bf16_add_bits(total, accum[chain])
            lane_sums.append(total)
        stride = 2
        while stride <= lanes:
            for lane in range(0, lanes, stride):
                lane_sums[lane] = bf16_add_bits(
                    lane_sums[lane], lane_sums[lane + stride // 2]
                )
            stride *= 2
        output.append(lane_sums[0])
    return output


def fmt_u16(values: list[int]) -> str:
    return ",".join(f"0x{value:04x}" for value in values) + ","


def fmt_bf16_bits(values: list[int]) -> str:
    return ",".join(f"{bf16_bits_to_f32(value):.9g}" for value in values) + ","


def main() -> None:
    rng = random.Random(SEED)

    a_bits = [
        f32_to_bf16_bits(rng.gauss(0.0, 1.0))
        for _ in range(ROWS * COLS)
    ]
    x_bits = [
        f32_to_bf16_bits(rng.gauss(0.0, 1.0))
        for _ in range(COLS)
    ]
    y_init_bits = [0 for _ in range(ROWS)]

    high_precision_y_bits = []
    for row in range(ROWS):
        acc = 0.0
        row_offset = row * COLS
        for col in range(COLS):
            acc += bf16_bits_to_f32(a_bits[row_offset + col]) * bf16_bits_to_f32(x_bits[col])
        high_precision_y_bits.append(f32_to_bf16_bits(acc))
    y_bits = device_gemv_reference(a_bits, x_bits, ROWS, COLS)

    data_template = Template(
        """
__global uint16_t A_raw[] = {
    $a_bits
};

__global uint16_t x_raw[] = {
    $x_bits
};

__global uint16_t y_raw[] = {
    $y_init_bits
};

const uint32_t m = $rows;
const uint32_t n = $cols;
"""
    )

    expected_template = Template(
        """
rows: $rows
cols: $cols
seed: $seed

const _Float16 A_bf16[] = {
    $a_bf16
};

const _Float16 x_bf16[] = {
    $x_bf16
};

const uint16_t expected_gemv_raw[] = {
    $y_bits
};

const uint16_t high_precision_gemv_raw[] = {
    $high_precision_y_bits
};

const _Float16 expected_gemv_bf16[] = {
    $y_bf16
};
"""
    )

    data_output = data_template.substitute(
        a_bits=fmt_u16(a_bits),
        x_bits=fmt_u16(x_bits),
        y_init_bits=fmt_u16(y_init_bits),
        rows=ROWS,
        cols=COLS,
    ).lstrip()

    expected_output = expected_template.substitute(
        a_bf16=fmt_bf16_bits(a_bits),
        x_bf16=fmt_bf16_bits(x_bits),
        y_bits=fmt_u16(y_bits),
        high_precision_y_bits=fmt_u16(high_precision_y_bits),
        y_bf16=fmt_bf16_bits(y_bits),
        rows=ROWS,
        cols=COLS,
        seed=SEED,
    ).lstrip()

    Path("data").write_text(data_output)
    Path("expected").write_text(expected_output)


if __name__ == "__main__":
    main()
