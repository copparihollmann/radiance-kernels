#!/usr/bin/env python3
"""Independent ordered-FP32 CPU oracle for the pinned Vortex pgain fixture."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable


FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fnv_words(words: Iterable[int]) -> int:
    digest = FNV_OFFSET
    for word in words:
        digest = ((digest ^ int(word)) * FNV_PRIME) & MASK64
    return digest


def fnv_bytes(values: Iterable[int]) -> int:
    digest = FNV_OFFSET
    for value in values:
        digest = ((digest ^ int(value)) * FNV_PRIME) & MASK64
    return digest


class GlibcRand:
    """glibc random()/rand() TYPE_3 sequence used by pinned Vortex main.cc."""

    def __init__(self, seed: int):
        if not 0 < seed < 0x80000000:
            raise ValueError("seed must be in [1, 2^31)")
        state = [seed]
        for _ in range(1, 31):
            state.append((16807 * state[-1]) % 2147483647)
        state.extend((state[0], state[1], state[2]))
        for index in range(34, 344):
            state.append((state[index - 31] + state[index - 3]) & 0xFFFFFFFF)
        self._state = state
        self._index = 344

    def rand(self) -> int:
        value = (self._state[self._index - 31] +
                 self._state[self._index - 3]) & 0xFFFFFFFF
        self._state.append(value)
        self._index += 1
        return value >> 1


@dataclass(frozen=True)
class Point:
    weight: float
    assign: int
    cost: float


@dataclass(frozen=True)
class Dataset:
    num: int
    dim: int
    seed: int
    center_stride: int
    candidate: int
    coordinates: tuple[float, ...]
    points: tuple[Point, ...]
    center_table: tuple[int, ...]
    centers: tuple[int, ...]

    @property
    def k(self) -> int:
        return len(self.centers)


@dataclass(frozen=True)
class OracleResult:
    work: tuple[float, ...]
    switch: bytes


def squared_distance(coordinates: tuple[float, ...], num: int, dim: int,
                     lhs: int, rhs: int) -> float:
    total = f32(0.0)
    for dimension in range(dim):
        difference = f32(coordinates[dimension * num + lhs] -
                         coordinates[dimension * num + rhs])
        product = f32(difference * difference)
        total = f32(total + product)
    return total


def make_dataset(num: int = 128, dim: int = 32, seed: int = 9,
                 center_stride: int = 8, candidate: int | None = None) -> Dataset:
    if num < 2 or dim < 1 or center_stride < 1:
        raise ValueError("num>=2, dim>=1, and center_stride>=1 are required")
    if candidate is None:
        candidate = num // 2
    if not 0 <= candidate < num:
        raise ValueError("candidate is outside the point domain")

    generator = GlibcRand(seed)
    coordinates = tuple(
        f32(f32(generator.rand() % 1000) / f32(1000.0))
        for _ in range(num * dim)
    )
    centers = tuple(range(0, num, center_stride))
    if not centers:
        centers = (0,)
    center_table = [-1] * num
    for compact, center in enumerate(centers):
        center_table[center] = compact

    points = []
    for point in range(num):
        best_center = 0
        best_distance = f32(1.0e30)
        for center in centers:
            distance = squared_distance(coordinates, num, dim, point, center)
            if distance < best_distance:
                best_distance = distance
                best_center = center
        weight = f32(1.0)
        points.append(Point(weight, best_center, f32(weight * best_distance)))

    return Dataset(
        num=num, dim=dim, seed=seed, center_stride=center_stride,
        candidate=candidate, coordinates=coordinates, points=tuple(points),
        center_table=tuple(center_table), centers=centers,
    )


def pgain(dataset: Dataset) -> OracleResult:
    """Execute the pinned memset + pgain final-output semantics."""
    row_width = dataset.k + 1
    work = [f32(0.0)] * (dataset.num * row_width)
    switch = bytearray(dataset.num)
    for point in range(dataset.num):
        x_cost = squared_distance(
            dataset.coordinates, dataset.num, dataset.dim,
            point, dataset.candidate,
        )
        x_cost = f32(x_cost * dataset.points[point].weight)
        current_cost = dataset.points[point].cost
        base = point * row_width
        if x_cost < current_cost:
            switch[point] = ord("1")
            work[base + dataset.k] = f32(x_cost - current_cost)
        else:
            assign = dataset.points[point].assign
            compact = dataset.center_table[assign]
            if compact < 0:
                raise ValueError("point assignment is not an open center")
            delta = f32(current_cost - x_cost)
            work[base + compact] = f32(work[base + compact] + delta)
    return OracleResult(tuple(work), bytes(switch))
