#!/usr/bin/env python3
"""Prepare and build the pinned one-step scalar Radiance LBM workload."""

from __future__ import annotations

from array import array
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
DEFAULT_REFERENCE = Path(
    "/scratch/agustin/radiance-fpga-1a/reference/vortex"
)

REFERENCE_REPOSITORY = "vortexgpgpu/vortex"
REFERENCE_REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"
REFERENCE_FILES = {
    "tests/opencl/lbm/Makefile": {
        "blob": "85477be0064228b261b50c12394136e4b1b66517",
        "sha256": "015b20678dd8c002cf0eb9e0c9276ecfff187bc02c7959f7d67a5b7660df5592",
        "bytes": 598,
    },
    "tests/opencl/lbm/kernel.cl": {
        "blob": "3f34ea5ef25943ac7eeb18eead429f70ec4cf807",
        "sha256": "174ff2dfbe7b629c4277b22adfc186c551eb20c16cc3b6ce04937471fb0a083b",
        "bytes": 5707,
    },
    "tests/opencl/lbm/layout_config.h": {
        "blob": "94f3d1f41d4850c752682e3719fb39e944fe60cd",
        "sha256": "5109355f422a788b44efeaeaf5bddd6153c53b2ae7cd2f2a21a7bb6db3735be6",
        "bytes": 2125,
    },
    "tests/opencl/lbm/lbm_macros.h": {
        "blob": "24fad43205f11da1c05cc8aa5895e7aa2688d3f4",
        "sha256": "c789733f5e051b360941ba6acba2758826ecc43c4f5e6be15f77f8f060f303cb",
        "bytes": 6818,
    },
    "tests/opencl/lbm/lbm.c": {
        "blob": "dbe58caa6b7f8921843e420182a092fc17ca24d2",
        "sha256": "4b7f2af7d222988194b929fda0e0745fa54ecdc2e6c28ab0f69b56ca3bf83e94",
        "bytes": 9969,
    },
    "tests/opencl/lbm/main.cc": {
        "blob": "2da74351082d8de176a777d40f3a5ac8ca450fce",
        "sha256": "2217f8a440e49f423ec29267af66fea00cee935428620f1aa63a9532751bb740",
        "bytes": 10961,
    },
    "tests/opencl/lbm/32_32_8_ldc.of": {
        "blob": "169c353fd681df27d387bc02a2e52eec4c544d9c",
        "sha256": "3d9454aaf7abb4d4b4d9866a1b11d0024e770ad25887b43d0bbe8d951517f2c6",
        "bytes": 8456,
    },
    "tests/opencl/lbm/32_32_8_ldc.gold": {
        "blob": "f58031d3d7fe61f0a9bc50bf994e0367b4cf507a",
        "sha256": "727feb99cd3db0f6951ab7a2bbd8fd850edb007837e774590448dff652923f52",
        "bytes": 98304,
    },
}

SIZE_X = 32
SIZE_Y = 32
SIZE_Z = 8
PADDING_X = 8
PADDING_Y = 0
PADDING_Z = 4
PADDED_X = SIZE_X + PADDING_X
PADDED_Y = SIZE_Y + PADDING_Y
PADDED_Z = SIZE_Z + PADDING_Z
CELL_ENTRIES = 20
POPULATIONS = 19
ACTIVE_CELLS = SIZE_X * SIZE_Y * SIZE_Z
TOTAL_PADDED_CELLS = PADDED_X * PADDED_Y * PADDED_Z
MARGIN_FLOATS = CELL_ENTRIES * 2 * PADDED_X * PADDED_Y
GROUP_CELLS = 4
GROUP_FLOATS = GROUP_CELLS * CELL_ENTRIES
GROUP_BYTES = GROUP_FLOATS * 4
GROUP_COUNT = ACTIVE_CELLS // GROUP_CELLS
CACHE_LINE_BYTES = 64

# Gather offsets indexed by the pinned C,N,S,E,W,T,B,...,WB enumeration.
GATHER_OFFSETS = (
    (0, 0, 0),
    (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0),
    (0, 0, -1), (0, 0, 1),
    (-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0),
    (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
    (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),
)

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def word_float(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value))[0]


def fadd(lhs: float, rhs: float) -> float:
    return f32(lhs + rhs)


def fsub(lhs: float, rhs: float) -> float:
    return f32(lhs - rhs)


def fmul(lhs: float, rhs: float) -> float:
    return f32(lhs * rhs)


def fdiv(lhs: float, rhs: float) -> float:
    return f32(lhs / rhs)


def digest(words: Iterable[int]) -> int:
    value = FNV_OFFSET
    for word in words:
        value = ((value ^ int(word)) * FNV_PRIME) & MASK64
    return value


def git_output(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as error:
        detail = error.output.decode(errors="replace").strip()
        raise ValueError(f"cannot read pinned reference: {detail}") from error


def reference_blob(repo: Path, path: str) -> bytes:
    return git_output(repo, "show", f"{REFERENCE_REVISION}:{path}")


def parse_define(text: str, name: str) -> int:
    match = re.search(
        rf"^\s*#define\s+{re.escape(name)}\s+\(?\s*(\d+)\s*\)?\s*$",
        text, re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"missing integer definition {name}")
    return int(match.group(1))


def verify_reference(repo: Path) -> dict[str, bytes]:
    repo = repo.resolve()
    if not repo.is_dir():
        raise ValueError(f"reference repository does not exist: {repo}")
    head = git_output(repo, "rev-parse", "HEAD").decode().strip()
    if head != REFERENCE_REVISION:
        raise ValueError(
            f"reference HEAD is {head}; required {REFERENCE_REVISION}"
        )

    blobs: dict[str, bytes] = {}
    for path, pinned in REFERENCE_FILES.items():
        blob_id = git_output(
            repo, "rev-parse", f"{REFERENCE_REVISION}:{path}"
        ).decode().strip()
        data = reference_blob(repo, path)
        if blob_id != pinned["blob"]:
            raise ValueError(f"reference blob mismatch for {path}")
        if len(data) != pinned["bytes"]:
            raise ValueError(f"reference byte count mismatch for {path}")
        if sha256_bytes(data) != pinned["sha256"]:
            raise ValueError(f"reference SHA-256 mismatch for {path}")
        blobs[path] = data

    layout = blobs["tests/opencl/lbm/layout_config.h"].decode("ascii")
    dimensions = {
        "SIZE_X": SIZE_X, "SIZE_Y": SIZE_Y, "SIZE_Z": SIZE_Z,
        "PADDING_X": PADDING_X, "PADDING_Y": PADDING_Y,
        "PADDING_Z": PADDING_Z,
    }
    for name, expected in dimensions.items():
        if parse_define(layout, name) != expected:
            raise ValueError(f"official dimension {name} changed")
    if "#if 1\n#define GATHER" not in layout.replace("\r\n", "\n"):
        raise ValueError("official layout is no longer the GATHER contract")
    makefile = blobs["tests/opencl/lbm/Makefile"].decode("ascii")
    if not re.search(r"^OPTS\s*\?=\s*1\s+-i\s+.*32_32_8_ldc\.of\s*$",
                     makefile, re.MULTILINE):
        raise ValueError("official default is no longer exactly one timestep")
    if len(blobs["tests/opencl/lbm/32_32_8_ldc.gold"]) != ACTIVE_CELLS * 3 * 4:
        raise ValueError("official gold does not contain 3*8192 float32 values")
    return blobs


def grid_index(x: int, y: int, z: int, entry: int) -> int:
    index = MARGIN_FLOATS + entry + CELL_ENTRIES * (
        x + y * PADDED_X + z * PADDED_X * PADDED_Y
    )
    if not 0 <= index < TOTAL_PADDED_CELLS * CELL_ENTRIES:
        raise ValueError(f"grid index outside padded allocation: {(x, y, z, entry)}")
    return index


def set_flag(grid: list[float], x: int, y: int, z: int, flag: int) -> None:
    index = grid_index(x, y, z, POPULATIONS)
    grid[index] = word_float(float_word(grid[index]) | flag)


def initialize_grid(obstacles: bytes) -> list[float]:
    expected_bytes = ACTIVE_CELLS + (SIZE_Y + 1) * SIZE_Z
    if len(obstacles) != expected_bytes:
        raise ValueError("official obstacle byte count changed")
    grid = [f32(0.0)] * (TOTAL_PADDED_CELLS * CELL_ENTRIES)
    weights = [f32(1.0 / 3.0)] + [f32(1.0 / 18.0)] * 6
    weights += [f32(1.0 / 36.0)] * 12
    for z in range(SIZE_Z):
        for y in range(SIZE_Y):
            for x in range(SIZE_X):
                for entry, (dx, dy, dz) in enumerate(GATHER_OFFSETS):
                    grid[grid_index(x + dx, y + dy, z + dz, entry)] = weights[entry]
                grid[grid_index(x, y, z, POPULATIONS)] = f32(0.0)

    cursor = 0
    for z in range(SIZE_Z):
        for y in range(SIZE_Y):
            for x in range(SIZE_X):
                if obstacles[cursor] != ord("."):
                    set_flag(grid, x, y, z, 1)
                cursor += 1
            if obstacles[cursor] != ord("\n"):
                raise ValueError("official obstacle row framing changed")
            cursor += 1
        if obstacles[cursor] != ord("\n"):
            raise ValueError("official obstacle slice framing changed")
        cursor += 1
    if cursor != len(obstacles):
        raise ValueError("official obstacle parse did not consume the file")

    # Exact pinned LBM_initializeSpecialCellsForLDC loop and precedence.
    for z in range(-2, SIZE_Z + 2):
        for y in range(SIZE_Y):
            for x in range(SIZE_X):
                boundary = (
                    x == 0 or x == SIZE_X - 1 or
                    y == 0 or y == SIZE_Y - 1 or
                    z == 0 or z == SIZE_Z - 1
                )
                if boundary:
                    set_flag(grid, x, y, z, 1)
                elif ((z == 1 or z == SIZE_Z - 2) and
                      1 < x < SIZE_X - 2 and 1 < y < SIZE_Y - 2):
                    set_flag(grid, x, y, z, 2)
    return grid


def gather(grid: list[float], x: int, y: int, z: int) -> list[float]:
    return [
        grid[grid_index(x + dx, y + dy, z + dz, entry)]
        for entry, (dx, dy, dz) in enumerate(GATHER_OFFSETS)
    ]


def ordered_sum(values: list[float]) -> float:
    total = values[0]
    for value in values[1:]:
        total = fadd(total, value)
    return total


def momentum(pop: list[float]) -> tuple[float, float, float]:
    ux = fsub(pop[3], pop[4])
    ux = fadd(ux, pop[7]); ux = fsub(ux, pop[8])
    ux = fadd(ux, pop[9]); ux = fsub(ux, pop[10])
    ux = fadd(ux, pop[15]); ux = fadd(ux, pop[16])
    ux = fsub(ux, pop[17]); ux = fsub(ux, pop[18])
    uy = fsub(pop[1], pop[2])
    uy = fadd(uy, pop[7]); uy = fadd(uy, pop[8])
    uy = fsub(uy, pop[9]); uy = fsub(uy, pop[10])
    uy = fadd(uy, pop[11]); uy = fadd(uy, pop[12])
    uy = fsub(uy, pop[13]); uy = fsub(uy, pop[14])
    uz = fsub(pop[5], pop[6])
    uz = fadd(uz, pop[11]); uz = fsub(uz, pop[12])
    uz = fadd(uz, pop[13]); uz = fsub(uz, pop[14])
    uz = fadd(uz, pop[15]); uz = fsub(uz, pop[16])
    uz = fadd(uz, pop[17]); uz = fsub(uz, pop[18])
    return ux, uy, uz


def collide_axis(old: float, velocity: float, sign: float,
                 temp1: float, temp2: float, u2: float) -> float:
    inner = fadd(fmul(f32(4.5), velocity), f32(sign * 3.0))
    equilibrium = fsub(fmul(velocity, inner), u2)
    return fadd(fmul(temp2, old), fmul(temp1, equilibrium))


def collide_diagonal(old: float, velocity: float, temp1: float,
                     temp2: float, u2: float) -> float:
    inner = fadd(fmul(f32(4.5), velocity), f32(3.0))
    equilibrium = fsub(fmul(velocity, inner), u2)
    return fadd(fmul(temp2, old), fmul(temp1, equilibrium))


def collide_cell(source: list[float], x: int, y: int, z: int) -> list[float]:
    pop = gather(source, x, y, z)
    flags = float_word(source[grid_index(x, y, z, POPULATIONS)])
    if flags & 1:
        for lhs, rhs in (
            (1, 2), (3, 4), (5, 6), (7, 10), (8, 9),
            (11, 14), (12, 13), (15, 18), (16, 17),
        ):
            pop[lhs], pop[rhs] = pop[rhs], pop[lhs]
        return pop

    rho = ordered_sum(pop)
    ux, uy, uz = momentum(pop)
    ux = fdiv(ux, rho); uy = fdiv(uy, rho); uz = fdiv(uz, rho)
    if flags & 2:
        ux, uy, uz = f32(0.005), f32(0.002), f32(0.0)

    squared = fadd(fmul(ux, ux), fmul(uy, uy))
    squared = fadd(squared, fmul(uz, uz))
    u2 = fsub(fmul(f32(1.5), squared), f32(1.0))
    temp_base = fmul(f32(1.95), rho)
    temp1 = fmul(f32(1.0 / 3.0), temp_base)
    # kernel.cl repeats these assignments before writing tempC.
    temp_base = fmul(f32(1.95), rho)
    temp1 = fmul(f32(1.0 / 3.0), temp_base)
    temp2 = fsub(f32(1.0), f32(1.95))
    pop[0] = fadd(fmul(temp2, pop[0]), fmul(temp1, f32(-u2)))

    temp1 = fmul(f32(1.0 / 18.0), temp_base)
    pop[1] = collide_axis(pop[1], uy, +1.0, temp1, temp2, u2)
    pop[2] = collide_axis(pop[2], uy, -1.0, temp1, temp2, u2)
    pop[5] = collide_axis(pop[5], uz, +1.0, temp1, temp2, u2)
    pop[6] = collide_axis(pop[6], uz, -1.0, temp1, temp2, u2)
    pop[3] = collide_axis(pop[3], ux, +1.0, temp1, temp2, u2)
    pop[4] = collide_axis(pop[4], ux, -1.0, temp1, temp2, u2)

    temp1 = fmul(f32(1.0 / 36.0), temp_base)
    velocities = (
        (11, fadd(uy, uz)), (12, fsub(uy, uz)),
        (13, fadd(f32(-uy), uz)), (14, fsub(f32(-uy), uz)),
        (7, fadd(ux, uy)), (9, fsub(ux, uy)),
        (15, fadd(ux, uz)), (16, fsub(ux, uz)),
        (8, fadd(f32(-ux), uy)), (10, fsub(f32(-ux), uy)),
        (17, fadd(f32(-ux), uz)), (18, fsub(f32(-ux), uz)),
    )
    for entry, velocity in velocities:
        pop[entry] = collide_diagonal(
            pop[entry], velocity, temp1, temp2, u2
        )
    return pop


def reference_timestep(source: list[float]) -> tuple[list[float], list[int]]:
    destination = source.copy()
    packed: list[int] = []
    for z in range(SIZE_Z):
        for y in range(SIZE_Y):
            for x in range(SIZE_X):
                pop = collide_cell(source, x, y, z)
                for entry, value in enumerate(pop):
                    destination[grid_index(x, y, z, entry)] = value
                    packed.append(float_word(value))
    return destination, packed


def velocity_field(grid: list[float]) -> list[int]:
    words: list[int] = []
    for z in range(SIZE_Z):
        for y in range(SIZE_Y):
            for x in range(SIZE_X):
                pop = gather(grid, x, y, z)
                rho = ordered_sum(pop)
                ux, uy, uz = momentum(pop)
                words.extend((
                    float_word(fdiv(ux, rho)),
                    float_word(fdiv(uy, rho)),
                    float_word(fdiv(uz, rho)),
                ))
    return words


def official_contract(blobs: dict[str, bytes]) -> dict:
    source = initialize_grid(blobs["tests/opencl/lbm/32_32_8_ldc.of"])
    destination, expected_populations = reference_timestep(source)
    velocity = velocity_field(destination)
    gold_data = blobs["tests/opencl/lbm/32_32_8_ldc.gold"]
    gold_words = list(struct.unpack(f"<{ACTIVE_CELLS * 3}I", gold_data))
    if velocity != gold_words:
        mismatches = sum(a != b for a, b in zip(velocity, gold_words))
        raise ValueError(
            f"ordered one-step oracle differs from bundled gold at "
            f"{mismatches} velocity words"
        )
    output_initial = source.copy()
    poison = word_float(0x7FC00000)
    for z in range(SIZE_Z):
        for y in range(SIZE_Y):
            for x in range(SIZE_X):
                for entry in range(POPULATIONS):
                    output_initial[grid_index(x, y, z, entry)] = poison
    return {
        "source": source,
        "output_initial": output_initial,
        "expected_output": destination,
        "expected_populations": expected_populations,
        "gold_velocity": gold_words,
    }


def group_boundaries(sms: int) -> list[int]:
    if sms not in (1, 2, 4):
        raise ValueError("sms must be 1, 2, or 4")
    if GROUP_COUNT % sms:
        raise ValueError("official group count is not divisible by SM count")
    return [index * GROUP_COUNT // sms for index in range(sms + 1)]


def partition_audit(sms: int) -> dict:
    boundaries = group_boundaries(sms)
    owners = []
    claimed_groups: list[int] = []
    cache_line_owner: dict[int, int] = {}
    for sm, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        groups = list(range(begin, end))
        claimed_groups.extend(groups)
        first_cell = begin * GROUP_CELLS
        last_cell = end * GROUP_CELLS
        line_count = 0
        for group in groups:
            cell = group * GROUP_CELLS
            x = cell % SIZE_X
            yz = cell // SIZE_X
            y = yz % SIZE_Y
            z = yz // SIZE_Y
            start = grid_index(x, y, z, 0) * 4
            if start % CACHE_LINE_BYTES:
                raise ValueError("LBM group is not cache-line aligned")
            if GROUP_BYTES != 5 * CACHE_LINE_BYTES:
                raise ValueError("LBM group is not exactly five cache lines")
            for line in range(start // CACHE_LINE_BYTES,
                              (start + GROUP_BYTES) // CACHE_LINE_BYTES):
                if line in cache_line_owner:
                    raise ValueError("cache line is shared by physical SMs")
                cache_line_owner[line] = sm
                line_count += 1
        owners.append({
            "physical_threadblock_id": sm,
            "group_begin_inclusive": begin,
            "group_end_exclusive": end,
            "group_count": end - begin,
            "cell_begin_inclusive": first_cell,
            "cell_end_exclusive": last_cell,
            "cell_count": last_cell - first_cell,
            "population_writes": (last_cell - first_cell) * POPULATIONS,
            "destination_cache_lines": line_count,
        })
    if claimed_groups != list(range(GROUP_COUNT)):
        raise ValueError("partition has a gap or overlap")
    serialized = json.dumps(owners, sort_keys=True).encode()
    return {
        "scheme": "physical-threadblock-contiguous-groups-of-four-complete-cells",
        "sm_count": sms,
        "group_cells": GROUP_CELLS,
        "group_floats_including_flags": GROUP_FLOATS,
        "group_bytes": GROUP_BYTES,
        "cache_lines_per_group": GROUP_BYTES // CACHE_LINE_BYTES,
        "group_boundaries": boundaries,
        "owners": owners,
        "complete": True,
        "overlap": False,
        "cache_line_safe": True,
        "ownership_sha256": hashlib.sha256(serialized).hexdigest(),
    }


def write_words(path: Path, words: Iterable[int]) -> None:
    values = array("I", (int(word) for word in words))
    if values.itemsize != 4:
        raise RuntimeError("native unsigned int is not 32 bits")
    if sys.byteorder != "little":
        values.byteswap()
    path.write_bytes(values.tobytes())


def source_words(values: Iterable[float]) -> list[int]:
    return [float_word(value) for value in values]


def source_digest() -> str:
    value = hashlib.sha256()
    paths = (
        ROOT.parent / "addr_hash_stamp.mk",
        ROOT.parents[1] / "lib/include/nightly/device.h",
        ROOT.parents[1] / "lib/include/nightly/host.h",
        ROOT / "Makefile",
        ROOT / "emit_symbols.py",
        ROOT / "functional_smoke.py",
        ROOT / "host.cpp",
        ROOT / "kernel.cpp",
        ROOT / "lbm.py",
        ROOT / "partition_audit.py",
    )
    for path in paths:
        value.update(path.relative_to(ROOT.parents[1]).as_posix().encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def kernel_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT.parents[1]), "rev-parse", "HEAD"], text=True,
    ).strip()


def prepare(out: Path, *, sms: int = 1, loader: str = "logical",
            reference: Path = DEFAULT_REFERENCE, warps_per_core: int = 8,
            robust_end_drain: bool = False,
            robust_spin_until_reset: bool = False,
            robust_serial_fence: bool = False,
            robust_post_spin: int = 500) -> dict:
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

    audit = partition_audit(sms)
    blobs = verify_reference(reference)
    contract = official_contract(blobs)
    input_words = source_words(contract["source"])
    output_initial_words = source_words(contract["output_initial"])
    expected_output_words = source_words(contract["expected_output"])
    expected_words = contract["expected_populations"]
    gold_words = contract["gold_velocity"]
    expected_output_digest = digest(expected_output_words)
    expected_digest = digest(expected_words)
    gold_digest = digest(gold_words)

    GENERATED.mkdir(exist_ok=True)
    write_words(GENERATED / "input.bin", input_words)
    write_words(GENERATED / "output_initial.bin", output_initial_words)
    write_words(GENERATED / "expected_populations.bin", expected_words)
    write_words(GENERATED / "gold_velocity.bin", gold_words)

    boundaries = "{" + ",".join(
        f"{value}u" for value in audit["group_boundaries"]
    ) + "}"
    config = [
        "#pragma once",
        f"#define HPC_SM_COUNT {sms}u",
        f"#define NIGHTLY_CLUSTERS {sms}",
        f"#define NIGHTLY_OCC {warps_per_core}",
        f"#define NIGHTLY_ROBUST_GMEM_END {int(robust_end_drain)}",
        f"#define NIGHTLY_ROBUST_SPIN_UNTIL_RESET {int(robust_spin_until_reset)}",
        f"#define NIGHTLY_ROBUST_SERIAL_FENCE {int(robust_serial_fence)}",
        f"#define NIGHTLY_ROBUST_POST_SPIN {robust_post_spin}u",
        f"#define LBM_SIZE_X {SIZE_X}u",
        f"#define LBM_SIZE_Y {SIZE_Y}u",
        f"#define LBM_SIZE_Z {SIZE_Z}u",
        f"#define LBM_PADDED_X {PADDED_X}u",
        f"#define LBM_PADDED_Y {PADDED_Y}u",
        f"#define LBM_PADDED_Z {PADDED_Z}u",
        f"#define LBM_CELL_ENTRIES {CELL_ENTRIES}u",
        f"#define LBM_POPULATIONS {POPULATIONS}u",
        f"#define LBM_ACTIVE_CELLS {ACTIVE_CELLS}u",
        f"#define LBM_TOTAL_FLOATS {TOTAL_PADDED_CELLS * CELL_ENTRIES}u",
        f"#define LBM_MARGIN_FLOATS {MARGIN_FLOATS}u",
        f"#define LBM_GROUP_CELLS {GROUP_CELLS}u",
        f"#define LBM_GROUP_FLOATS {GROUP_FLOATS}u",
        f"#define LBM_GROUP_COUNT {GROUP_COUNT}u",
        f"#define LBM_SM_GROUP_BOUNDARIES {boundaries}",
        f"#define LBM_EXPECTED_POPULATION_WORDS {len(expected_words)}u",
        f"#define LBM_GOLD_VELOCITY_WORDS {len(gold_words)}u",
        f"#define LBM_EXPECTED_POPULATION_DIGEST 0x{expected_digest:016x}ULL",
        f"#define LBM_GOLD_VELOCITY_DIGEST 0x{gold_digest:016x}ULL",
    ]
    (GENERATED / "config.h").write_text("\n".join(config) + "\n")

    assembly = [
        '.section .data,"aw",@progbits',
        '.balign 64', '.globl lbm_input_raw', 'lbm_input_raw:',
        '.incbin "generated/input.bin"',
        '.balign 64', '.globl lbm_expected_populations',
        'lbm_expected_populations:',
        '.incbin "generated/expected_populations.bin"',
        '.balign 64', '.globl lbm_gold_velocity', 'lbm_gold_velocity:',
        '.incbin "generated/gold_velocity.bin"',
        '.balign 64', '.globl lbm_guard_before', 'lbm_guard_before:',
        '.zero 64',
        '.balign 64', '.globl lbm_output_raw', 'lbm_output_raw:',
        '.incbin "generated/output_initial.bin"',
        '.balign 64', '.globl lbm_guard_after', 'lbm_guard_after:',
        '.zero 64',
    ]
    (ROOT / "data.S").write_text("\n".join(assembly) + "\n")

    reference_provenance = {
        path: {
            "git_blob": metadata["blob"],
            "sha256": metadata["sha256"],
            "bytes": metadata["bytes"],
        }
        for path, metadata in REFERENCE_FILES.items()
    }
    per_sm = [
        {
            "physical_threadblock_id": owner["physical_threadblock_id"],
            "groups": owner["group_count"],
            "cells": owner["cell_count"],
            "population_writes": owner["population_writes"],
        }
        for owner in audit["owners"]
    ]
    manifest = {
        "name": f"lbm--official-32x32x8-one-step-gather--{sms}sm",
        "family": "lbm",
        "kind": "d3q19-gather-stream-collide",
        "benchmark_contract": "vortex-parboil-lbm-official-default-one-timestep",
        "dataset_contract": "32_32_8_ldc.of-and-32_32_8_ldc.gold",
        "dimensions": [SIZE_X, SIZE_Y, SIZE_Z],
        "padded_dimensions": [PADDED_X, PADDED_Y, PADDED_Z],
        "timesteps": 1,
        "layout": "gather-aos-20-float-cell",
        "active_cells": ACTIVE_CELLS,
        "destination_populations_checked": len(expected_words),
        "velocity_words_checked": len(gold_words),
        "sm_count": sms,
        "loader": loader,
        "gpu_address_hash": loader == "direct-hashed",
        "partition": audit,
        "model": {
            "kind": "complete-ordered-fp32-functional-oracle",
            "bundled_gold_exact_word_match": True,
            "per_sm_work": per_sm,
            "hardware_run": False,
            "timing_model": False,
        },
        "correctness": (
            "all-155648-destination-populations-exact-digest-plus-"
            "all-24576-derived-velocity-words-upstream-tolerance-guards-nan-poison"
        ),
        "upstream_velocity_tolerance": {
            "absolute": "1e-4 * max(abs(gold))",
            "relative": "strictly less than 0.002 * abs(gold)",
        },
        "expected_population_digest": f"{expected_digest:016x}",
        "expected_output_raw_digest": f"{expected_output_digest:016x}",
        "expected_output_raw_words": len(expected_output_words),
        "gold_velocity_digest": f"{gold_digest:016x}",
        "operation_order": "pinned-kernel.cl-scalar-fp32-no-contraction",
        "scalar_only": True,
        "no_vpu": True,
        "vpu": False,
        "vector_isa": False,
        "gemmini": False,
        "mx": False,
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_path": str(reference.resolve()),
        "reference_revision": REFERENCE_REVISION,
        "reference_files": reference_provenance,
        "reference_verified": True,
        "official_dimensions_verified": True,
        "official_default_one_timestep_verified": True,
        "muon_warps_per_core": warps_per_core,
        "robust_end_drain": robust_end_drain,
        "robust_spin_until_reset": robust_spin_until_reset,
        "robust_serial_fence": robust_serial_fence,
        "robust_post_spin": robust_post_spin,
        "kernel_revision": kernel_revision(),
        "kernel_source_sha256": source_digest(),
        "input_sha256": sha256(GENERATED / "input.bin"),
        "output_initial_sha256": sha256(GENERATED / "output_initial.bin"),
        "expected_populations_sha256": sha256(
            GENERATED / "expected_populations.bin"
        ),
        "gold_velocity_sha256": sha256(GENERATED / "gold_velocity.bin"),
        "generator_platform": platform.platform(),
        "status": "prepared",
        "simulator": None,
        "hardware_run": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "partition-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sms", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--loader", choices=("logical", "direct-hashed"),
                        default="logical")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
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
    manifest = prepare(
        out, sms=args.sms, loader=args.loader, reference=args.reference,
        warps_per_core=args.warps_per_core,
        robust_end_drain=args.robust_end_drain,
        robust_spin_until_reset=args.robust_spin_until_reset,
        robust_serial_fence=args.robust_serial_fence,
        robust_post_spin=args.robust_post_spin,
    )
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return

    env = os.environ.copy()
    env["MU_ADDR_HASH"] = "1" if args.loader == "direct-hashed" else "0"
    local_toolchain = (
        ROOT.parents[3] / "toolchains/riscv-tools-prefix/riscv-tools"
    )
    if "RISCV64_TOOLCHAIN_PATH" not in env and local_toolchain.is_dir():
        env["RISCV64_TOOLCHAIN_PATH"] = str(local_toolchain)
    with (out / "build.log").open("w") as log:
        subprocess.run(
            ["make", "kernel.soc.elf"], cwd=ROOT, stdout=log,
            stderr=subprocess.STDOUT, check=True, env=env,
        )
    elf = out / "kernel.soc.elf"
    device_elf = out / "kernel.radiance.elf"
    shutil.copy2(ROOT / "kernel.soc.elf", elf)
    shutil.copy2(ROOT / "kernel.radiance.elf", device_elf)
    manifest.update({
        "status": "built",
        "elf": str(elf),
        "elf_sha256": sha256(elf),
        "bytes": elf.stat().st_size,
        "device_elf": str(device_elf),
        "device_elf_sha256": sha256(device_elf),
    })
    (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
