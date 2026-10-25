#!/usr/bin/env python3
"""Generate and build a faithful scalar Vortex/Rodinia Pathfinder workload."""

from __future__ import annotations

import argparse
from array import array
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
REFERENCE_REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"
REFERENCE_MAIN = "tests/opencl/pathfinder/main.cc"
REFERENCE_KERNEL = "tests/opencl/pathfinder/kernel.cl"
REFERENCE_MAKEFILE = "tests/opencl/pathfinder/Makefile"
REFERENCE_MAIN_GIT_BLOB = "a2b9d676969f75f10bc196544dd5c0a2d5aeba10"
REFERENCE_KERNEL_GIT_BLOB = "d78878978ddd2a6ffb4a9ce0e9275f47b2e086af"
REFERENCE_MAKEFILE_GIT_BLOB = "0e11c09eb6a89be32c7d40a5cbbe0fe8d6b611ce"
REFERENCE_MAIN_SHA256 = "7fe6fb7a663b44f6673b98b05462c8afd061a2ee47739aaa61a5206cc65ff001"
REFERENCE_KERNEL_SHA256 = "c9084e4db3236f60bb90de31fdbdacd21f5bb411442f0bf1e65eda09ace67149"
REFERENCE_MAKEFILE_SHA256 = "4189687963afc7ebaa645240486cc7bbd694406e8d8685642e460404e966b437"

REFERENCE_SEED = 9
UPSTREAM_ROWS = 16
UPSTREAM_COLS = 64
UPSTREAM_PYRAMID_HEIGHT = 2
UPSTREAM_BLOCK_SIZE = 16
MEANINGFUL_ROWS = 256
MEANINGFUL_COLS = 4096
INTS_PER_CACHE_LINE = 16
MUON_CORES = 2
MUON_LANES = 16
PHASE_BARRIER_ID = 12
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv64(words) -> int:
    value = FNV_OFFSET
    for word in words:
        value = ((value ^ (int(word) & MASK32)) * FNV_PRIME) & MASK64
    return value


def write_i32(path: Path, values) -> None:
    packed = array("i", values)
    if packed.itemsize != 4:
        raise RuntimeError("host int is not 32 bits")
    if sys.byteorder != "little":
        packed.byteswap()
    path.write_bytes(packed.tobytes())


def fixture_wall(rows: int, cols: int) -> list[int]:
    """Reproduce the pinned host's srand(9)/rand()%10 stream exactly."""
    libc = ctypes.CDLL(None)
    libc.srand.argtypes = [ctypes.c_uint]
    libc.rand.restype = ctypes.c_int
    libc.srand(REFERENCE_SEED)
    return [libc.rand() % 10 for _ in range(rows * cols)]


def cpu_oracle(wall: list[int], rows: int, cols: int) -> list[int]:
    """Independent full-row DP oracle matching the upstream host reference."""
    previous = list(wall[:cols])
    for row in range(1, rows):
        current = [0] * cols
        base = row * cols
        for column in range(cols):
            up = previous[column]
            left = previous[column - 1] if column > 0 else up
            right = previous[column + 1] if column + 1 < cols else up
            current[column] = wall[base + column] + min(left, up, right)
        previous = current
    return previous


def partition_boundaries(cols: int, sms: int) -> list[int]:
    chunks = cols // INTS_PER_CACHE_LINE
    base, extra = divmod(chunks, sms)
    boundaries = [0]
    cursor = 0
    for sm in range(sms):
        cursor += base + int(sm < extra)
        boundaries.append(cursor * INTS_PER_CACHE_LINE)
    return boundaries


def ownership(cols: int, sms: int) -> tuple[list[int], list[int]]:
    boundaries = partition_boundaries(cols, sms)
    owners = [-1] * cols
    visits = [0] * cols
    for sm, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        for column in range(begin, end):
            owners[column] = sm
            visits[column] += 1
    if visits != [1] * cols:
        raise ValueError("partition does not own every final-row value once")
    for begin in range(0, cols, INTS_PER_CACHE_LINE):
        if len(set(owners[begin:begin + INTS_PER_CACHE_LINE])) != 1:
            raise ValueError("partition splits a final-row cache line")
    return owners, boundaries


def cone_intervals(rows: int, cols: int, sms: int,
                   row: int) -> list[tuple[int, int]]:
    """Per-SM columns evaluated at one semantic row (row 0 is the seed)."""
    if not 0 <= row < rows:
        raise ValueError("row is outside the workload")
    boundaries = partition_boundaries(cols, sms)
    remaining = rows - 1 - row
    return [
        (max(0, begin - remaining), min(cols, end + remaining))
        for begin, end in zip(boundaries, boundaries[1:])
    ]


def interval_accounting(intervals: list[tuple[int, int]]) -> dict:
    """Exact total, union, redundant overlap, and peak multiplicity."""
    if not intervals:
        return {"total_work": 0, "unique_work": 0,
                "redundant_work": 0, "max_multiplicity": 0}
    total = sum(end - begin for begin, end in intervals)
    events = []
    for begin, end in intervals:
        if begin > end:
            raise ValueError("invalid interval")
        events.append((begin, 1))
        events.append((end, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    peak = 0
    unique = 0
    previous = events[0][0]
    for position, delta in events:
        if active > 0:
            unique += position - previous
        active += delta
        peak = max(peak, active)
        previous = position
    return {
        "total_work": total,
        "unique_work": unique,
        "redundant_work": total - unique,
        "max_multiplicity": peak,
    }


def strided_thread_counts(begin: int, end: int, threads: int) -> list[int]:
    """Exact useful iterations assigned by `begin + tid; += threads`."""
    if begin < 0 or end < begin or threads <= 0:
        raise ValueError("invalid strided work interval")
    length = end - begin
    base, extra = divmod(length, threads)
    return [base + int(tid < extra) for tid in range(threads)]


def work_accounting(rows: int, cols: int, sms: int,
                    warps_per_core: int) -> dict:
    """Exact semantic, redundant-halo, and thread schedule accounting."""
    threads = warps_per_core * MUON_CORES * MUON_LANES
    phase_rows = []
    per_sm = [
        {"sm": sm, "seed_loads": 0, "dp_updates": 0,
         "final_writes": 0, "total_useful_iterations": 0}
        for sm in range(sms)
    ]
    minimum = None
    maximum = 0
    for row in range(rows):
        intervals = cone_intervals(rows, cols, sms, row)
        stats = interval_accounting(intervals)
        phase = {
            "row": row, "kind": "seed-load" if row == 0 else "dp-update",
            "remaining_halo": rows - 1 - row,
            "intervals": [
                {"sm": sm, "begin": begin, "end": end,
                 "work": end - begin}
                for sm, (begin, end) in enumerate(intervals)
            ],
            **stats,
        }
        phase_rows.append(phase)
        for sm, (begin, end) in enumerate(intervals):
            count = end - begin
            field = "seed_loads" if row == 0 else "dp_updates"
            per_sm[sm][field] += count
            per_sm[sm]["total_useful_iterations"] += count
            assignments = strided_thread_counts(begin, end, threads)
            local_min = min(assignments)
            local_max = max(assignments)
            minimum = local_min if minimum is None else min(minimum, local_min)
            maximum = max(maximum, local_max)

    boundaries = partition_boundaries(cols, sms)
    for sm, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        count = end - begin
        per_sm[sm]["final_writes"] = count
        per_sm[sm]["total_useful_iterations"] += count
        assignments = strided_thread_counts(begin, end, threads)
        minimum = min(minimum, min(assignments))
        maximum = max(maximum, max(assignments))

    seed = phase_rows[0]
    dp_total = sum(phase["total_work"] for phase in phase_rows[1:])
    dp_unique = sum(phase["unique_work"] for phase in phase_rows[1:])
    dp_redundant = sum(phase["redundant_work"] for phase in phase_rows[1:])
    return {
        "scheduled_threads_per_sm": threads,
        "phase_barrier_id": PHASE_BARRIER_ID,
        "uniform_fence_count_per_sm": rows + 1,
        "uniform_barrier_count_per_sm": rows + 1,
        "minimum_useful_iterations_per_thread_per_phase": minimum,
        "maximum_useful_iterations_per_thread_per_phase": maximum,
        "all_threads_have_useful_work_every_phase": minimum > 0,
        "seed_loads": seed["total_work"],
        "unique_seed_columns": seed["unique_work"],
        "redundant_seed_loads": seed["redundant_work"],
        "dp_updates": dp_total,
        "unique_dp_updates": dp_unique,
        "redundant_dp_updates": dp_redundant,
        "final_writes": cols,
        "unique_final_writes": cols,
        "redundant_final_writes": 0,
        "total_useful_iterations": seed["total_work"] + dp_total + cols,
        "per_sm": per_sm,
        "rows": phase_rows,
    }


def partitioned_model(wall: list[int], rows: int, cols: int,
                      sms: int) -> tuple[list[int], list[int]]:
    """Literal model of the device's independent contracting cones."""
    owners, boundaries = ownership(cols, sms)
    output = [None] * cols
    for sm, (owned_begin, owned_end) in enumerate(
            zip(boundaries, boundaries[1:])):
        previous = [0] * cols
        following = [0] * cols
        halo = rows - 1
        begin = max(0, owned_begin - halo)
        end = min(cols, owned_end + halo)
        previous[begin:end] = wall[begin:end]
        for row in range(1, rows):
            remaining = rows - 1 - row
            begin = max(0, owned_begin - remaining)
            end = min(cols, owned_end + remaining)
            base = row * cols
            for column in range(begin, end):
                up = previous[column]
                left = previous[column - 1] if column > 0 else up
                right = previous[column + 1] if column + 1 < cols else up
                following[column] = wall[base + column] + min(left, up, right)
            previous, following = following, previous
        output[owned_begin:owned_end] = previous[owned_begin:owned_end]
    if any(value is None for value in output):
        raise ValueError("partitioned model left final-row values unwritten")
    return [int(value) for value in output], owners


def validate_parameters(rows: int, cols: int, sms: int, loader: str,
                        warps_per_core: int) -> None:
    if rows < 2 or rows > 1_000_000:
        raise ValueError("rows must be between 2 and 1000000")
    if cols < sms * INTS_PER_CACHE_LINE:
        raise ValueError("cols needs at least one cache line per SM")
    if cols % INTS_PER_CACHE_LINE:
        raise ValueError("cols must be a multiple of 16 int32 values")
    if rows * cols > 16_777_216:
        raise ValueError("workload exceeds the 16M-cell generation limit")
    if rows * 9 > 0x7fffffff:
        raise ValueError("maximum path cost does not fit int32_t")
    if sms not in (1, 2, 4):
        raise ValueError("sms must be 1, 2, or 4")
    if loader not in ("logical", "direct-hashed"):
        raise ValueError("loader must be logical or direct-hashed")
    if warps_per_core not in (1, 2, 4, 8):
        raise ValueError("warps_per_core must be 1, 2, 4, or 8")


def prepare(rows: int, cols: int, out: Path, *, sms: int = 1,
            loader: str = "logical", warps_per_core: int = 8,
            preset: str = "custom", robust_end_drain: bool = False,
            robust_spin_until_reset: bool = False,
            robust_serial_fence: bool = False,
            robust_post_spin: int = 500) -> dict:
    validate_parameters(rows, cols, sms, loader, warps_per_core)
    if robust_spin_until_reset and not robust_end_drain:
        raise ValueError("robust_spin_until_reset requires robust_end_drain")
    if robust_serial_fence and not robust_end_drain:
        raise ValueError("robust_serial_fence requires robust_end_drain")
    if robust_post_spin < 0:
        raise ValueError("robust_post_spin must be nonnegative")
    wall = fixture_wall(rows, cols)
    expected = cpu_oracle(wall, rows, cols)
    modeled, owners = partitioned_model(wall, rows, cols, sms)
    if modeled != expected:
        raise ValueError("partitioned device model disagrees with CPU oracle")

    output_digest = fnv64(expected)
    owner_digest = fnv64(owners)
    accounting = work_accounting(rows, cols, sms, warps_per_core)
    GENERATED.mkdir(exist_ok=True)
    write_i32(GENERATED / "wall.bin", wall)
    write_i32(GENERATED / "expected.bin", expected)
    config = [
        "#pragma once",
        f"#define HPC_SM_COUNT {sms}u",
        f"#define NIGHTLY_CLUSTERS {sms}",
        f"#define NIGHTLY_OCC {warps_per_core}",
        f"#define NIGHTLY_ROBUST_GMEM_END {int(robust_end_drain)}",
        f"#define NIGHTLY_ROBUST_SPIN_UNTIL_RESET {int(robust_spin_until_reset)}",
        f"#define NIGHTLY_ROBUST_SERIAL_FENCE {int(robust_serial_fence)}",
        f"#define NIGHTLY_ROBUST_POST_SPIN {robust_post_spin}u",
        f"#define PATHFINDER_ROWS {rows}u",
        f"#define PATHFINDER_COLS {cols}u",
        f"#define PATHFINDER_INTS_PER_LINE {INTS_PER_CACHE_LINE}u",
        f"#define PATHFINDER_PHASE_BARRIER_ID {PHASE_BARRIER_ID}u",
        f"#define PATHFINDER_OUTPUT_DIGEST 0x{output_digest:016x}ULL",
        f"#define PATHFINDER_OWNER_DIGEST 0x{owner_digest:016x}ULL",
    ]
    (GENERATED / "config.h").write_text("\n".join(config) + "\n")
    assembly = [
        '.section .data,"aw",@progbits',
        '.balign 64', '.globl pathfinder_wall', 'pathfinder_wall:',
        '.incbin "generated/wall.bin"',
        '.balign 64', '.globl pathfinder_expected', 'pathfinder_expected:',
        '.incbin "generated/expected.bin"',
        '.balign 64', '.globl pathfinder_scratch', 'pathfinder_scratch:',
        f'.zero {sms * 2 * cols * 4}',
        '.balign 64', '.globl pathfinder_guard_before',
        'pathfinder_guard_before:',
        *(f'.long 0x{(0xFAEE0000 ^ index):08x}' for index in range(16)),
        '.globl pathfinder_output', 'pathfinder_output:',
        f'.zero {cols * 4}',
        '.balign 64', '.globl pathfinder_guard_middle',
        'pathfinder_guard_middle:',
        *(f'.long 0x{(0xFADD0000 ^ index):08x}' for index in range(16)),
        '.globl pathfinder_owners', 'pathfinder_owners:',
        f'.zero {cols * 4}',
        '.balign 64', '.globl pathfinder_guard_after',
        'pathfinder_guard_after:',
        *(f'.long 0x{(0xFAAA0000 ^ index):08x}' for index in range(16)),
    ]
    (ROOT / "data.S").write_text("\n".join(assembly) + "\n")

    source_files = (
        ROOT.parent / "addr_hash_stamp.mk",
        ROOT.parents[1] / "lib/include/mu_partition.h",
        ROOT.parents[1] / "lib/include/nightly/device.h",
        ROOT.parents[1] / "lib/include/nightly/host.h",
        ROOT / "run.py", ROOT / "Makefile", ROOT / "kernel.cpp",
        ROOT / "host.cpp", ROOT / "emit_symbols.py", ROOT / "model_check.py",
    )
    source_digest = hashlib.sha256()
    for path in source_files:
        source_digest.update(path.relative_to(ROOT.parents[1]).as_posix().encode())
        source_digest.update(path.read_bytes())
    generated_names = ("wall.bin", "expected.bin", "config.h")
    boundaries = partition_boundaries(cols, sms)
    case = {
        "schema": "radiance-pathfinder-build-v2",
        "name": f"pathfinder--{preset}--{rows}x{cols}--parallel--{sms}sm",
        "family": "pathfinder", "kind": "three-neighbor-minimum-cost-dp",
        "preset": preset, "rows": rows, "cols": cols,
        "cells": rows * cols, "sm_count": sms, "loader": loader,
        "gpu_address_hash": loader == "direct-hashed",
        "seed": REFERENCE_SEED, "wall_value_range": [0, 9],
        "output_words": cols, "owner_words": cols,
        "output_digest": f"{output_digest:016x}",
        "owner_digest": f"{owner_digest:016x}",
        "correctness": "complete-exact-final-row-and-owner-check",
        "oracle": "independent-python-full-row-dynamic-programming",
        "generator_model_check": "passed",
        "phase_order": "row-0-seed-then-rows-1-through-N-minus-1",
        "boundary_rule": "missing-left-or-right-neighbor-equals-up",
        "partition": "disjoint-contiguous-16-int-64-byte-final-row-lines",
        "partition_boundaries": boundaries,
        "partition_complete": True, "partition_overlap": 0,
        "partition_cache_line_safe": True, "cache_line_bytes": 64,
        "intermediate_strategy": "private-per-SM-contracting-dependency-cone",
        "thread_schedule": "all-scheduled-threads-stride-every-local-phase",
        "phase_synchronization": "uniform-all-warp-fence-then-local-barrier",
        "work_accounting": accounting,
        "scalar_only": True, "no_vpu": True, "vpu": False,
        "vector_isa": False, "gemmini": False, "mx": False,
        "muon_warps_per_core": warps_per_core,
        "robust_end_drain": robust_end_drain,
        "robust_spin_until_reset": robust_spin_until_reset,
        "robust_serial_fence": robust_serial_fence,
        "robust_post_spin": robust_post_spin,
        "reference_repository": "vortexgpgpu/vortex",
        "reference_revision": REFERENCE_REVISION,
        "reference_main": REFERENCE_MAIN,
        "reference_main_sha256": REFERENCE_MAIN_SHA256,
        "reference_main_git_blob": REFERENCE_MAIN_GIT_BLOB,
        "reference_kernel": REFERENCE_KERNEL,
        "reference_kernel_sha256": REFERENCE_KERNEL_SHA256,
        "reference_kernel_git_blob": REFERENCE_KERNEL_GIT_BLOB,
        "reference_makefile": REFERENCE_MAKEFILE,
        "reference_makefile_sha256": REFERENCE_MAKEFILE_SHA256,
        "reference_makefile_git_blob": REFERENCE_MAKEFILE_GIT_BLOB,
        "reference_seed": REFERENCE_SEED,
        "upstream_default_contract": {
            "rows": UPSTREAM_ROWS, "cols": UPSTREAM_COLS,
            "pyramid_height": UPSTREAM_PYRAMID_HEIGHT,
            "block_size": UPSTREAM_BLOCK_SIZE,
        },
        "is_upstream_sized_canary": rows == UPSTREAM_ROWS and cols == UPSTREAM_COLS,
        "meaningful_workload_contract": {
            "rows": MEANINGFUL_ROWS, "cols": MEANINGFUL_COLS,
            "cells": MEANINGFUL_ROWS * MEANINGFUL_COLS,
        },
        "is_meaningful_workload": rows == MEANINGFUL_ROWS and cols == MEANINGFUL_COLS,
        "benchmark_contract": "vortex-rodinia-pathfinder-final-row",
        "kernel_source_sha256": source_digest.hexdigest(),
        "generated_sha256": {
            name: sha256(GENERATED / name) for name in generated_names
        },
        "status": "prepared", "simulator": None,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    return case


def build(case: dict, out: Path) -> dict:
    env = os.environ.copy()
    env["MU_ADDR_HASH"] = "1" if case["loader"] == "direct-hashed" else "0"
    private_host_toolchain = (
        ROOT.parents[3] / "toolchains/riscv-tools-prefix/riscv-tools"
    )
    if "RISCV64_TOOLCHAIN_PATH" not in env and private_host_toolchain.is_dir():
        env["RISCV64_TOOLCHAIN_PATH"] = str(private_host_toolchain)
    with (out / "build.log").open("w") as log:
        subprocess.run(["make", "clean"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, env=env)
        subprocess.run(["make", "kernel.soc.elf"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, env=env)
    for name in ("kernel.soc.elf", "kernel.radiance.elf"):
        shutil.copy2(ROOT / name, out / name)
    elf = out / "kernel.soc.elf"
    device = out / "kernel.radiance.elf"
    case.update({
        "status": "built", "elf": str(elf.resolve()),
        "elf_sha256": sha256(elf), "bytes": elf.stat().st_size,
        "device_elf": str(device.resolve()),
        "device_elf_sha256": sha256(device),
    })
    (out / "result.json").write_text(json.dumps(case, indent=2) + "\n")
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("upstream-canary", "meaningful", "custom"),
                        default="upstream-canary")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
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
    presets = {
        "upstream-canary": (UPSTREAM_ROWS, UPSTREAM_COLS),
        "meaningful": (MEANINGFUL_ROWS, MEANINGFUL_COLS),
    }
    if args.preset == "custom":
        if args.rows is None or args.cols is None:
            parser.error("custom preset requires --rows and --cols")
        rows, cols = args.rows, args.cols
    else:
        if args.rows is not None or args.cols is not None:
            parser.error("--rows/--cols are valid only with --preset custom")
        rows, cols = presets[args.preset]
    out = args.out.resolve()
    case = prepare(rows, cols, out, sms=args.sms, loader=args.loader,
                   warps_per_core=args.warps_per_core, preset=args.preset,
                   robust_end_drain=args.robust_end_drain,
                   robust_spin_until_reset=args.robust_spin_until_reset,
                   robust_serial_fence=args.robust_serial_fence,
                   robust_post_spin=args.robust_post_spin)
    if not args.prepare_only:
        case = build(case, out)
        print(json.dumps(case, indent=2))


if __name__ == "__main__":
    main()
