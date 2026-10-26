#!/usr/bin/env python3
"""Generate, audit, and build the scalar pinned Vortex/Parboil JDS SpMV."""

from __future__ import annotations

from array import array
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
REFERENCE_REPOSITORY = "vortexgpgpu/vortex"
REFERENCE_REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"
REFERENCE_KERNEL = "tests/opencl/spmv/kernel.cl"
REFERENCE_KERNEL_GIT_BLOB = "3f44205e892b6523a010a04cbb1fd5cad995a02f"
REFERENCE_KERNEL_SHA256 = "8902bc12280ff7474fa6ea2c5dcb421006e80a7641badeefb2095942d3430674"
REFERENCE_MAIN = "tests/opencl/spmv/main.cc"
REFERENCE_MAIN_SHA256 = "f3ed7dc317c2bca7fe238ff2c665c1263fa13ade8598675b911df6ca40037f49"
REFERENCE_CONVERTER = "tests/opencl/spmv/convert_dataset.c"
REFERENCE_CONVERTER_SHA256 = "99b58152f88e4ace7a84fbfcb5882384bd58d7fce235ec36c4495c6f892b253b"
JDS_WARP_SIZE = 32
OUTPUT_LINE_FLOATS = 16
ABS_TOL_SCALE = 1.0e-4
REL_TOL = 0.002
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def digest(words: Iterable[int]) -> int:
    value = FNV_OFFSET
    for word in words:
        value = ((value ^ int(word)) * FNV_PRIME) & MASK64
    return value


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


@dataclass
class Dataset:
    name: str
    rows: list[list[tuple[int, float]]]
    vector: list[float]
    stored_nnz: int
    source_kind: str
    source_files: dict[str, str]
    upstream_gold: list[float] | None = None


@dataclass
class Jds:
    dimension: int
    data: list[float]
    indices: list[int]
    row_to_jds: list[int]
    pointers: list[int]
    group_bounds: list[int]
    row_nnz: list[int]
    padded_rows: int


def fixture_dataset(rows: int, seed: int) -> Dataset:
    if rows < 64 or rows > 1_000_000:
        raise ValueError("fixture rows must be between 64 and 1000000")
    expanded: list[list[tuple[int, float]]] = [[] for _ in range(rows)]
    stored = 0
    for row in range(rows):
        diagonal = f32(2.0 + ((row * 17 + seed) % 11) / 8.0)
        expanded[row].append((row, diagonal))
        stored += 1

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for row in range(rows):
        offsets = [1]
        if row % 2 == 0:
            offsets.append(3)
        if row % 3 == 0:
            offsets.append(7)
        if row % 5 == 0:
            offsets.append(13)
        if row % 11 == 0:
            offsets.append(29)
        for offset in offsets:
            col = (row + offset) % rows
            pair = (min(row, col), max(row, col))
            if pair[0] != pair[1] and pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    for ordinal, (row, col) in enumerate(pairs):
        numerator = ((row * 31 + col * 17 + ordinal * 7 + seed) % 23) - 11
        if numerator == 0:
            numerator = 1
        value = f32(numerator / 16.0)
        expanded[row].append((col, value))
        expanded[col].append((row, value))
        stored += 1

    vector = [f32((((index * 13 + seed) % 29) + 1) / 17.0)
              for index in range(rows)]
    return Dataset(
        name=f"fixture-{rows}-seed-{seed}", rows=expanded, vector=vector,
        stored_nnz=stored, source_kind="generated-deterministic-symmetric",
        source_files={},
    )


def read_matrix_market(path: Path) -> tuple[list[list[tuple[int, float]]], int]:
    with path.open("r", encoding="ascii") as stream:
        banner = stream.readline().strip().lower().split()
        if banner != ["%%matrixmarket", "matrix", "coordinate", "real", "symmetric"]:
            raise ValueError("expected a real symmetric coordinate MatrixMarket file")
        line = stream.readline()
        while line.startswith("%"):
            line = stream.readline()
        dimensions = line.split()
        if len(dimensions) != 3:
            raise ValueError("invalid MatrixMarket dimensions")
        row_count, col_count, declared = map(int, dimensions)
        if row_count != col_count or row_count <= 0:
            raise ValueError("SpMV input must be a nonempty square matrix")
        rows: list[list[tuple[int, float]]] = [[] for _ in range(row_count)]
        stored = 0
        for line in stream:
            if not line.strip() or line.startswith("%"):
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"invalid MatrixMarket entry: {line.rstrip()}")
            row = int(fields[0]) - 1
            col = int(fields[1]) - 1
            if not (0 <= row < row_count and 0 <= col < col_count):
                raise ValueError("MatrixMarket index outside declared dimensions")
            value = f32(float(fields[2]))
            rows[row].append((col, value))
            if row != col:
                rows[col].append((row, value))
            stored += 1
        if stored != declared:
            raise ValueError(f"declared {declared} entries but read {stored}")
    return rows, stored


def read_raw_f32(path: Path, count: int) -> list[float]:
    data = path.read_bytes()
    if len(data) != count * 4:
        raise ValueError(f"{path} has {len(data)} bytes; expected {count * 4}")
    return list(struct.unpack(f"<{count}f", data))


def read_gold(path: Path, count: int) -> list[float]:
    data = path.read_bytes()
    if len(data) < 4:
        raise ValueError(f"{path} is too short")
    (declared,) = struct.unpack_from("<I", data)
    if declared != count or len(data) != 4 + count * 4:
        raise ValueError(
            f"{path} declares {declared} floats and has {len(data)} bytes; "
            f"expected {count} floats")
    return list(struct.unpack_from(f"<{count}f", data, 4))


def imported_dataset(matrix: Path, vector: Path, gold: Path,
                     name: str) -> Dataset:
    rows, stored = read_matrix_market(matrix)
    dimension = len(rows)
    return Dataset(
        name=name, rows=rows, vector=read_raw_f32(vector, dimension),
        stored_nnz=stored, source_kind="pinned-matrix-market-symmetric",
        source_files={
            "matrix": str(matrix.resolve()), "matrix_sha256": sha256(matrix),
            "vector": str(vector.resolve()), "vector_sha256": sha256(vector),
            "gold": str(gold.resolve()), "gold_sha256": sha256(gold),
        },
        upstream_gold=read_gold(gold, dimension),
    )


def build_jds(rows: list[list[tuple[int, float]]]) -> Jds:
    dimension = len(rows)
    if dimension == 0:
        raise ValueError("matrix cannot be empty")
    for entries in rows:
        for column, value in entries:
            if not 0 <= column < dimension:
                raise ValueError("column outside matrix")
            if not (-3.4028235e38 <= value <= 3.4028235e38):
                raise ValueError("matrix contains a non-finite value")

    # Python's stable sort gives the deterministic form of the two upstream
    # row-only qsorts: descending nnz, then original row order for ties.
    permutation = sorted(range(dimension), key=lambda row: -len(rows[row]))
    padded_rows = ((dimension + JDS_WARP_SIZE - 1) // JDS_WARP_SIZE) * JDS_WARP_SIZE
    permutation.extend([-1] * (padded_rows - dimension))
    row_to_jds = [0] * dimension
    for jds_row, original in enumerate(permutation):
        if original >= 0:
            row_to_jds[original] = jds_row

    group_bounds = []
    for begin in range(0, padded_rows, JDS_WARP_SIZE):
        original = permutation[begin]
        group_bounds.append(len(rows[original]) if original >= 0 else 0)
    depth = group_bounds[0]
    data: list[float] = []
    indices: list[int] = []
    pointers: list[int] = []
    for k in range(depth):
        pointers.append(len(data))
        for group, bound in enumerate(group_bounds):
            if bound <= k:
                break
            begin = group * JDS_WARP_SIZE
            for jds_row in range(begin, begin + JDS_WARP_SIZE):
                original = permutation[jds_row]
                if original >= 0 and k < len(rows[original]):
                    column, value = rows[original][k]
                    data.append(f32(value))
                    indices.append(column)
                else:
                    # Pinned convert_dataset.c writes index=-1 for this zero.
                    # Index 0 removes its undefined x[-1] access while retaining
                    # the exact zero term and group-bound loop iteration.
                    data.append(f32(0.0))
                    indices.append(0)

    jds = Jds(
        dimension=dimension, data=data, indices=indices,
        row_to_jds=row_to_jds, pointers=pointers,
        group_bounds=group_bounds, row_nnz=[len(row) for row in rows],
        padded_rows=padded_rows,
    )
    validate_jds(jds)
    return jds


def validate_jds(jds: Jds) -> None:
    if sorted(jds.row_to_jds) != list(range(jds.dimension)):
        # The JDS positions include padded rows, so compare only real positions.
        positions = sorted(jds.row_to_jds)
        if len(positions) != jds.dimension or len(set(positions)) != jds.dimension:
            raise ValueError("row-to-JDS mapping is not one-to-one")
    if len(jds.data) != len(jds.indices):
        raise ValueError("JDS data/index length mismatch")
    for row, jds_row in enumerate(jds.row_to_jds):
        bound = jds.group_bounds[jds_row // JDS_WARP_SIZE]
        if bound < jds.row_nnz[row]:
            raise ValueError("JDS group bound omits a real matrix entry")
        for k in range(bound):
            address = jds.pointers[k] + jds_row
            if not 0 <= address < len(jds.data):
                raise ValueError("JDS pointer arithmetic is outside data")


def reference_row(jds: Jds, vector: list[float], row: int) -> float:
    jds_row = jds.row_to_jds[row]
    bound = jds.group_bounds[jds_row // JDS_WARP_SIZE]
    total = f32(0.0)
    for k in range(bound):
        address = jds.pointers[k] + jds_row
        product = f32(jds.data[address] * vector[jds.indices[address]])
        total = f32(total + product)
    return total


def reference_output(jds: Jds, vector: list[float]) -> list[float]:
    if len(vector) != jds.dimension:
        raise ValueError("vector length does not match matrix")
    return [reference_row(jds, vector, row) for row in range(jds.dimension)]


def partitioned_reference_output(jds: Jds, vector: list[float],
                                 boundaries: list[int]) -> tuple[list[float], list[int]]:
    output = [f32(float("nan"))] * jds.dimension
    visits = [0] * jds.dimension
    for sm in range(len(boundaries) - 1):
        for row in range(boundaries[sm], boundaries[sm + 1]):
            output[row] = reference_row(jds, vector, row)
            visits[row] += 1
    return output, visits


def balanced_boundaries(row_nnz: list[int], sms: int,
                        granule: int = OUTPUT_LINE_FLOATS) -> list[int]:
    if sms not in (1, 2, 4):
        raise ValueError("sms must be 1, 2, or 4")
    dimension = len(row_nnz)
    chunks = (dimension + granule - 1) // granule
    if chunks < sms:
        raise ValueError("matrix needs at least one output cache line per SM")
    prefix = [0]
    for value in row_nnz:
        prefix.append(prefix[-1] + value)
    total = prefix[-1]
    boundaries = [0]
    previous_chunk = 0
    for part in range(1, sms):
        minimum = previous_chunk + 1
        maximum = chunks - (sms - part)
        target = total * part / sms
        chunk = min(
            range(minimum, maximum + 1),
            key=lambda candidate: (abs(prefix[min(dimension, candidate * granule)] - target),
                                   candidate),
        )
        boundaries.append(min(dimension, chunk * granule))
        previous_chunk = chunk
    boundaries.append(dimension)
    return boundaries


def partition_work(row_nnz: list[int], boundaries: list[int]) -> list[int]:
    return [sum(row_nnz[boundaries[i]:boundaries[i + 1]])
            for i in range(len(boundaries) - 1)]


def block_cyclic_ranges(dimension: int, sms: int,
                        chunk_rows: int) -> list[list[tuple[int, int]]]:
    chunks = (dimension + chunk_rows - 1) // chunk_rows
    return [[(chunk * chunk_rows, min(dimension, (chunk + 1) * chunk_rows))
             for chunk in range(sm, chunks, sms)]
            for sm in range(sms)]


def ranged_reference_output(jds: Jds, vector: list[float],
                            ownership: list[list[tuple[int, int]]]
                            ) -> tuple[list[float], list[int]]:
    output = [f32(float("nan"))] * jds.dimension
    visits = [0] * jds.dimension
    for ranges in ownership:
        for begin, end in ranges:
            for row in range(begin, end):
                output[row] = reference_row(jds, vector, row)
                visits[row] += 1
    return output, visits


def ranged_work(costs: list[int],
                ownership: list[list[tuple[int, int]]]) -> list[int]:
    return [sum(sum(costs[begin:end]) for begin, end in ranges)
            for ranges in ownership]


def validate_boundaries(boundaries: list[int], dimension: int, sms: int) -> None:
    if len(boundaries) != sms + 1 or boundaries[0] != 0 or boundaries[-1] != dimension:
        raise ValueError("partition does not cover the output domain")
    if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("partition contains an empty or overlapping range")
    if any(value % OUTPUT_LINE_FLOATS for value in boundaries[1:-1]):
        raise ValueError("internal partition boundary splits a 64-byte output line")


def oracle_tolerance(expected: list[float]) -> tuple[float, float]:
    return f32(ABS_TOL_SCALE * max(abs(value) for value in expected)), f32(REL_TOL)


def compare_upstream(expected: list[float], gold: list[float]) -> dict:
    if len(expected) != len(gold):
        raise ValueError("gold length mismatch")
    abs_tol = f32(ABS_TOL_SCALE * max(abs(value) for value in gold))
    bad = 0
    max_diff = 0.0
    for actual, reference in zip(expected, gold):
        diff = abs(f32(reference - actual))
        max_diff = max(max_diff, diff)
        if not (diff <= abs_tol or diff < REL_TOL * abs(reference)):
            bad += 1
    if bad:
        raise ValueError(f"generated ordered JDS oracle disagrees with upstream gold at {bad} rows")
    return {"checked": len(gold), "bad": bad, "max_abs_diff": max_diff,
            "absolute_tolerance": abs_tol, "relative_tolerance": REL_TOL}


def write_array(path: Path, typecode: str, values: Iterable[int | float]) -> None:
    data = array(typecode, values)
    if data.itemsize != 4:
        raise RuntimeError(f"array type {typecode} is not 32-bit")
    if sys.byteorder != "little":
        data.byteswap()
    path.write_bytes(data.tobytes())


def source_digest() -> str:
    value = hashlib.sha256()
    paths = (
        ROOT.parent / "addr_hash_stamp.mk",
        ROOT.parents[1] / "lib/include/nightly/device.h",
        ROOT.parents[1] / "lib/include/nightly/host.h",
        ROOT / "Makefile", ROOT / "run.py", ROOT / "kernel.cpp",
        ROOT / "host.cpp", ROOT / "emit_symbols.py", ROOT / "test_run.py",
        ROOT / "README.md",
    )
    for path in paths:
        value.update(path.relative_to(ROOT.parents[1]).as_posix().encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def prepare(dataset: Dataset, out: Path, *, sms: int = 1,
            loader: str = "logical", warps_per_core: int = 8,
            robust_end_drain: bool = False,
            robust_spin_until_reset: bool = False,
            robust_serial_fence: bool = False,
            robust_post_spin: int = 500,
            block_cyclic_rows: int = 0) -> dict:
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
    threads_per_block = warps_per_core * 2 * 16
    if (block_cyclic_rows < 0 or
            (block_cyclic_rows and
             (block_cyclic_rows % OUTPUT_LINE_FLOATS or
              block_cyclic_rows < threads_per_block))):
        raise ValueError(
            "block-cyclic rows must be zero or a cache-line multiple at least "
            "as large as the thread block")

    jds = build_jds(dataset.rows)
    expected = reference_output(jds, dataset.vector)
    expected_words = [float_word(value) for value in expected]
    expected_digest = digest(expected_words)
    boundaries = balanced_boundaries(jds.row_nnz, sms)
    validate_boundaries(boundaries, jds.dimension, sms)
    ownership = None
    loop_costs = [jds.group_bounds[row // JDS_WARP_SIZE]
                  for row in jds.row_to_jds]
    if block_cyclic_rows:
        ownership = block_cyclic_ranges(jds.dimension, sms,
                                        block_cyclic_rows)
        work = ranged_work(jds.row_nnz, ownership)
        loop_work = ranged_work(loop_costs, ownership)
        partitioned, visits = ranged_reference_output(
            jds, dataset.vector, ownership)
    else:
        work = partition_work(jds.row_nnz, boundaries)
        loop_work = partition_work(loop_costs, boundaries)
        partitioned, visits = partitioned_reference_output(
            jds, dataset.vector, boundaries)
    if sum(work) != sum(jds.row_nnz):
        raise ValueError("partition does not execute every real nonzero exactly once")
    if visits != [1] * jds.dimension:
        raise ValueError("partition does not execute every output exactly once")
    if [float_word(value) for value in partitioned] != expected_words:
        raise ValueError("partitioned model changes the ordered JDS result")
    upstream_validation = None
    if dataset.upstream_gold is not None:
        upstream_validation = compare_upstream(expected, dataset.upstream_gold)

    abs_tol, rel_tol = oracle_tolerance(expected)
    GENERATED.mkdir(exist_ok=True)
    write_array(GENERATED / "data.bin", "f", jds.data)
    write_array(GENERATED / "indices.bin", "I", jds.indices)
    write_array(GENERATED / "row_to_jds.bin", "I", jds.row_to_jds)
    write_array(GENERATED / "x.bin", "f", dataset.vector)
    write_array(GENERATED / "jds_ptr.bin", "I", jds.pointers)
    write_array(GENERATED / "group_bounds.bin", "I", jds.group_bounds)
    write_array(GENERATED / "expected.bin", "f", expected)

    total_bytes = (len(jds.data) * 8 + jds.dimension * 16 +
                   len(jds.pointers) * 4 + len(jds.group_bounds) * 4 + 256)
    if total_bytes + 0x200000 > 0x70000000:
        raise ValueError("JDS SpMV arrays exceed the GPU DRAM address window")

    boundary_initializer = "{" + ",".join(f"{value}u" for value in boundaries) + "}"
    config = [
        "#pragma once",
        f"#define HPC_SM_COUNT {sms}u",
        f"#define NIGHTLY_CLUSTERS {sms}",
        f"#define NIGHTLY_OCC {warps_per_core}",
        f"#define NIGHTLY_ROBUST_GMEM_END {int(robust_end_drain)}",
        f"#define NIGHTLY_ROBUST_SPIN_UNTIL_RESET {int(robust_spin_until_reset)}",
        f"#define NIGHTLY_ROBUST_SERIAL_FENCE {int(robust_serial_fence)}",
        f"#define NIGHTLY_ROBUST_POST_SPIN {robust_post_spin}u",
        f"#define SPMV_DIMENSION {jds.dimension}u",
        f"#define SPMV_JDS_WARP_SIZE {JDS_WARP_SIZE}u",
        f"#define SPMV_JDS_DEPTH {len(jds.pointers)}u",
        f"#define SPMV_BLOCK_CYCLIC_ROWS {block_cyclic_rows}u",
        f"#define SPMV_SM_BOUNDARIES {boundary_initializer}",
        f"#define SPMV_ABS_TOL_BITS 0x{float_word(abs_tol):08x}u",
        f"#define SPMV_REL_TOL_BITS 0x{float_word(rel_tol):08x}u",
        f"#define SPMV_EXPECTED_DIGEST 0x{expected_digest:016x}ULL",
    ]
    (GENERATED / "config.h").write_text("\n".join(config) + "\n")
    asm = [
        '.section .data,"aw",@progbits',
        '.balign 64', '.globl spmv_jds_data', 'spmv_jds_data:',
        '.incbin "generated/data.bin"',
        '.balign 64', '.globl spmv_jds_indices', 'spmv_jds_indices:',
        '.incbin "generated/indices.bin"',
        '.balign 64', '.globl spmv_jds_row_to_jds', 'spmv_jds_row_to_jds:',
        '.incbin "generated/row_to_jds.bin"',
        '.balign 64', '.globl spmv_jds_x', 'spmv_jds_x:',
        '.incbin "generated/x.bin"',
        '.balign 64', '.globl spmv_jds_ptr', 'spmv_jds_ptr:',
        '.incbin "generated/jds_ptr.bin"',
        '.balign 64', '.globl spmv_jds_group_bounds', 'spmv_jds_group_bounds:',
        '.incbin "generated/group_bounds.bin"',
        '.balign 64', '.globl spmv_jds_expected', 'spmv_jds_expected:',
        '.incbin "generated/expected.bin"',
        '.section .bss,"aw",@nobits', '.balign 64',
        '.globl spmv_jds_guard_before', 'spmv_jds_guard_before:', '.zero 64',
        '.globl spmv_jds_output', 'spmv_jds_output:', f'.zero {jds.dimension * 4}',
        '.balign 64',
        '.globl spmv_jds_guard_after', 'spmv_jds_guard_after:', '.zero 64',
    ]
    (ROOT / "data.S").write_text("\n".join(asm) + "\n")

    generated_hashes = {
        path.name: sha256(path) for path in sorted(GENERATED.glob("*.bin"))
    }
    schedule = (f"-rowblockcyclic{block_cyclic_rows}"
                if block_cyclic_rows else "")
    manifest = {
        "name": f"spmv-jds{schedule}--{dataset.name}--{sms}sm",
        "family": "spmv_jds", "kind": "sparse-matrix-vector-product",
        "sm_count": sms, "loader": loader,
        "gpu_address_hash": loader == "direct-hashed",
        "dimension": jds.dimension, "stored_nnz": dataset.stored_nnz,
        "expanded_nnz": sum(jds.row_nnz),
        "jds_padded_entries": len(jds.data),
        "jds_padding_entries": len(jds.data) - sum(jds.row_nnz),
        "jds_depth": len(jds.pointers), "jds_warp_size": JDS_WARP_SIZE,
        "jds_padded_rows": jds.padded_rows,
        "output_element_count": jds.dimension,
        "element_format": "float32",
        "expected_digest": f"{expected_digest:016x}",
        "absolute_tolerance": abs_tol, "relative_tolerance": rel_tol,
        "correctness": "complete-output-upstream-tolerance-exact-digest-and-guards",
        "oracle": "ordered-float32-jds-multiply-then-add-every-output",
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_revision": REFERENCE_REVISION,
        "reference_kernel": REFERENCE_KERNEL,
        "reference_kernel_git_blob": REFERENCE_KERNEL_GIT_BLOB,
        "reference_kernel_sha256": REFERENCE_KERNEL_SHA256,
        "reference_main": REFERENCE_MAIN,
        "reference_main_sha256": REFERENCE_MAIN_SHA256,
        "reference_converter": REFERENCE_CONVERTER,
        "reference_converter_sha256": REFERENCE_CONVERTER_SHA256,
        "benchmark_contract": "pinned-vortex-parboil-spmv-jds-naive",
        "reference_contract": "vortex-parboil-jds-equation-and-upstream-gold-tolerance",
        "equation": "sum += data[jds_ptr[k]+ix] * x[index[jds_ptr[k]+ix]]; y[perm[ix]]=sum",
        "accumulation_order": "strict-increasing-k-no-contraction",
        "scalar_only": True, "no_vpu": True, "vpu": False, "vector_isa": False,
        "gemmini": False, "mx": False,
        "partition": (
            "disjoint-block-cyclic-cache-line-complete-output-row-chunks"
            if block_cyclic_rows else
            "nonzero-balanced-contiguous-16-float-output-line-aligned"),
        "partition_boundaries": boundaries if not block_cyclic_rows else [],
        "partition_ownership": ownership,
        "partition_block_cyclic_rows": block_cyclic_rows,
        "partition_nnz": work,
        "partition_loop_iterations": loop_work,
        "partition_complete": True, "partition_overlap": False,
        "partition_cache_line_safe": True,
        "padding_index_policy": "upstream-minus-one-normalized-to-zero-for-zero-data",
        "dataset": dataset.name, "dataset_source": dataset.source_kind,
        "dataset_files": dataset.source_files,
        "upstream_gold_validation": upstream_validation,
        "generator_model_check": "passed",
        "generator_platform": platform.platform(),
        "muon_warps_per_core": warps_per_core,
        "robust_end_drain": robust_end_drain,
        "robust_spin_until_reset": robust_spin_until_reset,
        "robust_serial_fence": robust_serial_fence,
        "robust_post_spin": robust_post_spin,
        "logical_payload_bytes": sum(jds.row_nnz) * 8 + jds.dimension * 8,
        "resident_generated_bytes": total_bytes,
        "kernel_revision": git_revision(ROOT),
        "kernel_source_sha256": source_digest(),
        "generated_sha256": generated_hashes,
        "status": "prepared", "simulator": None,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-rows", type=int, default=257)
    parser.add_argument("--fixture-seed", type=int, default=7)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--vector", type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--dataset-name", default="imported")
    parser.add_argument("--sms", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--loader", choices=("logical", "direct-hashed"),
                        default="logical")
    parser.add_argument("--warps-per-core", type=int, choices=(1, 2, 4, 8),
                        default=8)
    parser.add_argument("--robust-end-drain", action="store_true")
    parser.add_argument("--robust-spin-until-reset", action="store_true")
    parser.add_argument("--robust-serial-fence", action="store_true")
    parser.add_argument("--robust-post-spin", type=int, default=500)
    parser.add_argument("--block-cyclic-rows", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    supplied = [args.matrix is not None, args.vector is not None, args.gold is not None]
    if any(supplied) and not all(supplied):
        parser.error("--matrix, --vector, and --gold must be supplied together")
    if args.matrix is not None:
        dataset = imported_dataset(args.matrix, args.vector, args.gold,
                                   args.dataset_name)
    else:
        dataset = fixture_dataset(args.fixture_rows, args.fixture_seed)
    out = args.out.resolve()
    manifest = prepare(
        dataset, out, sms=args.sms, loader=args.loader,
        warps_per_core=args.warps_per_core,
        robust_end_drain=args.robust_end_drain,
        robust_spin_until_reset=args.robust_spin_until_reset,
        robust_serial_fence=args.robust_serial_fence,
        robust_post_spin=args.robust_post_spin,
        block_cyclic_rows=args.block_cyclic_rows,
    )
    if args.prepare_only:
        return
    env = os.environ.copy()
    env["MU_ADDR_HASH"] = "1" if args.loader == "direct-hashed" else "0"
    with (out / "build.log").open("w") as log:
        subprocess.run(["make", "kernel.soc.elf"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, env=env)
    for name in ("kernel.soc.elf", "kernel.radiance.elf"):
        shutil.copy2(ROOT / name, out / name)
    elf = out / "kernel.soc.elf"
    manifest.update({
        "status": "built", "elf": str(elf),
        "elf_sha256": sha256(elf), "bytes": elf.stat().st_size,
        "device_elf": str(out / "kernel.radiance.elf"),
        "device_elf_sha256": sha256(out / "kernel.radiance.elf"),
    })
    (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
