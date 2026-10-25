#!/usr/bin/env python3
"""Generate and build the pinned Vortex/Rodinia B+Tree correctness canary."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import ctypes
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
REFERENCE_REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"
REFERENCE_MAIN = "tests/opencl/btree/main.cc"
REFERENCE_KERNEL = "tests/opencl/btree/kernel.cl"
REFERENCE_MAKEFILE = "tests/opencl/btree/Makefile"
REFERENCE_MAIN_GIT_BLOB = "a4a2d1e940691119720edb7115dd81cc17203230"
REFERENCE_KERNEL_GIT_BLOB = "25b613fd2a0f55b59da6ccf06662a09752e12adf"
REFERENCE_MAKEFILE_GIT_BLOB = "b743ffc8207e27114d80ba28e3fbf0de6eb00113"
REFERENCE_MAIN_SHA256 = "b0de0a5a8ee00fe7d8612cfc38652314b8a19aa376f3a34e34d37d5cd538df52"
REFERENCE_KERNEL_SHA256 = "24adee3ea370fe666a42a04db1bd6becab8b59102324d7ac6527a6dd6738b3a0"
REFERENCE_MAKEFILE_SHA256 = "960ffd255f1e8a757214be9aeaf603e5e42ce0d3d8ded214a9d9514877a8e98b"

ORDER = 16
LEAF_CAPACITY = ORDER - 1
PAYLOAD_BASE = 100_000
REFERENCE_SEED = 23
DEFAULT_KEYS = 256
DEFAULT_POINT_QUERIES = 32
DEFAULT_RANGE_QUERIES = 16
DEFAULT_RANGE_SPAN = 8
OUTPUT_RECORD_BYTES = 16
CACHE_LINE_BYTES = 64
OUTPUT_RECORDS_PER_LINE = CACHE_LINE_BYTES // OUTPUT_RECORD_BYTES
POINT_TAG = 0xB7010000
RANGE_TAG = 0xB7020000
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
NODE_STRUCT = struct.Struct("<i17i17iII")


@dataclass
class TmpNode:
    is_leaf: bool
    min_key: int
    keys: list[int] = field(default_factory=list)
    children: list[int] = field(default_factory=list)
    recids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class WireNode:
    location: int
    indices: tuple[int, ...]
    keys: tuple[int, ...]
    is_leaf: int
    num_keys: int

    def pack(self) -> bytes:
        if len(self.indices) != ORDER + 1 or len(self.keys) != ORDER + 1:
            raise ValueError("wire node arrays must have ORDER+1 entries")
        return NODE_STRUCT.pack(
            self.location, *self.indices, *self.keys,
            self.is_leaf, self.num_keys,
        )


def sha256(path: Path) -> str:
    digest_value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest_value.update(block)
    return digest_value.hexdigest()


def fnv64(words) -> int:
    value = FNV_OFFSET
    for word in words:
        value = ((value ^ (int(word) & MASK32)) * FNV_PRIME) & MASK64
    return value


def write_i32(path: Path, values) -> None:
    values = list(values)
    path.write_bytes(struct.pack(f"<{len(values)}i", *values))


def fixture_keys(count: int) -> tuple[list[int], list[int]]:
    keys = [2 * index for index in range(count)]
    records = [key + PAYLOAD_BASE for key in keys]
    return keys, records


def fixture_queries(key_count: int, point_count: int, range_count: int,
                    range_span: int) -> tuple[list[int], list[tuple[int, int]]]:
    """Reproduce the pinned host's single srand(23)/rand() stream exactly."""
    libc = ctypes.CDLL(None)
    libc.srand.argtypes = [ctypes.c_uint]
    libc.rand.restype = ctypes.c_int
    libc.srand(REFERENCE_SEED)
    points = [libc.rand() % (2 * key_count) for _ in range(point_count)]
    ranges = []
    for _ in range(range_count):
        start_rank = libc.rand() % key_count
        end_rank = min(key_count - 1,
                       start_rank + (libc.rand() % range_span))
        ranges.append((2 * start_rank, 2 * end_rank))
    return points, ranges


def build_tree(keys: list[int]) -> tuple[list[TmpNode], int]:
    if not keys or any(lhs >= rhs for lhs, rhs in zip(keys, keys[1:])):
        raise ValueError("keys must be nonempty, sorted, and unique")
    pool: list[TmpNode] = []
    level: list[int] = []
    for begin in range(0, len(keys), LEAF_CAPACITY):
        leaf_keys = keys[begin:begin + LEAF_CAPACITY]
        pool.append(TmpNode(
            is_leaf=True, min_key=leaf_keys[0], keys=leaf_keys,
            recids=list(range(begin, begin + len(leaf_keys))),
        ))
        level.append(len(pool) - 1)
    while len(level) > 1:
        next_level = []
        for begin in range(0, len(level), ORDER):
            children = level[begin:begin + ORDER]
            separators = [pool[child].min_key for child in children[1:]]
            pool.append(TmpNode(
                is_leaf=False, min_key=pool[children[0]].min_key,
                keys=separators, children=children,
            ))
            next_level.append(len(pool) - 1)
        level = next_level
    return pool, level[0]


def tree_height(pool: list[TmpNode], root: int) -> int:
    height = 0
    node = root
    while not pool[node].is_leaf:
        node = pool[node].children[0]
        height += 1
    return height


def flatten_tree(pool: list[TmpNode], root: int) -> list[WireNode]:
    bfs = []
    queue = [root]
    for node in queue:
        bfs.append(node)
        queue.extend(pool[node].children)
    bfs_index = {node: index for index, node in enumerate(bfs)}
    output = []
    for node_index in bfs:
        node = pool[node_index]
        indices = [0] * (ORDER + 1)
        wire_keys = [INT32_MAX] * (ORDER + 1)
        wire_keys[0] = INT32_MIN
        if node.is_leaf:
            for slot, (key, record) in enumerate(
                    zip(node.keys, node.recids), start=1):
                wire_keys[slot] = key
                indices[slot] = record
            num_keys = len(node.keys) + 2
        else:
            for slot, key in enumerate(node.keys, start=1):
                wire_keys[slot] = key
            for slot, child in enumerate(node.children):
                indices[slot] = bfs_index[child]
            num_keys = len(node.keys) + 2
        output.append(WireNode(
            location=bfs_index[node_index], indices=tuple(indices),
            keys=tuple(wire_keys), is_leaf=int(node.is_leaf),
            num_keys=num_keys,
        ))
    return output


def point_oracle(keys: list[int], records: list[int], query: int) -> int:
    """Independent sorted-array lower_bound oracle; it never walks the tree."""
    index = bisect_left(keys, query)
    return records[index] if index < len(keys) and keys[index] == query else -1


def range_oracle(keys: list[int], start: int, end: int) -> tuple[int, int]:
    """Inclusive exact-key range result using two independent lower_bounds."""
    start_index = bisect_left(keys, start)
    end_index = bisect_left(keys, end)
    if (start_index == len(keys) or keys[start_index] != start or
            end_index == len(keys) or keys[end_index] != end or
            end_index < start_index):
        return -1, 0
    return start_index, end_index - start_index + 1


def descend(nodes: list[WireNode], height: int, query: int) -> int:
    node_index = 0
    for _ in range(height):
        node = nodes[node_index]
        next_index = node_index
        for slot in range(ORDER):
            if node.keys[slot] <= query < node.keys[slot + 1]:
                candidate = node.indices[slot]
                if 0 <= candidate < len(nodes):
                    next_index = candidate
                break
        node_index = next_index
    return node_index


def model_record_index(nodes: list[WireNode], height: int,
                       query: int) -> int:
    leaf = nodes[descend(nodes, height, query)]
    for slot in range(ORDER):
        if leaf.keys[slot] == query:
            return leaf.indices[slot]
    return -1


def model_point(nodes: list[WireNode], records: list[int], height: int,
                query: int) -> int:
    record = model_record_index(nodes, height, query)
    return records[record] if 0 <= record < len(records) else -1


def partition_boundaries(total: int, sms: int,
                         granule: int = OUTPUT_RECORDS_PER_LINE) -> list[int]:
    if sms not in (1, 2, 4):
        raise ValueError("sms must be 1, 2, or 4")
    if total <= 0 or granule <= 0:
        raise ValueError("partition sizes must be positive")
    chunks = (total + granule - 1) // granule
    base, extra = divmod(chunks, sms)
    boundaries = [0]
    cursor = 0
    for sm in range(sms):
        cursor += base + int(sm < extra)
        boundaries.append(min(total, cursor * granule))
    return boundaries


def ownership(total: int, sms: int) -> tuple[list[int], list[int]]:
    boundaries = partition_boundaries(total, sms)
    owners = [-1] * total
    visits = [0] * total
    for sm, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        for index in range(begin, end):
            owners[index] = sm
            visits[index] += 1
    if visits != [1] * total:
        raise ValueError("partition does not own every output exactly once")
    for line in range((total + OUTPUT_RECORDS_PER_LINE - 1) //
                      OUTPUT_RECORDS_PER_LINE):
        begin = line * OUTPUT_RECORDS_PER_LINE
        end = min(total, begin + OUTPUT_RECORDS_PER_LINE)
        if len(set(owners[begin:end])) != 1:
            raise ValueError("partition splits an output cache line")
    return owners, boundaries


def output_records(point_expected: list[int], range_expected: list[tuple[int, int]],
                   sms: int) -> tuple[list[tuple[int, int, int, int]],
                                      list[tuple[int, int, int, int]],
                                      dict]:
    point_owners, point_boundaries = ownership(len(point_expected), sms)
    range_owners, range_boundaries = ownership(len(range_expected), sms)
    points = [
        (value, index, point_owners[index], POINT_TAG ^ index)
        for index, value in enumerate(point_expected)
    ]
    ranges = [
        (start, length, range_owners[index], RANGE_TAG ^ index)
        for index, (start, length) in enumerate(range_expected)
    ]
    return points, ranges, {
        "point_boundaries": point_boundaries,
        "range_boundaries": range_boundaries,
        "point_owners": point_owners,
        "range_owners": range_owners,
    }


def validate_parameters(key_count: int, point_count: int, range_count: int,
                        range_span: int, sms: int, loader: str,
                        warps_per_core: int) -> None:
    if key_count < 2 or key_count > 2_000_000:
        raise ValueError("keys must be between 2 and 2000000")
    if 2 * (key_count - 1) > INT32_MAX - PAYLOAD_BASE:
        raise ValueError("keys and payloads must fit int32_t")
    if sms not in (1, 2, 4):
        raise ValueError("sms must be 1, 2, or 4")
    for name, count in (("point_queries", point_count),
                        ("range_queries", range_count)):
        if count < sms * OUTPUT_RECORDS_PER_LINE:
            raise ValueError(f"{name} needs at least one cache line per SM")
        if count % OUTPUT_RECORDS_PER_LINE:
            raise ValueError(f"{name} must be a multiple of four records")
    if range_span < 1:
        raise ValueError("range_span must be positive")
    if loader not in ("logical", "direct-hashed"):
        raise ValueError("loader must be logical or direct-hashed")
    if warps_per_core not in (1, 2, 4, 8):
        raise ValueError("warps_per_core must be 1, 2, 4, or 8")


def prepare(key_count: int, point_count: int, range_count: int,
            range_span: int, out: Path, *, sms: int = 1,
            loader: str = "logical", warps_per_core: int = 8) -> dict:
    validate_parameters(key_count, point_count, range_count, range_span,
                        sms, loader, warps_per_core)
    keys, records = fixture_keys(key_count)
    point_queries, range_queries = fixture_queries(
        key_count, point_count, range_count, range_span)
    pool, root = build_tree(keys)
    height = tree_height(pool, root)
    nodes = flatten_tree(pool, root)
    point_expected = [point_oracle(keys, records, query)
                      for query in point_queries]
    range_expected = [range_oracle(keys, start, end)
                      for start, end in range_queries]

    model_points = [model_point(nodes, records, height, query)
                    for query in point_queries]
    model_ranges = []
    for start, end in range_queries:
        begin = model_record_index(nodes, height, start)
        finish = model_record_index(nodes, height, end)
        model_ranges.append((begin, finish - begin + 1
                             if begin >= 0 and finish >= begin else 0))
    if model_points != point_expected or model_ranges != range_expected:
        raise ValueError("flattened-tree model does not match lower_bound oracle")

    point_records, range_records, partition = output_records(
        point_expected, range_expected, sms)
    point_words = [word for record in point_records for word in record]
    range_words = [word for record in range_records for word in record]

    GENERATED.mkdir(exist_ok=True)
    (GENERATED / "nodes.bin").write_bytes(b"".join(node.pack() for node in nodes))
    write_i32(GENERATED / "records.bin", records)
    write_i32(GENERATED / "point_queries.bin", point_queries)
    write_i32(GENERATED / "range_queries.bin",
              (word for pair in range_queries for word in pair))
    write_i32(GENERATED / "point_expected.bin", point_expected)
    write_i32(GENERATED / "range_expected.bin",
              (word for pair in range_expected for word in pair))

    point_digest = fnv64(point_words)
    range_digest = fnv64(range_words)
    config = [
        "#pragma once",
        f"#define HPC_SM_COUNT {sms}u",
        f"#define NIGHTLY_CLUSTERS {sms}",
        f"#define NIGHTLY_OCC {warps_per_core}",
        f"#define BTREE_ORDER {ORDER}u",
        f"#define BTREE_HEIGHT {height}u",
        f"#define BTREE_NODE_COUNT {len(nodes)}u",
        f"#define BTREE_RECORD_COUNT {len(records)}u",
        f"#define BTREE_POINT_QUERY_COUNT {len(point_queries)}u",
        f"#define BTREE_RANGE_QUERY_COUNT {len(range_queries)}u",
        f"#define BTREE_OUTPUT_RECORDS_PER_LINE {OUTPUT_RECORDS_PER_LINE}u",
        f"#define BTREE_POINT_TAG 0x{POINT_TAG:08x}u",
        f"#define BTREE_RANGE_TAG 0x{RANGE_TAG:08x}u",
        f"#define BTREE_POINT_OUTPUT_DIGEST 0x{point_digest:016x}ULL",
        f"#define BTREE_RANGE_OUTPUT_DIGEST 0x{range_digest:016x}ULL",
    ]
    (GENERATED / "config.h").write_text("\n".join(config) + "\n")
    assembly = [
        '.section .data,"aw",@progbits',
        '.balign 64', '.globl btree_nodes', 'btree_nodes:',
        '.incbin "generated/nodes.bin"',
        '.balign 64', '.globl btree_records', 'btree_records:',
        '.incbin "generated/records.bin"',
        '.balign 64', '.globl btree_point_queries', 'btree_point_queries:',
        '.incbin "generated/point_queries.bin"',
        '.balign 64', '.globl btree_range_queries', 'btree_range_queries:',
        '.incbin "generated/range_queries.bin"',
        '.balign 64', '.globl btree_point_expected', 'btree_point_expected:',
        '.incbin "generated/point_expected.bin"',
        '.balign 64', '.globl btree_range_expected', 'btree_range_expected:',
        '.incbin "generated/range_expected.bin"',
        # Cyclotron launches RV32 directly without executing the fused RV64
        # checker, so guards are initialized in the writable image as well as
        # being refreshed by host.cpp on hardware.
        '.balign 64', '.globl btree_guard_before', 'btree_guard_before:',
        *(f'.long 0x{(0xB7EE0000 ^ index):08x}' for index in range(16)),
        '.globl btree_point_outputs', 'btree_point_outputs:',
        f'.zero {len(point_queries) * OUTPUT_RECORD_BYTES}',
        '.balign 64', '.globl btree_guard_middle', 'btree_guard_middle:',
        *(f'.long 0x{(0xB7DD0000 ^ index):08x}' for index in range(16)),
        '.globl btree_range_outputs', 'btree_range_outputs:',
        f'.zero {len(range_queries) * OUTPUT_RECORD_BYTES}',
        '.balign 64', '.globl btree_guard_after', 'btree_guard_after:',
        *(f'.long 0x{(0xB7AA0000 ^ index):08x}' for index in range(16)),
    ]
    (ROOT / "data.S").write_text("\n".join(assembly) + "\n")

    source_files = (
        ROOT.parent / "addr_hash_stamp.mk",
        ROOT.parents[1] / "lib/include/mu_partition.h",
        ROOT.parents[1] / "lib/include/nightly/device.h",
        ROOT.parents[1] / "lib/include/nightly/host.h",
        ROOT / "run.py", ROOT / "Makefile", ROOT / "kernel.cpp",
        ROOT / "host.cpp", ROOT / "wire_format.hpp", ROOT / "emit_symbols.py",
        ROOT / "model_check.py",
    )
    source_digest = hashlib.sha256()
    for path in source_files:
        source_digest.update(path.relative_to(ROOT.parents[1]).as_posix().encode())
        source_digest.update(path.read_bytes())
    generated_names = (
        "nodes.bin", "records.bin", "point_queries.bin", "range_queries.bin",
        "point_expected.bin", "range_expected.bin", "config.h",
    )
    case = {
        "schema": "radiance-btree-build-v1",
        "name": f"btree--vortex-default--{sms}sm",
        "family": "btree", "kind": "point-and-range-lookup",
        "sm_count": sms, "loader": loader,
        "gpu_address_hash": loader == "direct-hashed",
        "keys": key_count, "point_queries": point_count,
        "range_queries": range_count, "range_span": range_span,
        "order": ORDER, "node_count": len(nodes), "tree_height": height,
        "point_output_records": point_count,
        "range_output_records": range_count,
        "output_record_bytes": OUTPUT_RECORD_BYTES,
        "output_words": (point_count + range_count) * 4,
        "point_output_digest": f"{point_digest:016x}",
        "range_output_digest": f"{range_digest:016x}",
        "correctness": "complete-exact-point-and-range-output-check",
        "oracle": "independent-python-bisect-left-lower-bound",
        "generator_model_check": "passed",
        "partition": "disjoint-contiguous-four-record-64-byte-output-lines",
        "partition_ownership": partition,
        "partition_complete": True, "partition_overlap": 0,
        "partition_cache_line_safe": True,
        "cache_line_bytes": CACHE_LINE_BYTES,
        "scalar_only": True, "no_vpu": True, "vpu": False,
        "vector_isa": False, "gemmini": False, "mx": False,
        "muon_warps_per_core": warps_per_core,
        "robust_end_drain": False,
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
        "official_default_contract": {
            "keys": DEFAULT_KEYS, "point_queries": DEFAULT_POINT_QUERIES,
            "range_queries": DEFAULT_RANGE_QUERIES,
            "range_span": DEFAULT_RANGE_SPAN,
        },
        "is_official_default_canary": (
            key_count == DEFAULT_KEYS and point_count == DEFAULT_POINT_QUERIES
            and range_count == DEFAULT_RANGE_QUERIES
            and range_span == DEFAULT_RANGE_SPAN
        ),
        "benchmark_contract": "vortex-rodinia-btree-point-and-inclusive-range",
        "abi_mapping": {
            "opencl_int": "int32_t",
            "opencl_long_and_cl_long": "uint32_t after bounds validation",
            "knode_bool_plus_padding": "uint32_t 0-or-1 at byte offset 140",
            "knode_size_bytes": NODE_STRUCT.size,
        },
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
    parser.add_argument("--keys", type=int, default=DEFAULT_KEYS)
    parser.add_argument("--point-queries", type=int, default=DEFAULT_POINT_QUERIES)
    parser.add_argument("--range-queries", type=int, default=DEFAULT_RANGE_QUERIES)
    parser.add_argument("--range-span", type=int, default=DEFAULT_RANGE_SPAN)
    parser.add_argument("--sms", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--loader", choices=("logical", "direct-hashed"),
                        default="logical")
    parser.add_argument("--warps-per-core", type=int, choices=(1, 2, 4, 8),
                        default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    case = prepare(
        args.keys, args.point_queries, args.range_queries, args.range_span,
        out, sms=args.sms, loader=args.loader,
        warps_per_core=args.warps_per_core,
    )
    if not args.prepare_only:
        case = build(case, out)
        print(json.dumps(case, indent=2))


if __name__ == "__main__":
    main()
