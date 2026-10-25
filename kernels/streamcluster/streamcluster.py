#!/usr/bin/env python3
"""Prepare/build the bounded scalar Radiance StreamCluster pgain port."""

from __future__ import annotations

from array import array
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
from typing import Iterable

from oracle import Dataset, float_word, fnv_bytes, fnv_words, make_dataset, pgain


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
DEFAULT_REFERENCE = Path(
    "/scratch/agustin/radiance-fpga-1a/reference/vortex"
)

DEFAULT_NUM = 128
DEFAULT_DIM = 32
DEFAULT_SEED = 9
CENTER_STRIDE = 8
LOCAL_SIZE = 16
CACHE_LINE_BYTES = 64
RELATIVE_TOLERANCE = 1.0e-3

REFERENCE_REPOSITORY = "vortexgpgpu/vortex"
REFERENCE_REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"
REFERENCE_KERNEL = "tests/opencl/streamcluster/kernel.cl"
REFERENCE_KERNEL_GIT_BLOB = "be10036d449152cd6523b85bf7c44583abd7d9bb"
REFERENCE_KERNEL_SHA256 = (
    "8b29f8baefd367e250ed80dab6441c365c7924a9aa1db1dd8ea2948a1740b097"
)
REFERENCE_MAIN = "tests/opencl/streamcluster/main.cc"
REFERENCE_MAIN_GIT_BLOB = "97039e1c5bb3c52136cb342ddfed7d2176e6c6ca"
REFERENCE_MAIN_SHA256 = (
    "ea6c8d8acc2cb1017f73dcf74eb95dbc261c5dd1672b916e92c82809ee57fedc"
)
REFERENCE_MAKEFILE = "tests/opencl/streamcluster/Makefile"
REFERENCE_MAKEFILE_GIT_BLOB = "1471bde239819f5fe53b89ed62f0d60cb829beed"
REFERENCE_MAKEFILE_SHA256 = (
    "accfb9fcf8d13e8b4924235c7929144846a372d3c044c483aa85dc61b9297e2a"
)

OFFICIAL_HASHES = {
    "coordinates.bin": "682f95746edcb932db91bcd2bddcf1543dd13b0b92fffead9ef43021d45c9c3d",
    "points.bin": "caaf76388c773c66b20eeaf8d945b86d7d14117f0584f8c2eb6f9f4dea46f52e",
    "center_table.bin": "0525b8285a2c80df912a9da8de75c477d9d5b786d22e1a55f3af31e34cdec755",
    "work_expected.bin": "9c0f00f9961480adc3e8469006eaa8b40793bd18835b62b5bee4d12b2a316a1f",
    "switch_expected.bin": "38723a2e5e8a17aa7950dc008209944e898f69a7bd10a23c839d341e935fd5ca",
}
OFFICIAL_WORK_DIGEST = 0x35D322C643A905EA
OFFICIAL_SWITCH_DIGEST = 0x8421AE126C7CED25


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def git_output(repository: Path, *arguments: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], stderr=subprocess.STDOUT,
    )


def verify_reference(repository: Path) -> dict:
    if not repository.is_dir():
        raise ValueError(f"pinned reference repository is missing: {repository}")
    head = git_output(repository, "rev-parse", "HEAD").decode().strip()
    if head != REFERENCE_REVISION:
        raise ValueError(f"reference HEAD mismatch: {head}")
    files = (
        (REFERENCE_KERNEL, REFERENCE_KERNEL_GIT_BLOB, REFERENCE_KERNEL_SHA256),
        (REFERENCE_MAIN, REFERENCE_MAIN_GIT_BLOB, REFERENCE_MAIN_SHA256),
        (REFERENCE_MAKEFILE, REFERENCE_MAKEFILE_GIT_BLOB,
         REFERENCE_MAKEFILE_SHA256),
    )
    verified = {}
    for path, expected_blob, expected_sha in files:
        blob = git_output(repository, "rev-parse", f"{REFERENCE_REVISION}:{path}")
        blob_id = blob.decode().strip()
        contents = git_output(repository, "show", f"{REFERENCE_REVISION}:{path}")
        contents_sha = hashlib.sha256(contents).hexdigest()
        if blob_id != expected_blob or contents_sha != expected_sha:
            raise ValueError(f"pinned reference mismatch for {path}")
        verified[path] = {"git_blob": blob_id, "sha256": contents_sha}
    return {"repository": str(repository.resolve()), "head": head,
            "files": verified}


def pack_f32(values: Iterable[float]) -> bytes:
    return b"".join(struct.pack("<f", value) for value in values)


def pack_i32(values: Iterable[int]) -> bytes:
    return b"".join(struct.pack("<i", value) for value in values)


def pack_points(dataset: Dataset) -> bytes:
    return b"".join(
        struct.pack("<fif", point.weight, point.assign, point.cost)
        for point in dataset.points
    )


def point_granule(k: int) -> int:
    row_bytes = (k + 1) * 4
    return CACHE_LINE_BYTES // math.gcd(row_bytes, CACHE_LINE_BYTES)


def point_boundaries(num: int, k: int, sms: int) -> list[int]:
    if sms not in (1, 2, 4):
        raise ValueError("sms must be 1, 2, or 4")
    granule = point_granule(k)
    if num % granule:
        raise ValueError(
            f"num must be a multiple of {granule} points for work-row line safety")
    chunks = num // granule
    if chunks < sms:
        raise ValueError("dataset needs at least one cache-line-safe chunk per SM")
    base, extra = divmod(chunks, sms)
    boundaries = [0]
    for sm in range(sms):
        boundaries.append(boundaries[-1] + (base + (sm < extra)) * granule)
    if boundaries[-1] != num:
        raise AssertionError("partition arithmetic did not cover all points")
    return boundaries


def ownership_metadata(num: int, k: int, sms: int) -> tuple[list[int], list[dict], int]:
    boundaries = point_boundaries(num, k, sms)
    row_bytes = (k + 1) * 4
    maximum_points = max(b - a for a, b in zip(boundaries, boundaries[1:]))
    switch_stride = ((maximum_points + 63) // 64) * 64
    ownership = []
    visits = [0] * num
    work_lines: list[set[int]] = []
    switch_lines: list[set[int]] = []
    for sm, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        for point in range(begin, end):
            visits[point] += 1
        work_begin = begin * row_bytes
        work_end = end * row_bytes
        switch_begin = sm * switch_stride
        switch_end = switch_begin + (end - begin)
        if work_begin % 64 or work_end % 64 or switch_begin % 64:
            raise ValueError("ownership boundary is not cache-line aligned")
        work_lines.append(set(range(work_begin // 64, work_end // 64)))
        switch_lines.append(set(range(switch_begin // 64,
                                      (switch_end + 63) // 64)))
        ownership.append({
            "physical_threadblock_id": sm,
            "point_begin_inclusive": begin,
            "point_end_exclusive": end,
            "point_count": end - begin,
            "work_byte_begin": work_begin,
            "work_byte_end": work_end,
            "switch_storage_byte_begin": switch_begin,
            "switch_storage_byte_end": switch_end,
            "switch_segment_byte_end": switch_begin + switch_stride,
        })
    if visits != [1] * num:
        raise ValueError("point partition has a gap or overlap")
    for lhs in range(sms):
        for rhs in range(lhs + 1, sms):
            if work_lines[lhs] & work_lines[rhs]:
                raise ValueError("SMs share a work_mem output cache line")
            if switch_lines[lhs] & switch_lines[rhs]:
                raise ValueError("SMs share a switch output cache line")
    return boundaries, ownership, switch_stride


def physical_switch_image(expected: bytes, boundaries: list[int],
                          stride: int, padding: int) -> bytes:
    image = bytearray([padding] * (stride * (len(boundaries) - 1)))
    for sm, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        image[sm * stride:sm * stride + end - begin] = expected[begin:end]
    return bytes(image)


def write_blob(path: Path, contents: bytes) -> None:
    path.write_bytes(contents)


def source_digest() -> str:
    value = hashlib.sha256()
    paths = (
        ROOT.parent / "addr_hash_stamp.mk",
        ROOT.parents[1] / "lib/include/nightly/device.h",
        ROOT.parents[1] / "lib/include/nightly/host.h",
        ROOT / ".gitignore", ROOT / "Makefile", ROOT / "README.md",
        ROOT / "emit_symbols.py", ROOT / "functional_smoke.py",
        ROOT / "host.cpp", ROOT / "kernel.cpp", ROOT / "kernel_impl.hpp",
        ROOT / "oracle.py", ROOT / "streamcluster.py",
        ROOT / "test_streamcluster.py",
    )
    for path in paths:
        value.update(path.relative_to(ROOT.parents[1]).as_posix().encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def kernel_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT.parents[1]), "rev-parse", "HEAD"], text=True,
    ).strip()


def prepare(num: int, dim: int, out: Path, *, sms: int = 1,
            loader: str = "logical", warps_per_core: int = 8,
            reference_repo: Path = DEFAULT_REFERENCE,
            robust_end_drain: bool = False,
            robust_spin_until_reset: bool = False,
            robust_serial_fence: bool = False,
            robust_post_spin: int = 500) -> dict:
    if num < 8 or num % CENTER_STRIDE:
        raise ValueError("num must be at least 8 and divisible by center stride 8")
    if dim < 1:
        raise ValueError("dim must be positive")
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

    reference = verify_reference(reference_repo.resolve())
    dataset = make_dataset(num=num, dim=dim, seed=DEFAULT_SEED,
                           center_stride=CENTER_STRIDE, candidate=num // 2)
    result = pgain(dataset)
    k = dataset.k
    boundaries, ownership, switch_stride = ownership_metadata(num, k, sms)
    work_words = [float_word(value) for value in result.work]
    work_digest = fnv_words(work_words)
    switch_digest = fnv_bytes(result.switch)

    GENERATED.mkdir(exist_ok=True)
    blobs = {
        "coordinates.bin": pack_f32(dataset.coordinates),
        "points.bin": pack_points(dataset),
        "center_table.bin": pack_i32(dataset.center_table),
        "work_expected.bin": pack_f32(result.work),
        "switch_expected.bin": result.switch,
    }
    for name, contents in blobs.items():
        write_blob(GENERATED / name, contents)
    generated_hashes = {name: sha256(GENERATED / name) for name in blobs}

    official = num == DEFAULT_NUM and dim == DEFAULT_DIM
    if official:
        if generated_hashes != OFFICIAL_HASHES:
            raise ValueError("official deterministic dataset/output hash mismatch")
        if (work_digest != OFFICIAL_WORK_DIGEST or
                switch_digest != OFFICIAL_SWITCH_DIGEST):
            raise ValueError("official deterministic output digest mismatch")

    # The fused RV64 checker poisons padding with 0xa5. Cyclotron executes the
    # RV32 device ELF directly, so untouched BSS padding remains zero there.
    runtime_switch = physical_switch_image(
        result.switch, boundaries, switch_stride, 0xA5)
    functional_switch = physical_switch_image(
        result.switch, boundaries, switch_stride, 0x00)
    runtime_switch_words = struct.unpack(
        f"<{len(runtime_switch) // 4}I", runtime_switch)
    functional_switch_words = struct.unpack(
        f"<{len(functional_switch) // 4}I", functional_switch)
    runtime_switch_digest = fnv_words(runtime_switch_words)
    functional_switch_digest = fnv_words(functional_switch_words)
    boundary_initializer = "{" + ",".join(
        f"{value}u" for value in boundaries) + "}"
    config = [
        "#pragma once",
        f"#define HPC_SM_COUNT {sms}u",
        f"#define NIGHTLY_CLUSTERS {sms}",
        f"#define NIGHTLY_OCC {warps_per_core}",
        f"#define NIGHTLY_ROBUST_GMEM_END {int(robust_end_drain)}",
        f"#define NIGHTLY_ROBUST_SPIN_UNTIL_RESET {int(robust_spin_until_reset)}",
        f"#define NIGHTLY_ROBUST_SERIAL_FENCE {int(robust_serial_fence)}",
        f"#define NIGHTLY_ROBUST_POST_SPIN {robust_post_spin}u",
        f"#define STREAMCLUSTER_NUM {num}u",
        f"#define STREAMCLUSTER_DIM {dim}u",
        f"#define STREAMCLUSTER_K {k}u",
        f"#define STREAMCLUSTER_CANDIDATE {dataset.candidate}u",
        f"#define STREAMCLUSTER_LOCAL_SIZE {LOCAL_SIZE}u",
        f"#define STREAMCLUSTER_WORK_ROW_FLOATS {k + 1}u",
        f"#define STREAMCLUSTER_WORK_ROW_BYTES {(k + 1) * 4}u",
        f"#define STREAMCLUSTER_WORK_FLOATS {num * (k + 1)}u",
        f"#define STREAMCLUSTER_POINT_GRANULE {point_granule(k)}u",
        f"#define STREAMCLUSTER_SM_POINT_BOUNDARIES {boundary_initializer}",
        f"#define STREAMCLUSTER_SWITCH_SEGMENT_STRIDE {switch_stride}u",
        f"#define STREAMCLUSTER_SWITCH_STORAGE_BYTES {switch_stride * sms}u",
        f"#define STREAMCLUSTER_REL_TOL_BITS 0x{float_word(RELATIVE_TOLERANCE):08x}u",
        f"#define STREAMCLUSTER_WORK_EXPECTED_DIGEST 0x{work_digest:016x}ULL",
        f"#define STREAMCLUSTER_SWITCH_EXPECTED_DIGEST 0x{switch_digest:016x}ULL",
        f"#define STREAMCLUSTER_SWITCH_RUNTIME_DIGEST 0x{runtime_switch_digest:016x}ULL",
        f"#define STREAMCLUSTER_SWITCH_FUNCTIONAL_DIGEST 0x{functional_switch_digest:016x}ULL",
    ]
    (GENERATED / "config.h").write_text("\n".join(config) + "\n")

    assembly = [
        '.section .data,"aw",@progbits',
        '.balign 64', '.globl streamcluster_points',
        'streamcluster_points:', '.incbin "generated/points.bin"',
        '.balign 64', '.globl streamcluster_coordinates',
        'streamcluster_coordinates:', '.incbin "generated/coordinates.bin"',
        '.balign 64', '.globl streamcluster_center_table',
        'streamcluster_center_table:', '.incbin "generated/center_table.bin"',
        '.balign 64', '.globl streamcluster_work_expected',
        'streamcluster_work_expected:', '.incbin "generated/work_expected.bin"',
        '.balign 64', '.globl streamcluster_switch_expected',
        'streamcluster_switch_expected:', '.incbin "generated/switch_expected.bin"',
        '.section .bss,"aw",@nobits', '.balign 64',
        '.globl streamcluster_work_guard_before',
        'streamcluster_work_guard_before:', '.zero 64',
        '.globl streamcluster_work_output', 'streamcluster_work_output:',
        f'.zero {num * (k + 1) * 4}', '.balign 64',
        '.globl streamcluster_work_guard_after',
        'streamcluster_work_guard_after:', '.zero 64',
        '.globl streamcluster_switch_guard_before',
        'streamcluster_switch_guard_before:', '.zero 64',
        '.globl streamcluster_switch_output', 'streamcluster_switch_output:',
        f'.zero {switch_stride * sms}', '.balign 64',
        '.globl streamcluster_switch_guard_after',
        'streamcluster_switch_guard_after:', '.zero 64',
    ]
    (ROOT / "data.S").write_text("\n".join(assembly) + "\n")

    ownership_sha = hashlib.sha256(
        json.dumps(ownership, sort_keys=True).encode()).hexdigest()
    manifest = {
        "name": f"streamcluster-pgain--n{num}-d{dim}--{sms}sm",
        "family": "streamcluster", "kind": "rodinia-pgain",
        "sm_count": sms, "loader": loader,
        "gpu_address_hash": loader == "direct-hashed",
        "num": num, "dim": dim, "k": k,
        "seed": DEFAULT_SEED, "center_stride": CENTER_STRIDE,
        "candidate": dataset.candidate, "local_size": LOCAL_SIZE,
        "point_abi": "float-weight,int32-assign,float-cost",
        "coordinate_layout": "dimension-major-coordinate[dimension*num+point]",
        "initial_assignment": "nearest-open-center-strict-less-tie-first",
        "benchmark_contract": (
            "pinned-vortex-streamcluster-default-num128-dim32-local16-"
            "full-sequence-fused-memset"
            if official else
            "derived-size-pinned-vortex-streamcluster-equation-input-sequence-"
            "and-full-sequence-fused-memset"
        ),
        "timing_contract": "fused-memset-plus-pgain-full-sequence",
        "official_default_contract": official,
        "derived": not official,
        "equation": "x_cost=weight*sum_dim((coord[dim,point]-coord[dim,x])^2)",
        "branch_semantics": (
            "if x_cost<current: switch='1',work[row,K]=x_cost-current; "
            "else: work[row,center_table[assign]]+=current-x_cost"
        ),
        "accumulation_order": "strict-increasing-dimension-no-contraction",
        "memset_semantics": "zero-work-and-switch-fused-per-owned-point",
        "output_work_float_count": len(result.work),
        "output_switch_logical_byte_count": len(result.switch),
        "switch_true_count": result.switch.count(ord("1")),
        "work_expected_digest": f"{work_digest:016x}",
        "switch_expected_digest": f"{switch_digest:016x}",
        "switch_runtime_expected_digest": f"{runtime_switch_digest:016x}",
        "switch_functional_expected_digest": f"{functional_switch_digest:016x}",
        "correctness": (
            "every-work-float-upstream-tolerance-and-exact-digest;"
            "every-logical-switch-byte-exact;guards;untouched-switch-padding"
        ),
        "oracle": "independent-ordered-float32-cpu-oracle",
        "relative_tolerance": RELATIVE_TOLERANCE,
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_revision": REFERENCE_REVISION,
        "reference_kernel": REFERENCE_KERNEL,
        "reference_kernel_git_blob": REFERENCE_KERNEL_GIT_BLOB,
        "reference_kernel_sha256": REFERENCE_KERNEL_SHA256,
        "reference_main": REFERENCE_MAIN,
        "reference_main_git_blob": REFERENCE_MAIN_GIT_BLOB,
        "reference_main_sha256": REFERENCE_MAIN_SHA256,
        "reference_makefile": REFERENCE_MAKEFILE,
        "reference_makefile_git_blob": REFERENCE_MAKEFILE_GIT_BLOB,
        "reference_makefile_sha256": REFERENCE_MAKEFILE_SHA256,
        "reference_verification": reference,
        "dataset_hashes": generated_hashes,
        "official_hashes_verified": official,
        "partition": "physical-threadblock-contiguous-complete-point-rows",
        "partition_boundaries": boundaries,
        "partition_point_granule": point_granule(k),
        "partition_ownership": ownership,
        "partition_ownership_sha256": ownership_sha,
        "partition_complete": True, "partition_overlap": False,
        "partition_work_cache_line_safe": True,
        "partition_switch_cache_line_safe": True,
        "switch_segment_stride": switch_stride,
        "switch_storage_bytes": switch_stride * sms,
        "cache_line_bytes": CACHE_LINE_BYTES,
        "scalar_only": True, "no_vpu": True, "vpu": False,
        "vector_isa": False, "gemmini": False, "mx": False,
        "generator_model_check": "passed",
        "muon_warps_per_core": warps_per_core,
        "robust_end_drain": robust_end_drain,
        "robust_spin_until_reset": robust_spin_until_reset,
        "robust_serial_fence": robust_serial_fence,
        "robust_post_spin": robust_post_spin,
        "kernel_revision": kernel_revision(),
        "kernel_source_sha256": source_digest(),
        "generator_platform": platform.platform(),
        "status": "prepared", "simulator": None,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num", type=int, default=DEFAULT_NUM)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--sms", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--loader", choices=("logical", "direct-hashed"),
                        default="logical")
    parser.add_argument("--warps-per-core", type=int, choices=(1, 2, 4, 8),
                        default=8)
    parser.add_argument("--reference-repo", type=Path,
                        default=Path(os.environ.get(
                            "STREAMCLUSTER_REFERENCE_REPO", DEFAULT_REFERENCE)))
    parser.add_argument("--robust-end-drain", action="store_true")
    parser.add_argument("--robust-spin-until-reset", action="store_true")
    parser.add_argument("--robust-serial-fence", action="store_true")
    parser.add_argument("--robust-post-spin", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    out = args.out.resolve()
    manifest = prepare(
        args.num, args.dim, out, sms=args.sms, loader=args.loader,
        warps_per_core=args.warps_per_core,
        reference_repo=args.reference_repo,
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
    local_toolchain = ROOT.parents[3] / "toolchains/riscv-tools-prefix/riscv-tools"
    if "RISCV64_TOOLCHAIN_PATH" not in env and local_toolchain.is_dir():
        env["RISCV64_TOOLCHAIN_PATH"] = str(local_toolchain)
    with (out / "build.log").open("w") as log:
        subprocess.run(["make", "kernel.soc.elf"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, env=env)
    elf = out / "kernel.soc.elf"
    device_elf = out / "kernel.radiance.elf"
    shutil.copy2(ROOT / "kernel.soc.elf", elf)
    shutil.copy2(ROOT / "kernel.radiance.elf", device_elf)
    manifest.update({
        "status": "built", "elf": str(elf), "elf_sha256": sha256(elf),
        "bytes": elf.stat().st_size, "device_elf": str(device_elf),
        "device_elf_sha256": sha256(device_elf),
    })
    (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
