#!/usr/bin/env python3
"""Assemble a hash-pinned model/campaign manifest from existing ELF builds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[3]
PASSTHROUGH = (
    "case", "collision_policy", "destination_group_count",
    "muon_warps_per_core", "robust_end_drain", "robust_spin_until_reset",
    "robust_serial_fence", "robust_token_fence", "robust_drain_passes",
    "robust_post_spin",
    "robust_cluster_l1_manager", "robust_split_l1_drain",
    "robust_completion_mode",
    "gather_iteration_partition", "gather_scratch_elements",
    "gather_block_cyclic_16", "gather_block_cyclic_elements",
    "stream_block_cyclic_lines", "stream_block_cyclic_chunk_lines",
    "dot_block_cyclic_chunk_elements", "dot_block_cyclic_chunk_bytes_per_input",
    "pattern_length", "host_full_readback",
    "expected_pattern_digest", "expected_source_digest",
    "diagnostic_readback",
    "source_pad_cache_lines", "output_pad_cache_lines",
    "rows", "cols", "ilp", "unroll", "output_words", "element_format",
    "accumulation", "partition", "reference_contract", "no_vpu",
    "benchmark_contract",
    "size", "stream_elements", "output_element_count", "expected_digest",
    "expected_final_word", "host_final_reduction", "max_ulp",
    "reference_repository", "reference_revision", "reference_kernel",
    "reference_kernel_sha256", "reference_kernel_git_blob",
    "reference_main", "reference_main_sha256", "reference_converter",
    "reference_converter_sha256", "generated_sha256",
    "reference_seed", "records", "expected_top_indices",
    "device_elf_sha256",
    "dimension", "stored_nnz", "expanded_nnz", "jds_padded_entries",
    "jds_padding_entries", "jds_depth", "jds_warp_size", "jds_padded_rows",
    "partition_boundaries", "partition_nnz", "partition_complete",
    "partition_overlap", "accumulation_order", "scalar_only", "vpu",
    "partition_row_boundaries", "partition_ownership",
    "partition_ownership_sha256", "partition_cache_line_safe",
    "cache_line_bytes", "dataset_contract", "dataset_contract_preserved",
    "vector_isa", "gemmini", "mx", "padding_index_policy", "dataset",
    "dataset_source", "dataset_files", "upstream_gold_validation",
    "generator_model_check", "absolute_tolerance", "relative_tolerance",
    "oracle", "equation", "timing_contract", "official_default_contract",
    "derived", "num", "dim", "k", "candidate", "local_size",
    "output_work_float_count", "output_switch_logical_byte_count",
    "work_expected_digest", "switch_expected_digest",
    "switch_functional_expected_digest", "switch_storage_bytes",
    "partition_work_cache_line_safe", "partition_switch_cache_line_safe",
    "switch_segment_stride", "point_abi", "coordinate_layout",
    "initial_assignment", "branch_semantics", "memset_semantics",
    "dimensions", "padded_dimensions", "timesteps", "layout",
    "active_cells", "destination_populations_checked",
    "velocity_words_checked", "expected_population_digest",
    "expected_output_raw_digest", "expected_output_raw_words",
    "gold_velocity_digest", "upstream_velocity_tolerance",
    "operation_order", "reference_files", "reference_verified",
    "official_dimensions_verified", "official_default_one_timestep_verified",
    "workload_profile", "keys", "point_queries", "range_queries",
    "range_span", "order", "node_count", "tree_height",
    "point_output_records", "range_output_records", "output_record_bytes",
    "point_output_digest", "range_output_digest", "estimated_data_bytes",
    "is_official_default_canary", "is_full_size_performance_workload",
    "derived_performance_contract", "abi_mapping", "reference_makefile",
    "reference_makefile_sha256", "reference_makefile_git_blob",
    "profile", "nx", "ny", "nz", "iterations", "columns", "updates",
    "stage_expected_digests", "owner_digest", "scalar_no_vpu",
    "cells", "seed", "wall_value_range", "output_digest", "owner_words",
    "phase_order", "boundary_rule", "intermediate_strategy",
    "thread_schedule", "phase_synchronization", "work_accounting",
    "is_upstream_sized_canary", "is_meaningful_workload",
    "meaningful_workload_contract", "stages", "fan1_stages",
    "fan2_stages", "useful_scalar_updates",
    "upstream_generated_default_size",
    "meaningful_vs_upstream_default_cubic_work_ratio", "owner_stride",
    "solution", "relative_backward_error", "residual_norm",
    "residual_tolerance", "residual_tolerance_justification",
    "final_digests",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def operation(result: dict) -> str | None:
    kind = result.get("kind")
    if kind == "scatter":
        return f"scatter-{result.get('collision_policy', 'parallel')}"
    return kind


def assemble(root: Path, family: str) -> dict:
    rows = []
    keys = set()
    source_revisions = set()
    source_hashes = set()
    for result_path in sorted(root.rglob("result.json")):
        elf = result_path.parent / "kernel.soc.elf"
        if not elf.is_file():
            continue
        result = json.loads(result_path.read_text())
        if result.get("status") != "built":
            raise ValueError(f"build is not successful: {result_path}")
        name = result_path.parent.name
        key = (name, result["loader"], int(result["sm_count"]))
        if key in keys:
            raise ValueError(f"duplicate workload identity: {key}")
        keys.add(key)
        row = {
            "name": name,
            "family": family,
            "sm_count": int(result["sm_count"]),
            "loader": result["loader"],
            "elf": str(elf.resolve()),
            "elf_sha256": sha256(elf),
            "bytes": elf.stat().st_size,
            "correctness": result.get("correctness"),
            "status": "built",
        }
        if result.get("device_elf"):
            device_elf = Path(result["device_elf"]).resolve()
            if not device_elf.is_file():
                raise ValueError(f"device ELF is missing: {device_elf}")
            device_hash = sha256(device_elf)
            if (result.get("device_elf_sha256") and
                    result["device_elf_sha256"] != device_hash):
                raise ValueError(f"device ELF changed: {device_elf}")
            row["device_elf"] = str(device_elf)
            row["device_elf_sha256"] = device_hash
        op = operation(result)
        if op is not None:
            row["operation"] = op
        for key in PASSTHROUGH:
            if key in result:
                row[key] = result[key]
        rows.append(row)
        if result.get("kernel_revision"):
            source_revisions.add(result["kernel_revision"])
        if result.get("kernel_source_sha256"):
            source_hashes.add(result["kernel_source_sha256"])
    if not rows:
        raise ValueError(f"no completed ELF builds found under {root}")
    revision = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain=v1"], text=True,
    ))
    return {
        "schema": "radiance-fpga-1a-existing-builds-v1",
        "current_source_revision": revision,
        "current_source_dirty": dirty,
        "artifact_base_revisions": sorted(source_revisions),
        "artifact_source_hashes": sorted(source_hashes),
        "no_vpu": True,
        "hardware_rebuilt": False,
        "sms": sorted({row["sm_count"] for row in rows}),
        "loaders": sorted({row["loader"] for row in rows}),
        "workloads": sorted(rows, key=lambda row: row["name"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", required=True,
                        choices=("stream", "stream_dot", "spatter", "xrage",
                                 "fp8_gemm", "gemv_bf16", "stencil3d",
                                 "nearn", "spmv_jds", "sfilter",
                                 "streamcluster", "lbm", "btree",
                                 "hotspot3d", "pathfinder", "gaussian"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = assemble(args.root.resolve(), args.family)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"recorded {len(manifest['workloads'])} builds in {args.out}")


if __name__ == "__main__":
    main()
