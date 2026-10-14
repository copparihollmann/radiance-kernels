#!/usr/bin/env python3
"""Map compiled SmolVLA stage markers to the graph's execution regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from verify_functional import sha


SCHEDULE = re.compile(r"(?m)^  mu_schedule\(stage_([A-Za-z0-9_]+), nullptr, 4\);$")
REGION = re.compile(r"^v_(camera[0-2]|vlm|denoise[0-9])_")
CONDITIONING = (
    "v_language_embedding", "v_language_scale", "v_robot_state_linear",
    "v_robot_state_proj", "v_prefix_merge", "v_prefix_pad_mask",
    "v_prefix_attention_groups", "v_prefix_attention_mask",
    "v_prefix_position_ids",
)
EXPECTED = [
    *((f"camera{index}", 221) for index in range(3)),
    ("conditioning", 9),
    ("vlm", 241),
    *((f"denoise{index}", 276) for index in range(10)),
]


def region(symbol: str) -> str:
    match = REGION.match(symbol)
    if match:
        return match.group(1)
    if symbol in CONDITIONING:
        return "conditioning"
    raise ValueError(f"unrecognized stage: {symbol}")


def report(build: Path) -> dict:
    target = build / "smolvla_base"
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    elf = target / "kernel.radiance.elf"
    kernel = target / "kernel.cpp"
    if (manifest["stages"] != 3673
            or manifest["full_graph_stages"] != 3673
            or manifest["progress_every_stages"] != 10
            or manifest["scheduled_warps_per_stage"] != 4
            or manifest["use_all_cores"] is not True
            or manifest["device_elf_built"] is not True
            or manifest["output_validation"] != "all_elements_vs_upstream_policy"
            or sha(elf) != manifest["radiance_elf_sha256"]):
        raise ValueError("expected complete instrumented SmolVLA action ELF")
    source = kernel.read_text()
    if source.count("  mu_schedule(init_inputs, nullptr, 4);\n") != 1:
        raise ValueError("input initialization is missing or duplicated")
    stages = SCHEDULE.findall(source)
    if len(stages) != 3673:
        raise ValueError("compiled stage count differs from manifest")
    if tuple(stages[663:672]) != CONDITIONING:
        raise ValueError("conditioning prefix differs from compiled graph")
    labels = [region(symbol) for symbol in stages]
    groups = []
    start = 0
    for label, length in EXPECTED:
        end = start + length
        if labels[start:end] != [label] * length:
            raise ValueError(f"noncontiguous or incomplete {label} region")
        groups.append({
            "region": label,
            "first_stage_one_based": start + 1,
            "last_stage_one_based": end,
            "stage_count": length,
            "first_stage_symbol": stages[start],
            "last_stage_symbol": stages[end - 1],
            "first_marker_after_region": min(3673, ((end + 9) // 10) * 10),
        })
        start = end
    if start != len(stages):
        raise ValueError("unmapped tail stages")
    markers = [0, *range(10, 3674, 10), 3673]
    actual_markers = [int(value) for value in re.findall(
        r"(?m)^  smolvla_progress\((\d+)u\);$", source)]
    if actual_markers != markers:
        raise ValueError("compiled marker sequence differs from expected interval")
    return {
        "schema": "radiance.smolvla-progress-map.v1",
        "model": "smolvla_base",
        "scope": "full_checkpoint_action_chunk",
        "status": "verified_compiled_schedule",
        "stages": len(stages),
        "progress_interval_stages": 10,
        "progress_marker_count": len(markers),
        "regions": groups,
        "execution_schedule_sha256": manifest["execution_schedule_sha256"],
        "device_elf_sha256": sha(elf),
        "build_manifest_sha256": sha(manifest_path),
        "kernel_source_sha256": sha(kernel),
        "full_action_chunk_device_execution": False,
        "rtl_execution": False,
        "performance_measurement": False,
        "source_sha256": sha(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report(args.build), indent=2) + "\n")


if __name__ == "__main__":
    main()
