#!/usr/bin/env python3
"""Verify SmolVLA stage markers on a passing and timed-out device run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from verify_smolvla_all_cores import case, sha


HERE = Path(__file__).resolve().parent


def verify(baseline_root: Path, baseline_rebuilt: Path,
           progress_root: Path, progress_rebuilt: Path,
           timeout_root: Path, simulator: Path) -> dict:
    simulator_hash = sha(simulator)
    baseline, _ = case(baseline_root, baseline_rebuilt, 1, simulator_hash)
    progress, data = case(progress_root, progress_rebuilt, 1, simulator_hash)
    manifest = data["manifest"]
    target = progress_root / "smolvla_base"
    result_path = target / "functional-result.json"
    result = json.loads(result_path.read_text())
    log = (target / "functional.log").read_text()
    kernel = (target / "kernel.cpp").read_text()
    markers = re.findall(r"(?m)^S([0-9a-f]{4})$", log)
    if (manifest["progress_every_stages"] != 1
            or markers != ["0000", "0001"]
            or result["completed_stage_markers"] != [0, 1]
            or result["last_completed_stage_marker"] != 1
            or "smolvla_progress(0u);" not in kernel
            or "smolvla_progress(1u);" not in kernel
            or "*(volatile __global uint8_t*)IO_COUT_ADDR" not in kernel
            or "#include <VX_config.h>" not in kernel):
        raise ValueError("passing ELF lacks ordered progress markers")
    for field in ("weight_image_sha256", "input_image_sha256",
                  "golden_image_sha256"):
        if baseline[field] != progress[field]:
            raise ValueError(f"progress control changed {field}")
    old = baseline_root / "smolvla_base"
    for filename in ("model_data.cpp", "stage_chunk_000.cpp", "host.cpp"):
        if sha(old / filename) != sha(target / filename):
            raise ValueError(f"progress control changed {filename}")
    timed = timeout_root / "smolvla_base"
    timeout_result_path = timed / "functional-result.json"
    timeout_result = json.loads(timeout_result_path.read_text())
    timeout_log_path = timed / "functional.log"
    timeout_log = timeout_log_path.read_text()
    if (sha(timed / "kernel.radiance.elf") != progress["elf_sha256"]
            or timeout_result["device_elf_sha256"] != progress["elf_sha256"]
            or timeout_result["simulator_sha256"] != simulator_hash
            or timeout_result["log_sha256"] != sha(timeout_log_path)
            or timeout_result["status"] != "failed"
            or timeout_result["failure_reason"] != "wall_clock_timeout"
            or timeout_result["process_exit_code"] != 124
            or timeout_result["cycles_functional"] is not None
            or timeout_result["tohost"] is not None
            or timeout_result["completed_stage_markers"] != [0]
            or timeout_result["last_completed_stage_marker"] != 0
            or timeout_result["progress_markers_valid"] is not True
            or re.findall(r"(?m)^S([0-9a-f]{4})$", timeout_log) != ["0000"]
            or "isa-test passed" in timeout_log):
        raise ValueError("timeout did not retain a valid partial stage log")
    public_pass = HERE / "evaluation/smolvla-progress-stage1-functional.json"
    public_timeout = HERE / "evaluation/smolvla-progress-timeout.json"
    if (sha(public_pass) != sha(result_path)
            or sha(public_timeout) != sha(timeout_result_path)):
        raise ValueError("published result receipts differ from raw runs")
    return {
        "schema": "radiance.smolvla-progress-control.v1",
        "status": "passed",
        "scope": "exact-input first vision stage with upstream PyTorch golden",
        "checked_output_elements": 786432,
        "baseline_cycles_functional": baseline["cycles_functional"],
        "progress_cycles_functional": progress["cycles_functional"],
        "progress_cycle_difference": progress["cycles_functional"] - baseline[
            "cycles_functional"],
        "progress_markers": [0, 1],
        "timeout_status": "wall_clock_timeout_as_expected",
        "timeout_last_completed_stage_marker": 0,
        "device_elf_sha256": progress["elf_sha256"],
        "passing_result_sha256": sha(result_path),
        "passing_log_sha256": sha(target / "functional.log"),
        "timeout_result_sha256": sha(timeout_result_path),
        "timeout_log_sha256": sha(timeout_log_path),
        "simulator_sha256": simulator_hash,
        "console_header_sha256": sha(HERE.parents[1] / "lib/include/VX_config.h"),
        "full_action_chunk_device_execution": False,
        "rtl_execution": False,
        "performance_measurement": False,
        "source_sha256": sha(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-rebuilt", type=Path, required=True)
    parser.add_argument("--progress-root", type=Path, required=True)
    parser.add_argument("--progress-rebuilt", type=Path, required=True)
    parser.add_argument("--timeout-root", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.baseline_root, args.baseline_rebuilt,
                    args.progress_root, args.progress_rebuilt,
                    args.timeout_root, args.simulator)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
