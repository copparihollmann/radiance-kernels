#!/usr/bin/env python3
"""Check the full SmolVLA progress ELF against the uninstrumented build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from verify_smolvla_full_warp_build import sha


def verify(baseline_root: Path, progress_root: Path) -> dict:
    old = baseline_root / "smolvla_base"
    new = progress_root / "smolvla_base"
    baseline = json.loads((old / "manifest.json").read_text())
    progress = json.loads((new / "manifest.json").read_text())
    if (baseline["stages"] != progress["stages"] or progress["stages"] != 3673
            or baseline["device_elf_built"] is not True
            or progress["device_elf_built"] is not True
            or baseline["use_all_cores"] is not True
            or progress["use_all_cores"] is not True
            or baseline["scheduled_warps_per_stage"] != 4
            or progress["scheduled_warps_per_stage"] != 4
            or progress["progress_every_stages"] != 10
            or progress["output_validation"] != "all_elements_vs_upstream_policy"):
        raise ValueError("expected complete four-warp, two-core action builds")
    for field in ("source_spec_sha256", "execution_schedule_sha256",
                  "weight_image_sha256", "input_image_sha256",
                  "golden_output_sha256", "checkpoint_sha256",
                  "output_validation"):
        if baseline[field] != progress[field]:
            raise ValueError(f"progress build changed {field}")
    if progress["progress_console_header_sha256"] != sha(
            Path(__file__).resolve().parents[2] / "lib/include/VX_config.h"):
        raise ValueError("console header changed")
    chunks = sorted(old.glob("stage_chunk_*.cpp"))
    if len(chunks) != 92:
        raise ValueError("expected 92 stage source chunks")
    for source in chunks:
        if sha(source) != sha(new / source.name):
            raise ValueError(f"progress build changed {source.name}")
    for name in ("model_data.cpp", "host.cpp"):
        if sha(old / name) != sha(new / name):
            raise ValueError(f"progress build changed {name}")
    old_kernel = (old / "kernel.cpp").read_text()
    new_kernel = (new / "kernel.cpp").read_text()
    if new_kernel.count("#include <VX_config.h>\n") != 1:
        raise ValueError("missing console header")
    helper = re.search(
        r"void smolvla_putchar\(uint32_t value\) \{\n.*?\n\}\n"
        r"void smolvla_progress\(uint32_t stage\) \{\n.*?\n\}\n",
        new_kernel, flags=re.DOTALL)
    if helper is None or "*(volatile __global uint8_t*)IO_COUT_ADDR" not in helper.group():
        raise ValueError("progress helper does not use the console MMIO")
    schedule = new_kernel.split("\nint main() {\n", 1)[1].split(
        "\n  if (mu_hart_id() != 0)", 1)[0]
    completed = 0
    found = []
    for line in schedule.splitlines():
        if line.startswith("  mu_schedule(stage_"):
            completed += 1
        marker = re.fullmatch(r"  smolvla_progress\((\d+)u\);", line)
        if marker:
            value = int(marker.group(1))
            if value != completed:
                raise ValueError(f"marker {value} precedes stage {completed}")
            found.append(value)
    expected = [0, *range(10, 3674, 10), 3673]
    if completed != 3673 or found != expected:
        raise ValueError("stage markers do not follow the expected schedule")
    stripped = new_kernel.replace("#include <VX_config.h>\n", "", 1)
    stripped = stripped.replace(helper.group(), "", 1)
    stripped = re.sub(r"(?m)^  smolvla_progress\(\d+u\);\n", "", stripped)
    if stripped != old_kernel:
        raise ValueError("kernel changed beyond progress instrumentation")
    for manifest, root in ((baseline, old), (progress, new)):
        if sha(root / "kernel.radiance.elf") != manifest["radiance_elf_sha256"]:
            raise ValueError(f"ELF differs from build manifest: {root}")
    return {
        "schema": "radiance.smolvla-full-progress-build.v1",
        "status": "verified_build",
        "model": "smolvla_base",
        "stages": 3673,
        "scheduled_warps_per_core": 4,
        "source_chunks_identical": len(chunks),
        "progress_interval_stages": 10,
        "progress_marker_count": len(found),
        "first_progress_marker": found[0],
        "last_progress_marker": found[-1],
        "source_spec_sha256": progress["source_spec_sha256"],
        "execution_schedule_sha256": progress["execution_schedule_sha256"],
        "weight_image_sha256": progress["weight_image_sha256"],
        "input_image_sha256": progress["input_image_sha256"],
        "golden_output_sha256": progress["golden_output_sha256"],
        "baseline_elf_sha256": baseline["radiance_elf_sha256"],
        "progress_elf_sha256": progress["radiance_elf_sha256"],
        "progress_build_log_sha256": sha(new / "build.log"),
        "progress_manifest_sha256": sha(new / "manifest.json"),
        "full_action_chunk_device_execution": False,
        "rtl_execution": False,
        "performance_measurement": False,
        "source_sha256": sha(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--progress-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.baseline_root, args.progress_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
