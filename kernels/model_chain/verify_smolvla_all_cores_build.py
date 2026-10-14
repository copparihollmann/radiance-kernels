#!/usr/bin/env python3
"""Verify the full SmolVLA two-core ELF against the four-warp core-0 build."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from profile_smolvla_simulator import scalar_work
from split_decoder_weights import image_segments, verify_image, verify_image_placement


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE0_MAP = (
    "  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return;\n"
    "  tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS\n"
    "      + tid % MU_NUM_THREADS;\n"
    "  tpb /= MU_NUM_CORES;\n"
)
INIT_CORE0_MAP = (
    "  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return;\n"
    "  tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS\n"
    "      + tid % MU_NUM_THREADS; tpb /= MU_NUM_CORES;\n"
)
DRAIN = "  RAD_L0D_DRAIN(tid, tpb / MU_NUM_THREADS, 1, 2);\n"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(core0_root: Path, all_cores_root: Path,
           diagnostic_root: Path, simulator: Path) -> dict:
    one = core0_root / "smolvla_base"
    both = all_cores_root / "smolvla_base"
    original = json.loads((one / "manifest.json").read_text())
    candidate = json.loads((both / "manifest.json").read_text())
    if (original["stages"] != 3673 or candidate["stages"] != 3673
            or original["scheduled_warps_per_stage"] != 4
            or candidate["scheduled_warps_per_stage"] != 4
            or candidate["use_all_cores"] is not True
            or original["scope"] != candidate["scope"]
            or candidate["scope"] != "full_checkpoint_action_chunk"
            or original["output_validation"] != candidate["output_validation"]
            or candidate["output_validation"] != "all_elements_vs_upstream_policy"
            or not original["device_elf_built"] or not candidate["device_elf_built"]):
        raise ValueError("builds are not matching full checkpoint action graphs")
    for key in ("source_spec_sha256", "execution_schedule_sha256",
                "weight_image_sha256", "input_image_sha256",
                "golden_output_sha256", "checkpoint_sha256"):
        if original[key] != candidate[key]:
            raise ValueError(f"two-core build changed {key}")
    drain_header = ROOT / "lib/include/rad_tapeout.h"
    if candidate["cache_drain_header_sha256"] != sha(drain_header):
        raise ValueError("L0D drain source differs from build manifest")
    old_elf = one / "kernel.radiance.elf"
    new_elf = both / "kernel.radiance.elf"
    if (sha(old_elf) != original["radiance_elf_sha256"]
            or sha(new_elf) != candidate["radiance_elf_sha256"]):
        raise ValueError("device ELF differs from build manifest")
    images = []
    for field in ("weight", "input"):
        image, _ = verify_image(Path(candidate[field + "_image_manifest"]))
        if image["image_sha256"] != candidate[field + "_image_sha256"]:
            raise ValueError(f"{field} image differs from build manifest")
        images.append(image)
    scratch = {"segments": [{"gpu_base_address": 0x1F200000,
                             "image_size_bytes": 0x2000}]}
    verify_image_placement(new_elf, [*images, scratch])

    source_chunks = sorted(one.glob("stage_chunk_*.cpp"))
    new_chunks = sorted(both.glob("stage_chunk_*.cpp"))
    if len(source_chunks) != 92 or [p.name for p in source_chunks] != [
            p.name for p in new_chunks]:
        raise ValueError("stage source chunk inventory changed")
    drains = 0
    for baseline, updated in zip(source_chunks, new_chunks):
        source = baseline.read_text()
        expected = source.replace(
            '#include "pipeline_smolvla.hpp"\n',
            '#include "pipeline_smolvla.hpp"\n#include "rad_tapeout.h"\n')
        expected = expected.replace(CORE0_MAP, "")
        expected, count = re.subn(r"(?m)^}\n(?=\nvoid stage_|$)",
                                  DRAIN + "}\n", expected)
        if not count or expected != updated.read_text():
            raise ValueError(f"stage source changed beyond core mapping: {updated.name}")
        drains += count
    if drains != 3673:
        raise ValueError("two-core source has incomplete stage drains")
    old_data = (one / "model_data.cpp").read_text()
    expected_data = old_data.replace(
        "v_attention_scratch[MU_NUM_THREADS * 4 * 1024]",
        "v_attention_scratch[MU_NUM_THREADS * 4 * MU_NUM_CORES * 1024]")
    if expected_data == old_data or expected_data != (both / "model_data.cpp").read_text():
        raise ValueError("model data changed beyond attention scratch capacity")
    old_kernel = (one / "kernel.cpp").read_text()
    expected_kernel = old_kernel.replace(
        '#include "kernel_verify.h"\n',
        '#include "kernel_verify.h"\n#include "rad_tapeout.h"\n')
    expected_kernel = expected_kernel.replace(INIT_CORE0_MAP, "")
    expected_kernel = expected_kernel.replace(
        "}\n\nint main() {", DRAIN + "}\n\nint main() {")
    if expected_kernel != (both / "kernel.cpp").read_text():
        raise ValueError("main schedule changed beyond core mapping")
    diagnostic_path = diagnostic_root / "result.json"
    published_diagnostic = HERE / "evaluation/smolvla-full-two-core-diagnostic.json"
    if sha(diagnostic_path) != sha(published_diagnostic):
        raise ValueError("published diagnostic differs from the raw result")
    diagnostic = json.loads(diagnostic_path.read_text())
    probe = diagnostic["probe"]
    config = diagnostic_root / "smolvla-4000000-cycles.toml"
    log = diagnostic_root / "smolvla-4000000-cycles.log"
    text = log.read_text()
    for image in images:
        for segment in image_segments(image):
            marker = (f"preloaded {segment['image_size_bytes']} bytes of weights "
                      f"@0x{segment['gpu_base_address']:x}")
            if marker not in text:
                raise ValueError("bounded launch log lacks a verified preload")
    if (diagnostic["model"] != "smolvla_base"
            or diagnostic["mapping"] != scalar_work(both, 3673)
            or probe["scope"] != "bounded_cyclotron_diagnostic_not_model_completion"
            or probe["cycle_cap"] != 4000000
            or probe["status"] != "cycle_cap_reached"
            or probe["exit_code"] != 1
            or probe["all_images_preloaded"] is not True
            or probe["full_model_output_checked"] is not False
            or probe["performance_measurement"] is not False
            or probe["elf_sha256"] != sha(new_elf)
            or probe["weight_image_sha256"] != candidate["weight_image_sha256"]
            or probe["input_image_sha256"] != candidate["input_image_sha256"]
            or probe["simulator_sha256"] != sha(simulator)
            or probe["config_sha256"] != sha(config)
            or probe["log_sha256"] != sha(log)
            or 'timeout = 4000000' not in config.read_text()
            or not text.rstrip().endswith("Error: 0")
            or "simulation finished after" in text):
        raise ValueError("bounded two-core Cyclotron launch differs")
    return {
        "schema": "radiance.smolvla-full-two-core-build.v1",
        "status": "verified_build",
        "model": "smolvla_base",
        "stages": 3673,
        "scheduled_warps_per_core": 4,
        "stage_source_chunks": len(source_chunks),
        "stage_drains": drains,
        "source_spec_sha256": candidate["source_spec_sha256"],
        "execution_schedule_sha256": candidate["execution_schedule_sha256"],
        "weight_image_sha256": candidate["weight_image_sha256"],
        "input_image_sha256": candidate["input_image_sha256"],
        "golden_output_sha256": candidate["golden_output_sha256"],
        "core0_elf_sha256": sha(old_elf),
        "two_core_elf_sha256": sha(new_elf),
        "two_core_manifest_sha256": sha(both / "manifest.json"),
        "two_core_build_log_sha256": sha(both / "build.log"),
        "cache_drain_header_sha256": sha(drain_header),
        "bounded_launch_cycle_cap": probe["cycle_cap"],
        "bounded_launch_status": probe["status"],
        "bounded_launch_receipt_sha256": sha(diagnostic_path),
        "full_model_device_execution": False,
        "rtl_execution": False,
        "performance_measurement": False,
        "source_sha256": sha(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core0-root", type=Path, required=True)
    parser.add_argument("--all-cores-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.core0_root, args.all_cores_root,
                    args.diagnostic_root, args.simulator)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
