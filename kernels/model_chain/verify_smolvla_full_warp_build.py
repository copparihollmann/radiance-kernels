#!/usr/bin/env python3
"""Check that the four-warp SmolVLA ELF preserves the pinned action graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from split_decoder_weights import verify_image, verify_image_placement


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(baseline_root: Path, four_root: Path) -> dict:
    one = baseline_root / "smolvla_base"
    four = four_root / "smolvla_base"
    old = json.loads((one / "manifest.json").read_text())
    new = json.loads((four / "manifest.json").read_text())
    if (old["stages"] != 3673 or new["stages"] != 3673
            or old["scope"] != "full_checkpoint_action_chunk"
            or new["scope"] != "full_checkpoint_action_chunk"
            or new["scheduled_warps_per_stage"] != 4
            or old["output_validation"] != new["output_validation"]
            or new["output_validation"] != "all_elements_vs_upstream_policy"
            or not old["device_elf_built"] or not new["device_elf_built"]):
        raise ValueError("expected full 3,673-stage checkpoint action builds")
    for field in ("source_spec_sha256", "execution_schedule_sha256",
                  "weight_image_sha256", "input_image_sha256",
                  "golden_output_sha256", "checkpoint_sha256"):
        if old[field] != new[field]:
            raise ValueError(f"four-warp build changed {field}")
    old_elf = one / "kernel.radiance.elf"
    new_elf = four / "kernel.radiance.elf"
    if (sha(old_elf) != old["radiance_elf_sha256"]
            or sha(new_elf) != new["radiance_elf_sha256"]):
        raise ValueError("device ELF differs from build manifest")
    chunks = sorted(four.glob("stage_chunk_*.cpp"))
    old_chunks = sorted(one.glob("stage_chunk_*.cpp"))
    if len(chunks) != 92 or [p.name for p in chunks] != [p.name for p in old_chunks]:
        raise ValueError("the 92 stage source chunks differ")
    for baseline, candidate in zip(old_chunks, chunks):
        if sha(baseline) != sha(candidate):
            raise ValueError(f"stage source changed: {candidate.name}")
    original_data = (one / "model_data.cpp").read_text()
    expected_data = original_data.replace(
        "v_attention_scratch[MU_NUM_THREADS * 1024]",
        "v_attention_scratch[MU_NUM_THREADS * 4 * 1024]")
    if expected_data == original_data or expected_data != (four / "model_data.cpp").read_text():
        raise ValueError("model data changed beyond attention scratch size")
    original_kernel = (one / "kernel.cpp").read_text()
    expected_kernel = original_kernel.replace(
        "__mu_num_warps = 1;", "__mu_num_warps = 4;").replace(
            "nullptr, 1);", "nullptr, 4);")
    expected_kernel = expected_kernel.replace(
        "  for (uint32_t j = 0; j < 1600u; ++j) {\n",
        "  const __global float* golden = v_golden_output;\n"
        "  for (uint32_t j = 0; j < 1600u; ++j) {\n").replace(
            "mu_close(output[j], v_golden_output[j], 1e-3f, 1e-3f)",
            "mu_close(output[j], golden[j], 1e-3f, 1e-3f)")
    if (len(re.findall(r"mu_schedule\([^\n]+, nullptr, 4\);", expected_kernel))
            != 3674 or expected_kernel != (four / "kernel.cpp").read_text()):
        raise ValueError("kernel schedule changed beyond warp count")
    images = []
    for field in ("weight", "input"):
        image, _ = verify_image(Path(new[field + "_image_manifest"]))
        if image["image_sha256"] != new[field + "_image_sha256"]:
            raise ValueError(f"{field} image differs from build manifest")
        images.append(image)
    verify_image_placement(new_elf, images)
    return {
        "schema": "radiance.smolvla-four-warp-full-build.v1",
        "status": "verified_build",
        "model": "smolvla_base",
        "scope": "full_checkpoint_action_chunk",
        "stages": 3673,
        "scheduled_warps_per_stage": 4,
        "identical_stage_source_chunks": len(chunks),
        "output_check_alias_normalized": True,
        "source_spec_sha256": new["source_spec_sha256"],
        "execution_schedule_sha256": new["execution_schedule_sha256"],
        "weight_image_sha256": new["weight_image_sha256"],
        "input_image_sha256": new["input_image_sha256"],
        "golden_output_sha256": new["golden_output_sha256"],
        "baseline_elf_sha256": sha(old_elf),
        "four_warp_elf_sha256": sha(new_elf),
        "baseline_manifest_sha256": sha(one / "manifest.json"),
        "four_warp_manifest_sha256": sha(four / "manifest.json"),
        "four_warp_build_log_sha256": sha(four / "build.log"),
        "full_model_device_execution": False,
        "rtl_execution": False,
        "performance_measurement": False,
        "source_sha256": sha(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--four-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.baseline_root, args.four_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
