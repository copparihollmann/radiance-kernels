#!/usr/bin/env python3
"""Verify exact-input SmolVLA vision-prefix runs on both Radiance cores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from split_decoder_weights import image_segments, verify_image, verify_image_placement


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case(root: Path, rebuilt_root: Path, stages: int,
         simulator_sha256: str) -> tuple[dict, dict]:
    target = root / "smolvla_base"
    rebuilt = rebuilt_root / "smolvla_base"
    manifest_path = target / "manifest.json"
    result_path = target / "functional-result.json"
    log_path = target / "functional.log"
    elf_path = target / "kernel.radiance.elf"
    manifest = json.loads(manifest_path.read_text())
    result = json.loads(result_path.read_text())
    log = log_path.read_text()
    if (manifest["stages"] != stages or result["stage_count"] != stages
            or manifest["scope"] != "full_dimension_checkpoint_stage_probe"
            or manifest["scheduled_warps_per_stage"] != 4
            or manifest["use_all_cores"] is not True
            or manifest["output_validation"] != "all_elements_vs_stage_golden"
            or result["status"] != "passed" or result["tohost"] != 0
            or result["process_exit_code"] != 0
            or result["golden_output_comparison"] is not True
            or result["simulator_sha256"] != simulator_sha256
            or result["rtl_execution"] or result["performance_measurement"]):
        raise ValueError(f"{stages}-stage result is not a passing two-core check")
    if (sha(elf_path) != manifest["radiance_elf_sha256"]
            or sha(elf_path) != result["device_elf_sha256"]
            or sha(log_path) != result["log_sha256"]
            or log.count("isa-test passed with tohost=0") != 1
            or re.findall(r"simulation finished after (\d+) cycles", log)
            != [str(result["cycles_functional"])]):
        raise ValueError(f"{stages}-stage ELF or completion log differs")
    for name in ("kernel.cpp", "model_data.cpp", "stage_chunk_000.cpp",
                 "host.cpp"):
        if sha(target / name) != sha(rebuilt / name):
            raise ValueError(f"{stages}-stage source does not regenerate: {name}")
    kernel = (target / "kernel.cpp").read_text()
    stage_source = (target / "stage_chunk_000.cpp").read_text()
    data = (target / "model_data.cpp").read_text()
    if (kernel.count("RAD_L0D_DRAIN(") != 1
            or stage_source.count("RAD_L0D_DRAIN(") != stages
            or "if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return" in stage_source
            or "__mu_num_warps = 4;" not in kernel
            or "v_attention_scratch[MU_NUM_THREADS * 4 * MU_NUM_CORES * 1024]" not in data
            or "for (uint32_t j = 0; j < 786432u; ++j)" not in kernel
            or "mu_close(output[j], golden[j], 1e-3f, 1e-3f)" not in kernel):
        raise ValueError(f"{stages}-stage source lacks the expected two-core check")
    images = []
    for field in ("weight", "input", "stage_golden"):
        image, paths = verify_image(Path(manifest[field + "_image_manifest"]))
        digest = (manifest["golden_output_sha256"] if field == "stage_golden"
                  else manifest[field + "_image_sha256"])
        if image["image_sha256"] != digest or result[
                "golden_output_sha256" if field == "stage_golden"
                else field + "_image_sha256"] != digest:
            raise ValueError(f"{stages}-stage {field} image differs")
        for segment in image_segments(image):
            start = segment["gpu_base_address"]
            end = start + segment["image_size_bytes"]
            if start < 0x1F202000 and end > 0x1F200000:
                raise ValueError("preload overlaps the cache-drain scratch window")
            marker = (f"preloaded {segment['image_size_bytes']} bytes of weights "
                      f"@0x{start:x}")
            if marker not in log:
                raise ValueError(f"{stages}-stage preload is absent from log")
        if not paths:
            raise ValueError(f"{stages}-stage {field} image is empty")
        images.append((image, paths))
    verify_image_placement(elf_path, [image for image, _ in images])
    return ({
        "stages": stages,
        "status": result["status"],
        "cycles_functional": result["cycles_functional"],
        "tohost": result["tohost"],
        "elf_sha256": sha(elf_path),
        "manifest_sha256": sha(manifest_path),
        "result_sha256": sha(result_path),
        "log_sha256": sha(log_path),
        "kernel_source_sha256": sha(target / "kernel.cpp"),
        "stage_source_sha256": sha(target / "stage_chunk_000.cpp"),
        "weight_image_sha256": images[0][0]["image_sha256"],
        "input_image_sha256": images[1][0]["image_sha256"],
        "golden_image_sha256": images[2][0]["image_sha256"],
    }, {"manifest": manifest, "images": images})


def same_parameter(one: tuple[dict, list[Path]],
                   two: tuple[dict, list[Path]], name: str) -> bool:
    digests = []
    for image, paths in (one, two):
        item = next(x for x in image["parameters"] if x["logical_name"] == name)
        segment = image_segments(image)[item["segment_index"]]
        with paths[item["segment_index"]].open("rb") as stream:
            stream.seek(item["gpu_address"] - segment["gpu_base_address"])
            digests.append(hashlib.sha256(stream.read(item["size_bytes"])).hexdigest())
    return digests[0] == digests[1]


def verify(stage1_root: Path, stage2_root: Path,
           stage1_rebuilt: Path, stage2_rebuilt: Path,
           stage2_golden_recheck: Path,
           simulator: Path) -> dict:
    simulator_hash = sha(simulator)
    first, first_data = case(stage1_root, stage1_rebuilt, 1, simulator_hash)
    second, second_data = case(stage2_root, stage2_rebuilt, 2, simulator_hash)
    prior = json.loads((HERE / "evaluation/smolvla-stage1-warp-control.json").read_text())
    baseline = next(item for item in prior["cases"] if item["warps"] == 4)
    if (first["weight_image_sha256"] != baseline["weight_image_sha256"]
            or first["input_image_sha256"] != baseline["input_image_sha256"]
            or first["golden_image_sha256"] != baseline["golden_output_sha256"]
            or second["input_image_sha256"] != first["input_image_sha256"]):
        raise ValueError("two-core probes differ from the passing input or golden")
    reference = second_data["images"][2][0]
    original_golden = Path(second_data["manifest"]["stage_golden_image_manifest"])
    for filename in ("stage2.bin", "stage-golden-image.json"):
        if sha(original_golden.parent / filename) != sha(
                stage2_golden_recheck / filename):
            raise ValueError("second-stage PyTorch golden did not regenerate")
    if (reference["reference_stage"] != "camera0.vision.layer00.norm1"
            or reference["source_stage1_image_sha256"] != first["golden_image_sha256"]
            or reference["source_weight_image_sha256"] != second["weight_image_sha256"]
            or reference["epsilon"] != 1e-6
            or reference["reference_operation"] != "torch.nn.functional.layer_norm"):
        raise ValueError("second-stage golden does not bind to the pinned prefix")
    compact = first_data["images"][0]
    full = second_data["images"][0]
    names = [item["logical_name"] for item in compact[0]["parameters"]]
    if len(names) != 3 or any(not same_parameter(compact, full, name)
                              for name in names):
        raise ValueError("patch parameters differ between compact and full images")
    return {
        "schema": "radiance.smolvla-all-cores-prefix.v1",
        "status": "passed",
        "scope": "exact-input first two vision stages against pinned PyTorch values",
        "checked_output_elements_per_case": 786432,
        "stage1_single_core_four_warp_cycles": baseline["cycles_functional"],
        "stage1_two_core_four_warp_cycles": first["cycles_functional"],
        "stage1_functional_cycle_reduction_percent": 100 * (
            1 - first["cycles_functional"] / baseline["cycles_functional"]),
        "stage2_two_core_four_warp_cycles": second["cycles_functional"],
        "stage2_golden_generator_sha256": sha(HERE / "generate_smolvla_stage2_golden.py"),
        "stage2_golden_recomputed_identically": True,
        "cache_drain_header_sha256": sha(ROOT / "lib/include/rad_tapeout.h"),
        "simulator_sha256": simulator_hash,
        "cases": [first, second],
        "full_action_chunk_device_execution": False,
        "rtl_execution": False,
        "performance_measurement": False,
        "source_sha256": sha(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--stage1-rebuilt", type=Path, required=True)
    parser.add_argument("--stage2-rebuilt", type=Path, required=True)
    parser.add_argument("--stage2-golden-recheck", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.stage1_root, args.stage2_root,
                    args.stage1_rebuilt, args.stage2_rebuilt,
                    args.stage2_golden_recheck, args.simulator)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
