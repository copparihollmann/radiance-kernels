#!/usr/bin/env python3
"""Bind one-, four-, and eight-warp SmolVLA stage-1 Cyclotron outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from split_decoder_weights import image_segments, verify_image, verify_image_placement


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_case(root: Path, warps: int) -> dict:
    target = root / "smolvla_base"
    manifest_path = target / "manifest.json"
    result_path = target / "functional-result.json"
    log_path = target / "functional.log"
    manifest = json.loads(manifest_path.read_text())
    result = json.loads(result_path.read_text())
    log = log_path.read_text()
    kernel_path = target / "kernel.cpp"
    kernel = kernel_path.read_text()
    elf_hash = sha(target / "kernel.radiance.elf")
    if (manifest["stages"] != 1 or result["stage_count"] != 1
            or manifest["scope"] != "full_dimension_checkpoint_stage_probe"):
        raise ValueError("expected one-stage SmolVLA checkpoint probe")
    if (elf_hash != manifest["radiance_elf_sha256"]
            or elf_hash != result["device_elf_sha256"]
            or sha(log_path) != result["log_sha256"]
            or manifest["output_validation"] != "all_elements_vs_stage_golden"
            or f'__mu_num_warps = {warps};' not in kernel
            or "for (uint32_t j = 0; j < 786432u; ++j)" not in kernel
            or "mu_close(output[j], golden[j], 1e-3f, 1e-3f)" not in kernel
            or len(re.findall(r"mu_schedule\([^\n]+, nullptr, "
                              + str(warps) + r"\);", kernel)) != 2):
        raise ValueError(f"{warps}-warp source, ELF, or log does not match")
    if (warps > 1 and manifest["scheduled_warps_per_stage"] != warps):
        raise ValueError("scheduled warp count differs from manifest")
    images = []
    for field in ("weight", "input", "stage_golden"):
        key = field + "_image_manifest"
        image, paths = verify_image(Path(manifest[key]))
        expected = (manifest["golden_output_sha256"] if field == "stage_golden"
                    else manifest[field + "_image_sha256"])
        if image["image_sha256"] != expected or not paths:
            raise ValueError(f"{field} image differs from manifest")
        images.append(image)
        for segment, path in zip(image_segments(image), paths):
            marker = (f"preloaded {segment['image_size_bytes']} bytes of weights "
                      f"@0x{segment['gpu_base_address']:x}")
            if marker not in log:
                raise ValueError(f"{field} was not preloaded")
    verify_image_placement(target / "kernel.radiance.elf", images)
    return {
        "warps": warps,
        "status": result["status"],
        "cycles_functional": result["cycles_functional"],
        "tohost": result["tohost"],
        "exit_code": result["process_exit_code"],
        "elf_sha256": elf_hash,
        "kernel_source_sha256": sha(kernel_path),
        "model_data_sha256": sha(target / "model_data.cpp"),
        "stage_source_sha256": sha(target / "stage_chunk_000.cpp"),
        "source_spec_sha256": manifest["source_spec_sha256"],
        "execution_schedule_sha256": manifest["execution_schedule_sha256"],
        "manifest_sha256": sha(manifest_path),
        "result_sha256": sha(result_path),
        "log_sha256": sha(log_path),
        "simulator_sha256": result["simulator_sha256"],
        "weight_image_sha256": result["weight_image_sha256"],
        "input_image_sha256": result["input_image_sha256"],
        "golden_output_sha256": result["golden_output_sha256"],
        "passed_marker": "isa-test passed with tohost=0" in log,
        "rename_limit_marker": "globalOverSubscription" in log,
        "phys_regs": (int(match.group(1)) if (match := re.search(
            r"num_phys_regs=(\d+)", log)) else None),
    }


def verify(one_root: Path, four_root: Path, eight_root: Path,
           default_rebuild_root: Path) -> dict:
    cases = [read_case(root, warps) for root, warps in
             ((one_root, 1), (four_root, 4), (eight_root, 8))]
    one, four, eight = cases
    for key in ("weight_image_sha256", "input_image_sha256",
                "golden_output_sha256", "simulator_sha256",
                "stage_source_sha256", "source_spec_sha256",
                "execution_schedule_sha256"):
        if len({case[key] for case in cases}) != 1:
            raise ValueError(f"probe inputs differ: {key}")
    if any(case["status"] != "passed" or case["tohost"] != 0
           or not case["passed_marker"] for case in (one, four)):
        raise ValueError("one- or four-warp golden check did not pass")
    if (eight["status"] != "failed" or eight["exit_code"] != 101
            or eight["tohost"] is not None or not eight["rename_limit_marker"]
            or eight["phys_regs"] != 256 or eight["passed_marker"]):
        raise ValueError("eight-warp run did not show the expected rename limit")
    prior = json.loads((HERE / "evaluation/smolvla-2g-stage1-exact-functional-results.json").read_text())
    if (one["elf_sha256"] != prior["device_elf_sha256"]
            or one["cycles_functional"] != prior["cycles_functional"]):
        raise ValueError("one-warp baseline differs from the public result")
    original = one_root / "smolvla_base"
    rebuilt = default_rebuild_root / "smolvla_base"
    for filename in ("kernel.cpp", "model_data.cpp", "stage_chunk_000.cpp",
                     "host.cpp"):
        if sha(original / filename) != sha(rebuilt / filename):
            raise ValueError(f"default one-warp source changed: {filename}")
    return {
        "schema": "radiance.smolvla-stage1-warp-control.v1",
        "status": "passed",
        "scope": "one checkpoint-backed vision patch stage with exact input and all 786432 upstream output values",
        "cases": cases,
        "four_vs_one_functional_cycle_ratio": four["cycles_functional"] / one["cycles_functional"],
        "one_vs_four_functional_cycle_reduction_percent": 100 * (
            1 - four["cycles_functional"] / one["cycles_functional"]),
        "eight_warp_failure": "Cyclotron current 256-entry global rename pool exhausted before output validation",
        "default_one_warp_source_unchanged": True,
        "full_model_device_result": False,
        "rtl_execution": False,
        "performance_measurement": False,
        "source_sha256": sha(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-root", type=Path, required=True)
    parser.add_argument("--four-root", type=Path, required=True)
    parser.add_argument("--eight-root", type=Path, required=True)
    parser.add_argument("--default-rebuild-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.one_root, args.four_root, args.eight_root,
                    args.default_rebuild_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
