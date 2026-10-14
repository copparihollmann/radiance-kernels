#!/usr/bin/env python3
"""Verify a saved connected Cyclotron run against its retained build artifacts."""

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
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def same(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f"{label} differs or is missing: {path}")


def verify(result_path: Path, simulator: Path, config: Path,
           source_config: Path | None = None) -> dict:
    result = json.loads(result_path.read_text())
    target = result_path.parent
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (result["status"] != "passed" or result["tohost"] != 0
            or result["process_exit_code"] != 0
            or result["rtl_execution"] or result["performance_measurement"]
            or result["simulation"] != "Cyclotron functional ISA model"
            or not manifest["device_check_all_stages"]
            or not manifest["device_elf_built"]):
        raise ValueError("receipt is not a complete passing functional device run")
    stages = (manifest["native_verified_float_stages"]
              + manifest.get("native_verified_integer_stages", 0))
    if (result["stage_count"] != manifest["stages"] or stages != manifest["stages"]
            or result["checked_float_stages"] !=
            manifest["native_verified_float_stages"]
            or result["checked_integer_stages"] !=
            manifest.get("native_verified_integer_stages", 0)
            or len(manifest["verified_tensors"]) != stages
            or result["layers"] != manifest["layers"]
            or result["prefill_tokens"] != manifest["prefill_tokens"]
            or result["decode_steps"] != manifest["decode_steps"]):
        raise ValueError("receipt does not cover every compiled stage")

    elf = target / "kernel.radiance.elf"
    same(elf, result["device_elf_sha256"], "device ELF")
    if result["device_elf_sha256"] != manifest["radiance_elf_sha256"]:
        raise ValueError("ELF and build manifest hashes disagree")
    sources = result["device_source_files_sha256"]
    if set(sources) != set(manifest["device_source_files"]):
        raise ValueError("device source file inventory is incomplete")
    for name, expected in sources.items():
        same(target / name, expected, name)
    if result["generated_source_sha256"] != sources["kernel.cpp"]:
        raise ValueError("device entrypoint hash disagrees with source inventory")
    same(HERE / "pipeline_math.hpp", result["portable_ops_sha256"],
         "portable device operations")
    for key, name in (("quantized_ops_sha256", "pipeline_int8.hpp"),
                      ("moe_ops_sha256", "pipeline_moe.hpp"),
                      ("tied_embedding_ops_sha256", "pipeline_tied.hpp"),
                      ("fp32_tied_ops_sha256", "pipeline_fp32_tied.hpp"),
                      ("packed_ops_sha256", "pipeline_packed.hpp")):
        if result.get(key):
            same(HERE / name, result[key], name)
    same(ROOT / "kernels/evaluation/llm/reference.py",
         result["numpy_reference_sha256"], "NumPy reference")
    for key, name in (("numpy_moe_graph_reference_sha256", "moe_graph_reference.py"),
                      ("numpy_moe_stage_ops_sha256", "moe_ops.py"),
                      ("graph_builder_sha256", "stitch.py")):
        if result.get(key):
            same(ROOT / "kernels/evaluation/llm" / name, result[key], name)
    if result["model"] == "qwen3_30b_a3b" and any(not result.get(key) for key in (
            "moe_ops_sha256", "numpy_moe_graph_reference_sha256",
            "numpy_moe_stage_ops_sha256", "graph_builder_sha256")):
        raise ValueError("Qwen3 MoE receipt lacks sparse-stage source hashes")
    same(ROOT / "kernels/evaluation/llm/pr1-models.json",
         result["model_specs_sha256"], "model specs")
    if source_config is not None:
        same(source_config, result["source_config_sha256"], "pinned config")
    same(simulator, result["simulator_sha256"], "Cyclotron executable")
    same(config, result["config_sha256"], "Cyclotron config")
    log_path = target / "functional.log"
    same(log_path, result["log_sha256"], "Cyclotron completion log")
    log = log_path.read_text()
    cycles = re.findall(r"simulation finished after (\d+) cycles", log)
    if (cycles != [str(result["cycles_functional"])]
            or log.count("isa-test passed with tohost=0") != 1):
        raise ValueError("completion log lacks unique matching cycle/tohost markers")

    image_checked = False
    if result["checkpoint_weights"]:
        image, _ = verify_image(Path(manifest["weight_image_manifest"]))
        verify_image_placement(elf, [image])
        segments = image_segments(image)
        layout = [{key: segment[key] for key in
                   ("gpu_base_address", "image_size_bytes", "image_sha256")}
                  for segment in segments]
        if (image["image_sha256"] != result["weight_image_sha256"]
                or layout != result["weight_segment_layout"]
                or result["checkpoint_sha256"] != manifest["checkpoint_sha256"]
                or result["reference_output_sha256"] !=
                manifest["reference_output_sha256"]):
            raise ValueError("checkpoint image or reference differs from receipt")
        for segment in segments:
            marker = (f"preloaded {segment['image_size_bytes']} bytes of weights "
                      f"@0x{segment['gpu_base_address']:x}")
            if marker not in log:
                raise ValueError("Cyclotron log lacks expected weight preload")
        image_checked = True

    return {
        "schema": "radiance.connected-functional-verification.v1",
        "status": "passed", "model": result["model"],
        "layers": result["layers"], "prefill_tokens": result["prefill_tokens"],
        "decode_steps": result["decode_steps"],
        "checked_stages": stages,
        "cycles_functional": result["cycles_functional"],
        "weight_image_checked": image_checked,
        "rtl_execution": False, "performance_measurement": False,
        "result_sha256": sha(result_path),
        "manifest_sha256": sha(manifest_path),
        "elf_sha256": sha(elf),
        "log_sha256": sha(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.result, args.simulator, args.config,
                    args.source_config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['model']}: verified {report['checked_stages']} stages, "
          f"{report['cycles_functional']} functional cycles")


if __name__ == "__main__":
    main()
