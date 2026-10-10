#!/usr/bin/env python3
"""Audit four full-checkpoint Radiance builds and their independent controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from split_decoder_weights import verify_image, verify_image_placement


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernels/evaluation/llm"))
from stitch import build, model_specs  # noqa: E402


BUILDS = {
    "tinyllama": ("checkpoint-fp16-full-depth-elf",
                  "tinyllama-full-depth-fp16-native-results.json"),
    "deepseek_r1_distill_qwen_1_5b": (
        "checkpoint-fp16-full-depth-elf",
        "deepseek-full-depth-fp16-native-results.json"),
    "gemma_2_2b_it": (
        "checkpoint-int8-fp16-tied-full-depth-elf",
        "gemma-full-depth-int8-fp16-tied-native-results.json"),
    "smolvla_base": ("checkpoint-smolvla-exact-golden-elf",
                     "smolvla-full-fp32-native-results.json"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def upstream_reference(model: str, manifest: dict) -> dict:
    spec = model_specs()[model]
    if model == "smolvla_base":
        path = ROOT / "kernels/evaluation/llm/smolvla-policy-fp32-results.json"
        control = json.loads(path.read_text())
        require(control["passed"] and control["weights_promoted_to_fp32"] and
                control["graph_order_and_shapes_match_upstream"] and
                control["checkpoint_weight_sha256"] ==
                manifest["checkpoint_sha256"] and
                control["output_sha256"] == manifest["golden_output_sha256"] and
                control["observed_layer_calls"] ==
                {"vision": 36, "vlm": 16, "expert": 160},
                "SmolVLA FP32 upstream control differs from the ELF")
        return {"path": str(path), "output_sha256": control["output_sha256"],
                "precision": "checkpoint_weights_promoted_fp32"}
    if model == "gemma_2_2b_it":
        path = HERE / "evaluation/gemma-checkpoint-full-reference.json"
        control = json.loads(path.read_text())
        shards = hashlib.sha256(json.dumps(
            control["checkpoint_files_sha256"], sort_keys=True).encode()).hexdigest()
        require(control["passed"] and control["layers_checked"] ==
                spec["num_hidden_layers"] and shards == manifest["checkpoint_sha256"],
                "Gemma full-depth upstream reference differs from the ELF")
        return {"path": str(path),
                "maximum_absolute_error": control["maximum_absolute_error"],
                "precision": "unquantized_checkpoint_reference"}
    path = ROOT / "kernels/evaluation/llm/checkpoint-results.json"
    controls = json.loads(path.read_text())["checks"]
    matches = [item for item in controls if item["model"] == model and
               item["layers_checked"] == spec["num_hidden_layers"]]
    require(len(matches) == 1 and matches[0]["passed"] and
            matches[0]["checkpoint_sha256"] == manifest["checkpoint_sha256"],
            f"{model}: full-depth upstream reference differs from the ELF")
    return {"path": str(path),
            "maximum_absolute_error": matches[0]["maximum_absolute_error"],
            "precision": "unquantized_checkpoint_reference"}


def audit(model: str, generated_root: Path) -> dict:
    directory, native_name = BUILDS[model]
    target = (generated_root / directory / model).resolve()
    manifest = json.loads((target / "manifest.json").read_text())
    graph = build(model, prefill=1, decode_steps=1) if model != "smolvla_base" else build(model)
    require(manifest["model"] == model and manifest["device_elf_built"] and
            manifest["stages"] == len(graph.stages),
            f"{model}: ELF does not contain the full graph")
    if model == "smolvla_base":
        require(manifest["full_graph_stages"] == manifest["stages"] and
                sum(stage["op"] == "euler_step" for stage in graph.stages) == 10 and
                manifest["output_validation"] == "all_elements_vs_upstream_policy",
                "SmolVLA action loop or output check is incomplete")
    else:
        require(manifest["checkpoint_weights"] and
                manifest["layers"] == model_specs()[model]["num_hidden_layers"] and
                manifest["prefill_tokens"] == 1 and
                manifest["decode_steps"] == 1 and
                manifest["device_check_all_stages"] and
                len(manifest["verified_tensors"]) == manifest["stages"],
                f"{model}: full-depth prefill/decode or stage checks are incomplete")
    elf = target / "kernel.radiance.elf"
    require(sha256(elf) == manifest["radiance_elf_sha256"],
            f"{model}: device ELF differs from its build manifest")
    image, _ = verify_image(Path(manifest["weight_image_manifest"]))
    require(image["checkpoint_weights"] and
            image["image_sha256"] == manifest["weight_image_sha256"] and
            image["checkpoint_weight_sha256"] == manifest["checkpoint_sha256"],
            f"{model}: checkpoint image differs from its ELF")
    images = [image]
    if model == "smolvla_base":
        inputs, _ = verify_image(Path(manifest["input_image_manifest"]))
        require(inputs["image_sha256"] == manifest["input_image_sha256"],
                "SmolVLA exact input image differs from its ELF")
        images.append(inputs)
    verify_image_placement(elf, images)
    reference = upstream_reference(model, manifest)
    native_path = HERE / "evaluation" / native_name
    native = json.loads(native_path.read_text())
    require(native["status"] == "passed" and
            native["stage_count"] == manifest["stages"] and
            native["device_elf_sha256"] == manifest["radiance_elf_sha256"] and
            sha256(Path(native["log_path"])) == native["log_sha256"],
            f"{model}: full-depth host check is missing or stale")
    if model == "smolvla_base":
        source_digest = hashlib.sha256("".join(
            sha256(target / name) for name in manifest["device_source_files"]
        ).encode()).hexdigest()
        require(native["output_elements"] == 1600 and
                native["failed_output_elements"] == 0 and
                native["golden_output_sha256"] == manifest["golden_output_sha256"] and
                native["generated_source_sha256"] == source_digest and
                native["comparison_tolerance"] == manifest["output_check_tolerance"],
                "SmolVLA host action check does not cover the upstream output")
    else:
        sources = native["device_source_files_sha256"]
        require(native["layers"] == manifest["layers"] and
                native["checked_float_stages"] +
                native["checked_integer_stages"] == manifest["stages"] and
                native["weight_image_sha256"] == manifest["weight_image_sha256"] and
                set(sources) == set(manifest["device_source_files"]) and
                all(sha256(target / name) == digest
                    for name, digest in sources.items()) and
                native["reference_output_sha256"] == manifest[
                    "reference_output_sha256"],
                f"{model}: host check does not cover every mapped stage")
    device_path = target / "functional-result.json"
    device = None
    if device_path.exists():
        candidate = json.loads(device_path.read_text())
        require(candidate["device_elf_sha256"] == manifest["radiance_elf_sha256"] and
                candidate["stage_count"] == manifest["stages"] and
                candidate["weight_image_sha256"] == manifest["weight_image_sha256"],
                f"{model}: device result refers to a different ELF")
        require(sha256(target / "functional.log") == candidate["log_sha256"],
                f"{model}: device result log differs from its record")
        if candidate["status"] == "passed":
            require(candidate["tohost"] == 0 and candidate["process_exit_code"] == 0,
                    f"{model}: device result claims pass without successful execution")
            if model == "smolvla_base":
                require(candidate["golden_output_comparison"] and
                        candidate["golden_output_sha256"] ==
                        manifest["golden_output_sha256"] and
                        candidate["input_image_sha256"] ==
                        manifest["input_image_sha256"],
                        "SmolVLA device result omits the exact-input output check")
            else:
                require(candidate["checked_float_stages"] +
                        candidate["checked_integer_stages"] == manifest["stages"] and
                        candidate["reference_output_sha256"] ==
                        manifest["reference_output_sha256"],
                        f"{model}: device result omits all-stage validation")
        device = {"status": candidate["status"], "path": str(device_path),
                  "cycles_functional": candidate["cycles_functional"],
                  "failure_reason": candidate.get("failure_reason")}
    return {
        "model": model, "graph_stages": len(graph.stages),
        "radiance_elf_sha256": manifest["radiance_elf_sha256"],
        "checkpoint_weight_format": manifest["checkpoint_weight_format"],
        "checkpoint_image_sha256": image["image_sha256"],
        "upstream_reference": reference,
        "full_depth_host_check": {"status": "passed", "path": str(native_path)},
        "full_depth_device_check": device or {"status": "pending"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", type=Path,
                        default=HERE / "generated")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    entries = [audit(model, args.generated_root) for model in BUILDS]
    summary = {"scope": "four_full_checkpoint_stitched_radiance_models",
               "all_elfs_built_and_host_checked": True,
               "all_full_depth_device_checks_passed": all(
                   item["full_depth_device_check"]["status"] == "passed"
                   for item in entries),
               "models": entries, "rtl_execution": False,
               "performance_measurement": False}
    output = json.dumps(summary, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
