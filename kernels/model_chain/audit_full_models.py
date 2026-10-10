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
MULTITOKEN_RECORDS = {
    "tinyllama": ("tinyllama-full-depth-fp16-multitoken-build.json",
                  "tinyllama-full-depth-fp16-multitoken-native-results.json"),
    "deepseek_r1_distill_qwen_1_5b": (
        "deepseek-full-depth-fp16-multitoken-build.json",
        "deepseek-full-depth-fp16-multitoken-native-results.json"),
    "gemma_2_2b_it": (
        "gemma-full-depth-int8-fp16-tied-multitoken-build.json",
        "gemma-full-depth-int8-fp16-tied-multitoken-native-results.json"),
}
ONE_LAYER_MULTITOKEN_RECORDS = {
    "tinyllama": ("tinyllama-one-layer-multitoken-build.json",
                  "tinyllama-one-layer-multitoken-reference.json",
                  "tinyllama-one-layer-multitoken-native-results.json"),
    "deepseek_r1_distill_qwen_1_5b": (
        "deepseek-one-layer-multitoken-build.json",
        "deepseek-one-layer-multitoken-reference.json",
        "deepseek-one-layer-multitoken-native-results.json"),
    "gemma_2_2b_it": (
        "gemma-one-layer-int8-fp16-tied-multitoken-build.json",
        "gemma-checkpoint-one-layer-reference.json",
        "gemma-one-layer-int8-fp16-tied-multitoken-native-results.json"),
}
ONE_LAYER_MULTITOKEN_DEVICE_RECORDS = {
    "tinyllama": "tinyllama-one-layer-multitoken-functional-results.json",
    "deepseek_r1_distill_qwen_1_5b":
        "deepseek-one-layer-multitoken-functional-results.json",
    "gemma_2_2b_it": "gemma-one-layer-multitoken-functional-results.json",
}
UPSTREAM_TOKEN_IDS = {"prefill.token_ids": [1, 2, 3],
                      "decode0.token_ids": [4], "decode1.token_ids": [5]}


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
                spec["num_hidden_layers"] and
                control["prefill_tokens"] == 3 and
                control["cached_decode_tokens"] == 2 and shards == manifest["checkpoint_sha256"],
                "Gemma full-depth upstream reference differs from the ELF")
        return {"path": str(path),
                "maximum_absolute_error": control["maximum_absolute_error"],
                "precision": "unquantized_checkpoint_reference"}
    path = ROOT / "kernels/evaluation/llm/checkpoint-results.json"
    controls = json.loads(path.read_text())["checks"]
    matches = [item for item in controls if item["model"] == model and
               item["layers_checked"] == spec["num_hidden_layers"]]
    require(len(matches) == 1 and matches[0]["passed"] and
            matches[0]["prefill_tokens"] == 3 and
            matches[0]["cached_decode_tokens"] == 2 and
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
    probe = None
    euler_chain_probe = None
    expert_attention_probe = None
    expert_linear_probe = None
    if model == "smolvla_base":
        probe_path = HERE / "evaluation/smolvla-final-euler-functional-results.json"
        if probe_path.exists():
            candidate = json.loads(probe_path.read_text())
            probe_target = generated_root / "smolvla-final-euler-probe/smolvla_base"
            require(candidate["status"] == "passed" and candidate["tohost"] == 0 and
                    candidate["zero_output_rejected_elements"] > 0 and
                    candidate["source_full_elf_sha256"] ==
                    manifest["radiance_elf_sha256"] and
                    candidate["source_stage_chunk_sha256"] ==
                    sha256(target / "stage_chunk_091.cpp") and
                    candidate["upstream_output_sha256"] ==
                    manifest["golden_output_sha256"] and
                    candidate["device_elf_sha256"] ==
                    sha256(probe_target / "kernel.radiance.elf") and
                    candidate["log_sha256"] ==
                    sha256(probe_target / "functional.log"),
                    "SmolVLA last-stage device probe is stale or incomplete")
            probe = {"status": "passed", "stage_id": candidate["stage_id"],
                     "cycles_functional": candidate["cycles_functional"],
                     "path": str(probe_path)}
        chain_path = HERE / "evaluation/smolvla-euler-chain-functional-results.json"
        if chain_path.exists():
            candidate = json.loads(chain_path.read_text())
            chain_target = generated_root / "smolvla-euler-chain-probe/smolvla_base"
            expected_ids = [f"denoise{iteration}.euler" for iteration in range(10)]
            expected_indices = [index + 1 for index, stage in enumerate(graph.stages)
                                if stage["id"] in expected_ids]
            require(candidate["status"] == "passed" and
                    candidate["tohost"] == 0 and
                    candidate["process_exit_code"] == 0 and
                    candidate["stage_ids"] == expected_ids and
                    candidate["stage_indices_one_based"] == expected_indices and
                    len(candidate["zero_output_rejected_elements_per_iteration"]) == 10 and
                    all(count > 0 for count in candidate[
                        "zero_output_rejected_elements_per_iteration"]) and
                    candidate["source_full_elf_sha256"] ==
                    manifest["radiance_elf_sha256"] and
                    candidate["input_image_sha256"] ==
                    manifest["input_image_sha256"] and
                    candidate["upstream_output_sha256"] ==
                    manifest["golden_output_sha256"] and
                    all(sha256(target / name) == digest and
                        sha256(chain_target / name) == digest for name, digest in
                        candidate["source_stage_chunks_sha256"].items()) and
                    candidate["device_elf_sha256"] ==
                    sha256(chain_target / "kernel.radiance.elf") and
                    candidate["log_sha256"] ==
                    sha256(chain_target / "functional.log"),
                    "SmolVLA ten-step Euler device probe is stale or incomplete")
            euler_chain_probe = {"status": "passed", "stages": 10,
                                 "cycles_functional": candidate["cycles_functional"],
                                 "path": str(chain_path)}
            negative_path = (HERE / "evaluation" /
                             "smolvla-euler-chain-negative-control.json")
            if negative_path.exists():
                negative = json.loads(negative_path.read_text())
                negative_target = (generated_root /
                                   "smolvla-euler-chain-probe-negative/smolvla_base")
                require(negative["status"] == "failed" and
                        negative["failure_reason"] == "device_output_mismatch" and
                        negative["tohost"] == 65537 and
                        negative["mutation"] ==
                        "first_expected_action_replaced_by_1000000.0f" and
                        negative["positive_probe_elf_sha256"] ==
                        candidate["device_elf_sha256"] and
                        negative["device_elf_sha256"] ==
                        sha256(negative_target / "kernel.radiance.elf") and
                        negative["mutated_probe_data_sha256"] ==
                        sha256(negative_target / "probe_data.cpp") and
                        negative["log_sha256"] ==
                        sha256(negative_target / "functional.log"),
                        "SmolVLA Euler negative control is stale or incomplete")
                euler_chain_probe["negative_control"] = "passed"
        attention_path = (HERE / "evaluation" /
                          "smolvla-expert-attention-functional-results.json")
        if attention_path.exists():
            import numpy as np
            from probe_smolvla_attention import attention_reference

            candidate = json.loads(attention_path.read_text())
            attention_target = (generated_root /
                                "smolvla-attention-probe/smolvla_base")
            fixture_root = generated_root / "smolvla-attention-fixture"
            fixture = json.loads((fixture_root / "fixture-result.json").read_text())
            traces = fixture_root / "traces"
            read_names = ["denoise0.expert00.q_rope",
                          "denoise0.expert00.k_append",
                          "denoise0.expert00.v_append",
                          "denoise0.attention_mask"]
            read_indices = [932, 936, 937, 927]
            read_paths = [traces / f"{index:04d}.bin" for index in read_indices]
            require(candidate["status"] == "passed" and
                    candidate["tohost"] == 0 and
                    candidate["process_exit_code"] == 0 and
                    candidate["stage_id"] == "denoise0.expert00.attention" and
                    candidate["stage_index_one_based"] == 938 and
                    candidate["zero_output_rejected_elements"] > 0 and
                    candidate["source_full_elf_sha256"] ==
                    manifest["radiance_elf_sha256"] and
                    candidate["source_native_result_sha256"] ==
                    sha256(native_path) and
                    fixture["status"] == "passed" and
                    fixture["stage_count"] == 938 and
                    fixture["generated_source_sha256"] ==
                    native["generated_source_sha256"] and
                    fixture["native_binary_sha256"] ==
                    candidate["fixture_native_binary_sha256"] and
                    fixture["log_sha256"] ==
                    candidate["fixture_native_log_sha256"] and
                    sha256(Path(fixture["log_path"])) ==
                    fixture["log_sha256"] and
                    sha256(fixture_root /
                           "smolvla_base/native/smolvla_native") ==
                    fixture["native_binary_sha256"] and
                    all(candidate["fixture_input_sha256"][name] == sha256(path)
                        for name, path in zip(read_names, read_paths)) and
                    candidate["fixture_native_output_sha256"] ==
                    sha256(traces / "0938.bin") and
                    candidate["source_stage_chunk_sha256"] ==
                    sha256(target / candidate["source_stage_chunk_name"]) ==
                    sha256(attention_target / candidate[
                        "source_stage_chunk_name"]) and
                    candidate["device_elf_sha256"] ==
                    sha256(attention_target / "kernel.radiance.elf") and
                    candidate["log_sha256"] ==
                    sha256(attention_target / "functional.log"),
                    "SmolVLA expert attention probe is stale or incomplete")
            q = np.fromfile(read_paths[0], dtype="<f4").reshape(50, 15, 64)
            k = np.fromfile(read_paths[1], dtype="<f4").reshape(291, 5, 64)
            v = np.fromfile(read_paths[2], dtype="<f4").reshape(291, 5, 64)
            mask = np.fromfile(read_paths[3], dtype="<u4").reshape(50, 291)
            expected = attention_reference(q, k, v, mask)
            native_output = np.fromfile(traces / "0938.bin", dtype="<f4")
            error = float(np.max(np.abs(expected - native_output)))
            require(np.isclose(error, candidate[
                        "independent_numpy_max_abs_error_vs_native"],
                        rtol=1e-6, atol=1e-10) and
                    np.allclose(expected, native_output, rtol=5e-3, atol=5e-4),
                    "SmolVLA expert attention reference cannot be reproduced")
            expert_attention_probe = {
                "status": "passed", "stage_id": candidate["stage_id"],
                "cycles_functional": candidate["cycles_functional"],
                "independent_numpy_max_abs_error_vs_native": error,
                "path": str(attention_path)}
            negative_path = (HERE / "evaluation" /
                             "smolvla-expert-attention-negative-control.json")
            if negative_path.exists():
                negative = json.loads(negative_path.read_text())
                negative_target = (generated_root /
                                   "smolvla-attention-probe-negative/smolvla_base")
                require(negative["status"] == "failed" and
                        negative["failure_reason"] == "device_output_mismatch" and
                        negative["tohost"] == 1 and
                        negative["mutation"] ==
                        "first_expected_attention_value_replaced_by_1000000.0f" and
                        negative["positive_probe_elf_sha256"] ==
                        candidate["device_elf_sha256"] and
                        negative["device_elf_sha256"] ==
                        sha256(negative_target / "kernel.radiance.elf") and
                        negative["mutated_probe_data_sha256"] ==
                        sha256(negative_target / "probe_data.cpp") and
                        negative["log_sha256"] ==
                        sha256(negative_target / "functional.log"),
                        "SmolVLA attention negative control is stale or incomplete")
                expert_attention_probe["negative_control"] = "passed"
        linear_path = (HERE / "evaluation" /
                       "smolvla-expert-linear-functional-results.json")
        if linear_path.exists():
            import numpy as np

            candidate = json.loads(linear_path.read_text())
            linear_target = generated_root / "smolvla-linear-probe/smolvla_base"
            linear_manifest = json.loads((linear_target / "manifest.json").read_text())
            fixture_root = generated_root / "smolvla-linear-fixture"
            fixture = json.loads((fixture_root / "fixture-result.json").read_text())
            image_parameter = next(
                item for item in image["parameters"]
                if item["logical_name"] ==
                graph.stages[930]["attrs"]["parameter"])
            segment = image["segments"][image_parameter["segment_index"]]
            weight_path = (Path(manifest["weight_image_manifest"]).parent /
                           segment["image_file"])
            with weight_path.open("rb") as stream:
                stream.seek(image_parameter["offset_bytes"])
                packed = stream.read(image_parameter["size_bytes"])
            input_path = fixture_root / "traces/0930.bin"
            output_path = fixture_root / "traces/0931.bin"
            require(candidate["status"] == "passed" and
                    candidate["tohost"] == 0 and
                    candidate["process_exit_code"] == 0 and
                    candidate["stage_output_comparison"] and
                    candidate["output_elements"] == 48000 and
                    candidate["stage_id"] == "denoise0.expert00.q_proj" and
                    linear_manifest["zero_output_rejected_elements"] > 0 and
                    candidate["source_full_elf_sha256"] ==
                    manifest["radiance_elf_sha256"] and
                    candidate["source_stage_chunk_sha256"] ==
                    sha256(target / "stage_chunk_023.cpp") ==
                    sha256(linear_target / "stage_chunk_023.cpp") and
                    candidate["weight_image_sha256"] ==
                    manifest["weight_image_sha256"] and
                    candidate["packed_parameter_sha256"] ==
                    image_parameter["packed_sha256"] ==
                    hashlib.sha256(packed).hexdigest() and
                    candidate["packed_parameter_gpu_address"] ==
                    image_parameter["gpu_address"] and
                    fixture["status"] == "passed" and
                    fixture["stage_count"] == 931 and
                    fixture["device_elf_sha256"] ==
                    manifest["radiance_elf_sha256"] and
                    fixture["generated_source_sha256"] ==
                    native["generated_source_sha256"] and
                    sha256(Path(fixture["log_path"])) ==
                    fixture["log_sha256"] and
                    sha256(fixture_root /
                           "smolvla_base/native/smolvla_native") ==
                    fixture["native_binary_sha256"] and
                    candidate["fixture_input_sha256"] ==
                    sha256(input_path) and
                    linear_manifest["fixture_native_output_sha256"] ==
                    sha256(output_path) and
                    linear_manifest["radiance_elf_sha256"] ==
                    candidate["device_elf_sha256"] ==
                    sha256(linear_target / "kernel.radiance.elf") and
                    candidate["log_sha256"] ==
                    sha256(linear_target / "functional.log"),
                    "SmolVLA weighted expert projection probe is stale")
            x = np.fromfile(input_path, dtype="<f4").reshape(50, 720)
            w = np.frombuffer(packed, dtype="<f4").reshape(720, 960)
            expected = (x @ w).ravel()
            observed = np.fromfile(output_path, dtype="<f4")
            error = float(np.max(np.abs(expected - observed)))
            require(observed.size == 48000 and
                    np.allclose(expected, observed, rtol=5e-3, atol=5e-4) and
                    np.isclose(error, candidate[
                        "independent_numpy_max_abs_error_vs_native"],
                        rtol=0.25, atol=1e-6),
                    "SmolVLA expert projection reference differs")
            expert_linear_probe = {
                "status": "passed", "stage_id": candidate["stage_id"],
                "cycles_functional": candidate["cycles_functional"],
                "independent_numpy_max_abs_error_vs_native": error,
                "path": str(linear_path)}
            negative_path = (HERE / "evaluation" /
                             "smolvla-expert-linear-negative-control.json")
            if negative_path.exists():
                negative = json.loads(negative_path.read_text())
                negative_target = (generated_root /
                                   "smolvla-linear-probe-negative/smolvla_base")
                require(negative["status"] == "failed" and
                        negative["failure_reason"] ==
                        "device_output_mismatch_or_nonfinite" and
                        negative["tohost"] == 1 and
                        negative["mutation"] ==
                        "first_expected_projection_value_replaced_by_1000000.0f" and
                        negative["positive_probe_elf_sha256"] ==
                        candidate["device_elf_sha256"] and
                        negative["device_elf_sha256"] ==
                        sha256(negative_target / "kernel.radiance.elf") and
                        negative["mutated_probe_data_sha256"] ==
                        sha256(negative_target / "probe_data.cpp") and
                        negative["log_sha256"] ==
                        sha256(negative_target / "functional.log"),
                        "SmolVLA projection negative control is stale")
                expert_linear_probe["negative_control"] = "passed"
    return {
        "model": model, "graph_stages": len(graph.stages),
        "radiance_elf_sha256": manifest["radiance_elf_sha256"],
        "checkpoint_weight_format": manifest["checkpoint_weight_format"],
        "checkpoint_image_sha256": image["image_sha256"],
        "upstream_reference": reference,
        "full_depth_host_check": {"status": "passed", "path": str(native_path)},
        "full_depth_device_check": device or {"status": "pending"},
        "targeted_device_probe": probe,
        "euler_chain_device_probe": euler_chain_probe,
        "expert_attention_device_probe": expert_attention_probe,
        "expert_linear_device_probe": expert_linear_probe,
    }


def audit_multitoken(model: str, generated_root: Path) -> dict:
    build_name, native_name = MULTITOKEN_RECORDS[model]
    target = (generated_root / "checkpoint-multitoken-upstream-elf" / model).resolve()
    build_path = HERE / "evaluation" / build_name
    native_path = HERE / "evaluation" / native_name
    if not (build_path.exists() and native_path.exists()):
        return {"status": "pending"}
    manifest = json.loads((target / "manifest.json").read_text())
    tracked_build = json.loads(build_path.read_text())
    graph = build(model, prefill=3, decode_steps=2)
    require(manifest == tracked_build and manifest["model"] == model and
            manifest["full_model_dimensions"] and
            manifest["checkpoint_weights"] and
            manifest["device_elf_built"] and
            manifest["layers"] == model_specs()[model]["num_hidden_layers"] and
            manifest["prefill_tokens"] == 3 and
            manifest["decode_steps"] == 2 and
            manifest["input_token_ids"] == UPSTREAM_TOKEN_IDS and
            manifest["token_sequence_explicit"] and
            manifest["stages"] == len(graph.stages) and
            manifest["device_check_all_stages"] and
            len(manifest["verified_tensors"]) == manifest["stages"],
            f"{model}: upstream-token full-depth ELF is incomplete")
    base_dir = BUILDS[model][0]
    base_manifest = json.loads((generated_root / base_dir / model /
                                "manifest.json").read_text())
    require(manifest["weight_image_sha256"] ==
            base_manifest["weight_image_sha256"] and
            manifest["checkpoint_sha256"] ==
            base_manifest["checkpoint_sha256"],
            f"{model}: upstream-token ELF uses different checkpoint weights")
    elf = target / "kernel.radiance.elf"
    require(sha256(elf) == manifest["radiance_elf_sha256"],
            f"{model}: upstream-token ELF differs from manifest")
    image = json.loads(Path(manifest["weight_image_manifest"]).read_text())
    verify_image_placement(elf, [image])
    native = json.loads(native_path.read_text())
    sources = native["device_source_files_sha256"]
    require(native["status"] == "passed" and
            native["process_exit_code"] == 0 and
            native["layers"] == manifest["layers"] and
            native["stage_count"] == manifest["stages"] and
            native["checked_float_stages"] +
            native["checked_integer_stages"] == manifest["stages"] and
            native["input_token_ids"] == UPSTREAM_TOKEN_IDS and
            native["device_elf_sha256"] == manifest["radiance_elf_sha256"] and
            native["weight_image_sha256"] == manifest["weight_image_sha256"] and
            native["reference_output_sha256"] ==
            manifest["reference_output_sha256"] and
            set(sources) == set(manifest["device_source_files"]) and
            all(sha256(target / name) == digest
                for name, digest in sources.items()) and
            sha256(Path(native["log_path"])) == native["log_sha256"],
            f"{model}: upstream-token host check is stale or incomplete")
    return {"status": "passed", "layers": manifest["layers"],
            "stage_count": manifest["stages"],
            "input_token_ids": manifest["input_token_ids"],
            "radiance_elf_sha256": manifest["radiance_elf_sha256"],
            "checkpoint_weight_format": manifest["checkpoint_weight_format"],
            "quantization_comparison": manifest["quantization_comparison"],
            "build_record": str(build_path), "host_result": str(native_path),
            "device_execution": False}


def audit_one_layer_multitoken(model: str, generated_root: Path) -> dict:
    build_name, reference_name, native_name = ONE_LAYER_MULTITOKEN_RECORDS[model]
    target = (generated_root / "checkpoint-one-layer-multitoken-upstream-elf" /
              model).resolve()
    build_path = HERE / "evaluation" / build_name
    reference_path = HERE / "evaluation" / reference_name
    native_path = HERE / "evaluation" / native_name
    if not (build_path.exists() and reference_path.exists() and
            native_path.exists()):
        return {"status": "pending"}
    manifest = json.loads((target / "manifest.json").read_text())
    tracked = json.loads(build_path.read_text())
    spec = dict(model_specs()[model])
    spec["num_hidden_layers"] = 1
    graph = build(model, prefill=3, decode_steps=2, specs={model: spec})
    require(manifest == tracked and manifest["model"] == model and
            manifest["layers"] == 1 and manifest["prefill_tokens"] == 3 and
            manifest["decode_steps"] == 2 and
            manifest["input_token_ids"] == UPSTREAM_TOKEN_IDS and
            manifest["token_sequence_explicit"] and
            manifest["full_model_dimensions"] and
            manifest["checkpoint_weights"] and
            manifest["device_elf_built"] and
            manifest["device_check_all_stages"] and
            manifest["stages"] == len(graph.stages) and
            len(manifest["verified_tensors"]) == manifest["stages"],
            f"{model}: one-layer upstream-token ELF is incomplete")
    elf = target / "kernel.radiance.elf"
    require(sha256(elf) == manifest["radiance_elf_sha256"],
            f"{model}: one-layer upstream-token ELF differs from manifest")
    image, _ = verify_image(Path(manifest["weight_image_manifest"]))
    require(image["layers"] == 1 and
            image["image_sha256"] == manifest["weight_image_sha256"] and
            image["checkpoint_weight_sha256"] == manifest["checkpoint_sha256"],
            f"{model}: one-layer checkpoint image differs from ELF")
    verify_image_placement(elf, [image])
    reference = json.loads(reference_path.read_text())
    reference_hash = (hashlib.sha256(json.dumps(
        reference["checkpoint_files_sha256"], sort_keys=True).encode()).hexdigest()
        if model == "gemma_2_2b_it" else reference["checkpoint_sha256"])
    require(reference["passed"] and reference["layers_checked"] == 1 and
            reference["prefill_tokens"] == 3 and
            reference["cached_decode_tokens"] == 2 and
            reference_hash == manifest["checkpoint_sha256"],
            f"{model}: one-layer upstream reference differs from ELF")
    native = json.loads(native_path.read_text())
    sources = native["device_source_files_sha256"]
    require(native["scope"] == "partial_checkpoint_generated_cpp_on_cpu" and
            native["status"] == "passed" and
            native["process_exit_code"] == 0 and
            native["layers"] == 1 and
            native["stage_count"] == manifest["stages"] and
            native["checked_float_stages"] +
            native["checked_integer_stages"] == manifest["stages"] and
            native["input_token_ids"] == UPSTREAM_TOKEN_IDS and
            native["device_elf_sha256"] == manifest["radiance_elf_sha256"] and
            native["weight_image_sha256"] == manifest["weight_image_sha256"] and
            native["reference_output_sha256"] ==
            manifest["reference_output_sha256"] and
            set(sources) == set(manifest["device_source_files"]) and
            all(sha256(target / name) == digest
                for name, digest in sources.items()) and
            sha256(Path(native["log_path"])) == native["log_sha256"],
            f"{model}: one-layer generated C++ check is stale or incomplete")
    device_path = target / "functional-result.json"
    device = {"status": "pending"}
    if device_path.exists():
        candidate = json.loads(device_path.read_text())
        tracked_device_path = (HERE / "evaluation" /
                               ONE_LAYER_MULTITOKEN_DEVICE_RECORDS[model])
        if tracked_device_path.exists():
            require(json.loads(tracked_device_path.read_text()) == candidate,
                    f"{model}: tracked one-layer device result differs from local run")
        require(candidate["device_elf_sha256"] ==
                manifest["radiance_elf_sha256"] and
                candidate["weight_image_sha256"] ==
                manifest["weight_image_sha256"] and
                candidate["stage_count"] == manifest["stages"] and
                sha256(target / "functional.log") == candidate["log_sha256"],
                f"{model}: one-layer device result is stale")
        if candidate["status"] == "passed":
            require(candidate["tohost"] == 0 and
                    candidate["process_exit_code"] == 0 and
                    candidate["checked_float_stages"] +
                    candidate["checked_integer_stages"] == manifest["stages"] and
                    candidate["reference_output_sha256"] ==
                    manifest["reference_output_sha256"],
                    f"{model}: one-layer device result omits stage checks")
        device = {"status": candidate["status"],
                  "cycles_functional": candidate["cycles_functional"],
                  "failure_reason": candidate["failure_reason"],
                  "path": str(device_path)}
    return {"status": "passed", "stage_count": manifest["stages"],
            "input_token_ids": manifest["input_token_ids"],
            "checkpoint_weight_format": manifest["checkpoint_weight_format"],
            "radiance_elf_sha256": manifest["radiance_elf_sha256"],
            "upstream_reference": str(reference_path),
            "build_record": str(build_path),
            "host_check": {"status": "passed", "path": str(native_path)},
            "device_check": device}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", type=Path,
                        default=HERE / "generated")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    entries = [audit(model, args.generated_root) for model in BUILDS]
    for entry in entries:
        if entry["model"] in MULTITOKEN_RECORDS:
            entry["upstream_token_sequence_build"] = audit_multitoken(
                entry["model"], args.generated_root)
            entry["upstream_token_one_layer_device_control"] = (
                audit_one_layer_multitoken(entry["model"], args.generated_root))
    summary = {"scope": "four_full_checkpoint_stitched_radiance_models",
               "all_elfs_built_and_host_checked": True,
               "all_full_depth_device_checks_passed": all(
                   item["full_depth_device_check"]["status"] == "passed"
                   for item in entries),
               "all_upstream_token_sequence_builds_host_checked": all(
                   item["upstream_token_sequence_build"]["status"] == "passed"
                   for item in entries if item["model"] in MULTITOKEN_RECORDS),
               "all_upstream_token_one_layer_device_controls_passed": all(
                   item["upstream_token_one_layer_device_control"]["status"] ==
                   "passed" and item[
                       "upstream_token_one_layer_device_control"][
                           "device_check"]["status"] == "passed"
                   for item in entries if item["model"] in
                   ONE_LAYER_MULTITOKEN_RECORDS),
               "all_upstream_token_one_layer_host_controls_passed": all(
                   item["upstream_token_one_layer_device_control"]["status"] ==
                   "passed" and item[
                       "upstream_token_one_layer_device_control"][
                           "host_check"]["status"] == "passed"
                   for item in entries if item["model"] in
                   ONE_LAYER_MULTITOKEN_RECORDS),
               "models": entries, "rtl_execution": False,
               "performance_measurement": False}
    output = json.dumps(summary, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
