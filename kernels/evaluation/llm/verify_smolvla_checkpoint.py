#!/usr/bin/env python3
"""Check SmolVLA graph parameter bindings against a real safetensors header.

This checks names, shapes, and weight reuse without loading 865 MB of data.
It does not run the checkpoint or establish numerical equivalence.
"""

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

from stitch import SPECS, build, model_specs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safetensors_header(path: Path) -> dict:
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > path.stat().st_size - 8:
            raise ValueError("safetensors header exceeds file size")
        header = json.loads(stream.read(size))
    header.pop("__metadata__", None)
    if any("shape" not in entry or "dtype" not in entry or
           "data_offsets" not in entry for entry in header.values()):
        raise ValueError("incomplete safetensors header")
    return header


def check(checkpoint_dir: Path) -> dict:
    spec = model_specs()["smolvla_base"]
    config_path = checkpoint_dir / "config.json"
    weight_path = checkpoint_dir / "model.safetensors"
    if sha256(config_path) != spec["source_sha256"]:
        raise ValueError("checkpoint config differs from pinned SmolVLA config")
    if sha256(weight_path) != spec["checkpoint_weight_sha256"]:
        raise ValueError("checkpoint weights differ from pinned SmolVLA checkpoint")
    header = safetensors_header(weight_path)
    graph = build("smolvla_base")
    explicit: dict[str, list[int]] = {}

    def bind(target: dict[str, list[int]], name: str, shape: list[int]) -> None:
        if name in target and target[name] != shape:
            raise ValueError(f"inconsistent graph shape for {name}")
        if name not in header:
            raise ValueError(f"missing checkpoint tensor {name}")
        if header[name]["shape"] != shape:
            raise ValueError(f"{name}: expected {shape}, got {header[name]['shape']}")
        target[name] = shape

    parameter_uses = 0
    for stage in graph.stages:
        attrs = stage["attrs"]
        for name, shape in attrs.get("parameters", {}).items():
            bind(explicit, name, shape)
            parameter_uses += 1
        name = attrs.get("parameter")
        if name is None:
            continue
        shape = attrs.get("checkpoint_shape")
        if shape is None:
            shape = [stage["shape"][-1]]
        bind(explicit, name, shape)
        parameter_uses += 1

    def expect_binding(name: str, shape: list[int]) -> None:
        if explicit.get(name) != shape:
            raise ValueError(f"graph omits checkpoint parameter {name}")

    vision = "model.vlm_with_expert.vlm.model.vision_model"
    embedding = f"{vision}.embeddings"
    for branch in graph.execution_schedule["prefix_once_per_refill"]["camera_branches"]:
        if len(branch["vision_layers"]) != spec["num_vision_layers"]:
            raise ValueError("vision loop does not match the pinned checkpoint")
    vh, vi = spec["vision_hidden_size"], spec["vision_intermediate_size"]
    patches = (spec["image_size"] // spec["patch_size"]) ** 2
    expect_binding(f"{embedding}.patch_embedding.weight",
                   [vh, 3, spec["patch_size"], spec["patch_size"]])
    expect_binding(f"{embedding}.patch_embedding.bias", [vh])
    expect_binding(f"{embedding}.position_embedding.weight", [patches, vh])
    for layer in range(spec["num_vision_layers"]):
        base = f"{vision}.encoder.layers.{layer}"
        for norm in ("layer_norm1", "layer_norm2"):
            for kind in ("weight", "bias"):
                expect_binding(f"{base}.{norm}.{kind}", [vh])
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            expect_binding(f"{base}.self_attn.{projection}.weight", [vh, vh])
            expect_binding(f"{base}.self_attn.{projection}.bias", [vh])
        expect_binding(f"{base}.mlp.fc1.weight", [vi, vh])
        expect_binding(f"{base}.mlp.fc1.bias", [vi])
        expect_binding(f"{base}.mlp.fc2.weight", [vh, vi])
        expect_binding(f"{base}.mlp.fc2.bias", [vh])
    for kind in ("weight", "bias"):
        expect_binding(f"{vision}.post_layernorm.{kind}", [vh])

    eh, action = spec["expert_hidden_size"], spec["action_dim"]
    for name, width in (("action_in_proj", action),
                        ("action_time_mlp_in", 2 * eh),
                        ("action_time_mlp_out", eh)):
        if explicit.get(f"model.{name}.weight") != [eh, width]:
            raise ValueError(f"action embedding weight is not bound: {name}")
        if explicit.get(f"model.{name}.bias") != [eh]:
            raise ValueError(f"action embedding bias is not bound: {name}")

    unused = {"model.vlm_with_expert.vlm.lm_head.weight"}
    if set(header) != set(explicit) | unused:
        missing = sorted(set(header) - set(explicit) - unused)
        raise ValueError(f"checkpoint weights without graph binding: {missing[:8]}")
    if header[next(iter(unused))]["shape"] != [spec["vlm_vocab_size"],
                                                  spec["vlm_hidden_size"]]:
        raise ValueError("unused language-model head has unexpected shape")
    stage_status = Counter(stage["status"] for stage in graph.stages)
    missing_ops = Counter(stage["op"] for stage in graph.stages
                          if stage["status"] == "missing_device_stage")
    return {
        "model": "smolvla_base", "check": "checkpoint_header_parameter_binding",
        "passed": True, "checkpoint_revision": spec["checkpoint_revision"],
        "checkpoint_config_sha256": sha256(config_path),
        "checkpoint_weight_sha256": sha256(weight_path),
        "model_specs_sha256": sha256(SPECS),
        "checkpoint_tensors": len(header),
        "explicit_stage_parameter_uses": parameter_uses,
        "distinct_explicit_parameters": len(explicit),
        "unbound_checkpoint_parameters": len(set(header) - set(explicit) - unused),
        "unused_lm_head_parameters": len(unused),
        "graph_stage_count": len(graph.stages),
        "stage_status": dict(stage_status),
        "missing_device_operations": dict(sorted(missing_ops.items())),
        "numerical_execution": False, "device_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = check(args.checkpoint_dir)
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
