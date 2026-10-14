#!/usr/bin/env python3
"""Pack full TinyLlama or DeepSeek checkpoints for the 2 GiB GPU map.

Linear and embedding matrices use the existing per-channel INT8 kernels.
Norms and biases stay FP32. This conversion needs an output comparison with
the original checkpoint before the image is used as an evaluation workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from export_decoder_weights import (ALIGNMENT, SafeTensorWeights,
                                    checkpoint_sha256, parameter_specs,
                                    sha256_file)
from split_decoder_weights import STACK_BOTTOM, STACK_TOP, verify_image
from stitch import build, model_specs
from checkpoint import quantize_per_channel


MODELS = ("tinyllama", "deepseek_r1_distill_qwen_1_5b")
BASE = 0x14000000
GPU_SIZE = 0x80000000


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def plan(model: str, layers: int, base: int = BASE) -> dict:
    if model not in MODELS:
        raise ValueError(f"unsupported untied decoder model: {model}")
    spec = dict(model_specs()[model])
    if not 1 <= layers <= spec["num_hidden_layers"]:
        raise ValueError("layer count exceeds pinned model depth")
    if base < 0 or base % 4096 or base >= STACK_BOTTOM:
        raise ValueError("invalid page-aligned checkpoint base")
    spec["num_hidden_layers"] = layers
    graph = build(model, prefill=1, decode_steps=1, specs={model: spec})
    parameters, uses = parameter_specs(graph)
    size = 0
    for item in parameters:
        operation = item["operation"]
        if operation not in ("embedding", "linear", "rmsnorm", "bias_add"):
            raise ValueError(f"{item['logical_name']}: unsupported parameter operation")
        storage = "fp32" if operation in ("rmsnorm", "bias_add") else "int8_scaled"
        weight_bytes = math.prod(item["shape"]) * (4 if storage == "fp32" else 1)
        scale_count = (item["shape"][0] if operation == "embedding" else
                       item["shape"][1] if operation == "linear" else 0)
        relative_scale = align(weight_bytes) if scale_count else None
        length = relative_scale + 4 * scale_count if scale_count else weight_bytes
        offset = align(size)
        item.update(storage_dtype=storage, weight_size_bytes=weight_bytes,
                    scale_count=scale_count, size_bytes=length,
                    segment_index=0, offset_bytes=offset,
                    gpu_address=base + offset,
                    scale_offset_bytes=offset + relative_scale
                    if scale_count else None,
                    scale_gpu_address=base + offset + relative_scale
                    if scale_count else None,
                    overlaps_warp_stacks=False)
        size = offset + length
    if base + size > STACK_BOTTOM or base + size > GPU_SIZE:
        raise ValueError(f"{model}: INT8 checkpoint exceeds the 2 GiB GPU region")
    return {
        "model": model, "family": spec["family"], "layers": layers,
        "pinned_model_layers": model_specs()[model]["num_hidden_layers"],
        "graph_stages": len(graph.stages), "parameter_uses": uses,
        "distinct_parameters": len(parameters), "dtype": "int8_scaled",
        "linear_layout": "row_major_in_out", "alignment_bytes": ALIGNMENT,
        "gpu_base_address": base, "gpu_end_address_exclusive": base + size,
        "image_size_bytes": size, "fits_32_bit_address_space": True,
        "warp_stack_reserved_range": [STACK_BOTTOM, STACK_TOP],
        "stack_overlap_parameters": [],
        "segments": [{"index": 0, "image_file": "weights.bin",
                      "gpu_base_address": base, "image_size_bytes": size}],
        "parameters": parameters, "checkpoint_weights": False,
        "device_execution": False,
    }


def export(layout: dict, checkpoint_dir: Path, output_dir: Path) -> dict:
    model = layout["model"]
    spec = model_specs()[model]
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "model.safetensors"
    if sha256_file(config_path) != spec["source_sha256"]:
        raise ValueError("checkpoint config differs from pinned revision")
    weights_hash = checkpoint_sha256(model)
    if sha256_file(weights_path) != weights_hash:
        raise ValueError("checkpoint weights differ from pinned revision")
    provider = SafeTensorWeights(weights_path, spec["family"])
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "weights.bin"
    temporary = output_dir / "weights.bin.tmp"
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as stream:
            for item in layout["parameters"]:
                gap = item["offset_bytes"] - stream.tell()
                if gap < 0 or gap >= ALIGNMENT:
                    raise ValueError("invalid checkpoint image alignment")
                pad = bytes(gap)
                stream.write(pad)
                digest.update(pad)
                logical = item["logical_name"]
                shape = tuple(item["shape"])
                value = (provider.embedding_table(logical, *shape)
                         if item["operation"] == "embedding" else
                         provider(logical, shape))
                if item["gemma_weight_offset"]:
                    raise ValueError("Gemma offset is invalid for an untied decoder")
                if item["storage_dtype"] == "fp32":
                    packed = np.ascontiguousarray(value, dtype="<f4")
                    scales = None
                    metric = {"roundtrip_changed_elements": 0,
                              "roundtrip_max_abs_error": 0.0}
                else:
                    packed, scales, metric = quantize_per_channel(
                        value, item["operation"])
                raw = memoryview(np.ascontiguousarray(packed)).cast("B")
                if len(raw) != item["weight_size_bytes"]:
                    raise ValueError(f"{logical}: packed size differs from plan")
                stream.write(raw)
                digest.update(raw)
                parameter_hash = hashlib.sha256(raw)
                if scales is not None:
                    gap = item["scale_offset_bytes"] - stream.tell()
                    if gap < 0 or gap >= ALIGNMENT:
                        raise ValueError("invalid scale alignment")
                    pad = bytes(gap)
                    stream.write(pad)
                    digest.update(pad)
                    parameter_hash.update(pad)
                    scale_raw = memoryview(np.ascontiguousarray(scales)).cast("B")
                    stream.write(scale_raw)
                    digest.update(scale_raw)
                    parameter_hash.update(scale_raw)
                item.update(metric)
                item["checkpoint_key"] = provider.key(logical)
                item["packed_sha256"] = parameter_hash.hexdigest()
            if stream.tell() != layout["image_size_bytes"]:
                raise ValueError("checkpoint image size differs from plan")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    result = dict(layout)
    result["segments"][0]["image_sha256"] = digest.hexdigest()
    result.update(checkpoint_config_sha256=spec["source_sha256"],
                  checkpoint_weight_sha256=weights_hash,
                  image_sha256=digest.hexdigest(), checkpoint_weights=True,
                  roundtrip_changed_elements=sum(x["roundtrip_changed_elements"]
                                                 for x in layout["parameters"]),
                  roundtrip_max_abs_error=max(x["roundtrip_max_abs_error"]
                                              for x in layout["parameters"]))
    manifest = output_dir / "weights-image.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    verify_image(manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    layout = plan(args.model, args.layers)
    if args.plan_only:
        print(json.dumps({key: value for key, value in layout.items()
                          if key != "parameters"}, indent=2))
        return
    if args.checkpoint_dir is None:
        parser.error("--checkpoint-dir is required for export")
    result = export(layout, args.checkpoint_dir, args.out_dir / args.model)
    print(f"{args.model}: {result['distinct_parameters']} parameters, "
          f"{result['image_size_bytes']} bytes, SHA-256 {result['image_sha256']}")


if __name__ == "__main__":
    main()
