#!/usr/bin/env python3
"""Export pinned decoder checkpoints as segmented FP16/FP32 Radiance images.

Embeddings and linear weights use IEEE FP16. Norm and bias parameters remain
FP32. This is a numerical conversion of a checkpoint, not an execution result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from export_decoder_weights import (ALIGNMENT, MODELS, SafeTensorWeights,
                                    checkpoint_sha256, parameter_specs,
                                    sha256_file)
from split_decoder_weights import STACK_BOTTOM, STACK_TOP, CONSOLE_MMIO, verify_image
from stitch import build, model_specs


HERE = Path(__file__).resolve().parent
FIRST_BASE = 0x20000000
SECOND_BASE = 0x80000000
SECOND_LIMIT = 0xFF000000


def plan(model: str, layers: int, first_base: int = FIRST_BASE,
         second_base: int = SECOND_BASE) -> dict:
    if model not in MODELS:
        raise ValueError(f"unknown decoder model: {model}")
    spec = dict(model_specs()[model])
    if not 1 <= layers <= spec["num_hidden_layers"]:
        raise ValueError("layer count exceeds pinned model depth")
    if (first_base < 0x20000000 or first_base % ALIGNMENT or
            first_base >= STACK_BOTTOM or second_base < STACK_TOP or
            second_base % ALIGNMENT or second_base >= SECOND_LIMIT):
        raise ValueError("FP16 image bases intersect reserved device memory")
    spec["num_hidden_layers"] = layers
    graph = build(model, prefill=1, decode_steps=1, specs={model: spec})
    parameters, uses = parameter_specs(graph)
    segments = [
        {"index": 0, "image_file": "weights-0.bin", "gpu_base_address": first_base,
         "image_size_bytes": 0},
        {"index": 1, "image_file": "weights-1.bin", "gpu_base_address": second_base,
         "image_size_bytes": 0},
    ]
    limits = [STACK_BOTTOM, SECOND_LIMIT]
    chosen = 0
    for item in parameters:
        storage = "fp32" if item["operation"] in ("rmsnorm", "bias_add") else "fp16"
        size = math.prod(item["shape"]) * (4 if storage == "fp32" else 2)
        offset = (segments[chosen]["image_size_bytes"] + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        if segments[chosen]["gpu_base_address"] + offset + size > limits[chosen]:
            chosen += 1
            if chosen == len(segments):
                raise ValueError(f"{model}: FP16 image exceeds safe GMEM regions at {item['logical_name']}")
            offset = 0
        item.update(storage_dtype=storage, size_bytes=size,
                    segment_index=chosen, offset_bytes=offset,
                    gpu_address=segments[chosen]["gpu_base_address"] + offset,
                    overlaps_warp_stacks=False)
        segments[chosen]["image_size_bytes"] = offset + size
    segments = [item for item in segments if item["image_size_bytes"]]
    return {
        "model": model, "family": spec["family"], "layers": layers,
        "pinned_model_layers": model_specs()[model]["num_hidden_layers"],
        "graph_stages": len(graph.stages), "parameter_uses": uses,
        "distinct_parameters": len(parameters), "dtype": "mixed_fp16",
        "linear_layout": "row_major_in_out", "alignment_bytes": ALIGNMENT,
        "gpu_base_address": first_base,
        "gpu_end_address_exclusive": max(item["gpu_base_address"] + item["image_size_bytes"]
                                          for item in segments),
        "image_size_bytes": sum(item["image_size_bytes"] for item in segments),
        "fits_32_bit_address_space": True,
        "warp_stack_reserved_range": [STACK_BOTTOM, STACK_TOP],
        "stack_overlap_parameters": [],
        "segments": segments, "parameters": parameters,
        "checkpoint_weights": False, "device_execution": False,
    }


def export(layout: dict, checkpoint_dir: Path, output_dir: Path) -> dict:
    model = layout["model"]
    spec = model_specs()[model]
    config_path = checkpoint_dir / "config.json"
    weight_path = checkpoint_dir / "model.safetensors"
    if sha256_file(config_path) != spec["source_sha256"]:
        raise ValueError("checkpoint config differs from pinned model config")
    weight_hash = checkpoint_sha256(model)
    if sha256_file(weight_path) != weight_hash:
        raise ValueError("checkpoint weights differ from pinned full-model check")
    provider = SafeTensorWeights(weight_path, spec["family"])
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = hashlib.sha256()
    for segment in layout["segments"]:
        destination = output_dir / segment["image_file"]
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        digest = hashlib.sha256()
        try:
            with temporary.open("wb") as stream:
                for item in layout["parameters"]:
                    if item["segment_index"] != segment["index"]:
                        continue
                    gap = item["offset_bytes"] - stream.tell()
                    if gap < 0 or gap >= ALIGNMENT:
                        raise ValueError("invalid FP16 image alignment")
                    pad = bytes(gap)
                    stream.write(pad)
                    digest.update(pad)
                    overall.update(pad)
                    logical = item["logical_name"]
                    shape = tuple(item["shape"])
                    if item["operation"] == "embedding":
                        value = provider.embedding_table(logical, *shape)
                    else:
                        value = provider(logical, shape)
                    if item["gemma_weight_offset"]:
                        value = value + 1.0
                    dtype = "<f2" if item["storage_dtype"] == "fp16" else "<f4"
                    array = np.ascontiguousarray(value, dtype=dtype)
                    if array.shape != shape or array.nbytes != item["size_bytes"]:
                        raise ValueError(f"{logical}: checkpoint value has wrong shape or size")
                    restored = array.astype(np.float32)
                    if not np.isfinite(restored).all():
                        raise ValueError(f"{logical}: FP16 conversion produced nonfinite values")
                    delta = np.abs(restored - value)
                    item["roundtrip_changed_elements"] = int(np.count_nonzero(restored != value))
                    item["roundtrip_max_abs_error"] = float(np.max(delta))
                    raw = memoryview(array).cast("B")
                    stream.write(raw)
                    digest.update(raw)
                    overall.update(raw)
                    item["checkpoint_key"] = provider.key(logical)
                    item["packed_sha256"] = hashlib.sha256(raw).hexdigest()
                if stream.tell() != segment["image_size_bytes"]:
                    raise ValueError("FP16 image segment size differs from plan")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        segment["image_sha256"] = digest.hexdigest()
    result = dict(layout)
    result.update(checkpoint_config_sha256=spec["source_sha256"],
                  checkpoint_weight_sha256=weight_hash,
                  image_sha256=overall.hexdigest(), checkpoint_weights=True,
                  roundtrip_changed_elements=sum(item["roundtrip_changed_elements"]
                                                 for item in layout["parameters"]),
                  roundtrip_max_abs_error=max(item["roundtrip_max_abs_error"]
                                              for item in layout["parameters"]))
    manifest_path = output_dir / "weights-image.json"
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    verify_image(manifest_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--out-dir", type=Path,
                        default=HERE / "generated/checkpoint-fp16")
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
