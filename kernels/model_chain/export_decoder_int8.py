#!/usr/bin/env python3
"""Pack Gemma as INT8 body weights and one tied FP16 embedding table.

Linear weights use one FP32 scale per output column; embedding rows use one
shared FP16 table for input embedding and output projection. Norms and biases
remain FP32. The body conversion is lossy, so the compiler compares its NumPy
output with the original checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from export_decoder_weights import ALIGNMENT, SafeTensorWeights, parameter_specs, sha256_file
from export_decoder_fp16 import FIRST_BASE, SECOND_BASE, SECOND_LIMIT
from split_decoder_weights import STACK_BOTTOM, STACK_TOP, verify_image
from stitch import build, model_specs
from checkpoint import quantize_per_channel as quantize


HERE = Path(__file__).resolve().parent
MODEL = "gemma_2_2b_it"


def plan(layers: int, first_base: int = FIRST_BASE,
         second_base: int = SECOND_BASE) -> dict:
    spec = dict(model_specs()[MODEL])
    if not 1 <= layers <= spec["num_hidden_layers"]:
        raise ValueError("layer count exceeds pinned Gemma depth")
    if (first_base < FIRST_BASE or first_base % ALIGNMENT or
            first_base >= STACK_BOTTOM or second_base < STACK_TOP or
            second_base % ALIGNMENT or second_base >= SECOND_LIMIT):
        raise ValueError("INT8 image bases intersect reserved device memory")
    spec["num_hidden_layers"] = layers
    graph = build(MODEL, prefill=1, decode_steps=1, specs={MODEL: spec})
    parameters, uses = parameter_specs(graph)
    segments = [
        {"index": 0, "image_file": "weights-0.bin", "gpu_base_address": first_base,
         "image_size_bytes": 0},
        {"index": 1, "image_file": "weights-1.bin", "gpu_base_address": second_base,
         "image_size_bytes": 0},
    ]
    limits = [STACK_BOTTOM, SECOND_LIMIT]
    selected = 0
    embedding_item = None
    for item in parameters:
        operation = item["operation"]
        if item["logical_name"] == "lm_head":
            if embedding_item is None:
                raise ValueError("tied Gemma output has no input embedding")
            item.update(storage_dtype="fp16_tied", alias_of="embed_tokens",
                        weight_size_bytes=embedding_item["weight_size_bytes"],
                        scale_count=0, size_bytes=embedding_item["size_bytes"],
                        segment_index=embedding_item["segment_index"],
                        offset_bytes=embedding_item["offset_bytes"],
                        gpu_address=embedding_item["gpu_address"],
                        scale_offset_bytes=None, scale_gpu_address=None,
                        overlaps_warp_stacks=False)
            continue
        storage = ("fp16" if item["logical_name"] == "embed_tokens" else
                   "fp32" if operation in ("rmsnorm", "bias_add") else
                   "int8_scaled")
        weight_bytes = math.prod(item["shape"]) * (4 if storage == "fp32" else
                                                  2 if storage == "fp16" else 1)
        scale_count = ((item["shape"][0] if operation == "embedding" else
                        item["shape"][1]) if storage == "int8_scaled" else 0)
        scale_offset = (weight_bytes + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        size = scale_offset + 4 * scale_count if scale_count else weight_bytes
        offset = (segments[selected]["image_size_bytes"] + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        if segments[selected]["gpu_base_address"] + offset + size > limits[selected]:
            selected += 1
            if selected == len(segments):
                raise ValueError(f"Gemma INT8 image exceeds safe GMEM at {item['logical_name']}")
            offset = 0
            if segments[selected]["gpu_base_address"] + size > limits[selected]:
                raise ValueError(f"{item['logical_name']}: tensor exceeds safe segment")
        item.update(storage_dtype=storage, weight_size_bytes=weight_bytes,
                    scale_count=scale_count, size_bytes=size,
                    segment_index=selected, offset_bytes=offset,
                    gpu_address=segments[selected]["gpu_base_address"] + offset,
                    scale_offset_bytes=offset + scale_offset if scale_count else None,
                    scale_gpu_address=segments[selected]["gpu_base_address"] + offset +
                    scale_offset if scale_count else None,
                    overlaps_warp_stacks=False)
        segments[selected]["image_size_bytes"] = offset + size
        if item["logical_name"] == "embed_tokens":
            embedding_item = item
    segments = [item for item in segments if item["image_size_bytes"]]
    return {
        "model": MODEL, "family": "gemma2", "layers": layers,
        "pinned_model_layers": model_specs()[MODEL]["num_hidden_layers"],
        "graph_stages": len(graph.stages), "parameter_uses": uses,
        "distinct_parameters": len(parameters), "physical_parameters": len(parameters) - 1,
        "dtype": "int8_fp16_tied",
        "linear_layout": "row_major_in_out", "alignment_bytes": ALIGNMENT,
        "gpu_base_address": first_base,
        "gpu_end_address_exclusive": max(item["gpu_base_address"] +
                                          item["image_size_bytes"] for item in segments),
        "image_size_bytes": sum(item["image_size_bytes"] for item in segments),
        "fits_32_bit_address_space": True,
        "warp_stack_reserved_range": [STACK_BOTTOM, STACK_TOP],
        "stack_overlap_parameters": [], "segments": segments,
        "parameters": parameters, "checkpoint_weights": False,
        "device_execution": False,
    }


def export(layout: dict, checkpoint_dir: Path, output_dir: Path) -> dict:
    spec = model_specs()[MODEL]
    if sha256_file(checkpoint_dir / "config.json") != spec["source_sha256"]:
        raise ValueError("Gemma config differs from pinned revision")
    files = {name: sha256_file(checkpoint_dir / name)
             for name in spec["checkpoint_files_sha256"]}
    if files != spec["checkpoint_files_sha256"]:
        raise ValueError("Gemma checkpoint shard differs from pinned revision")
    provider = SafeTensorWeights(checkpoint_dir, "gemma2")
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
                    if item.get("alias_of"):
                        source = next(source for source in layout["parameters"]
                                      if source["logical_name"] == item["alias_of"])
                        item["checkpoint_key"] = provider.key(item["logical_name"])
                        if item["checkpoint_key"] != source["checkpoint_key"]:
                            raise ValueError("Gemma tied embedding checkpoint keys differ")
                        item["packed_sha256"] = source["packed_sha256"]
                        item["roundtrip_changed_elements"] = source[
                            "roundtrip_changed_elements"]
                        item["roundtrip_max_abs_error"] = source[
                            "roundtrip_max_abs_error"]
                        continue
                    gap = item["offset_bytes"] - stream.tell()
                    if gap < 0 or gap >= ALIGNMENT:
                        raise ValueError("invalid INT8 image alignment")
                    pad = bytes(gap)
                    stream.write(pad)
                    digest.update(pad)
                    overall.update(pad)
                    logical = item["logical_name"]
                    shape = tuple(item["shape"])
                    value = (provider.embedding_table(logical, *shape)
                             if item["operation"] == "embedding" else
                             provider(logical, shape))
                    if item["gemma_weight_offset"]:
                        value = value + 1.0
                    if item["storage_dtype"] in ("fp32", "fp16"):
                        packed = np.ascontiguousarray(
                            value, dtype="<f2" if item["storage_dtype"] == "fp16"
                            else "<f4")
                        scales = None
                        if item["storage_dtype"] == "fp16":
                            changed = 0
                            maximum = 0.0
                            for start in range(0, len(value), 64):
                                stop = min(start + 64, len(value))
                                delta = np.abs(packed[start:stop].astype(np.float32)
                                               - value[start:stop])
                                changed += int(np.count_nonzero(delta))
                                maximum = max(maximum, float(delta.max()))
                            metric = {"roundtrip_changed_elements": changed,
                                      "roundtrip_max_abs_error": maximum}
                        else:
                            metric = {"roundtrip_changed_elements": 0,
                                      "roundtrip_max_abs_error": 0.0}
                    else:
                        packed, scales, metric = quantize(value, item["operation"])
                    raw = memoryview(np.ascontiguousarray(packed)).cast("B")
                    if len(raw) != item["weight_size_bytes"]:
                        raise ValueError(f"{logical}: packed weight size differs from plan")
                    stream.write(raw)
                    digest.update(raw)
                    overall.update(raw)
                    parameter_hash = hashlib.sha256()
                    parameter_hash.update(raw)
                    if scales is not None:
                        gap = item["scale_offset_bytes"] - stream.tell()
                        if gap < 0 or gap >= ALIGNMENT:
                            raise ValueError("invalid scale alignment")
                        pad = bytes(gap)
                        stream.write(pad)
                        digest.update(pad)
                        overall.update(pad)
                        parameter_hash.update(pad)
                        scale_raw = memoryview(np.ascontiguousarray(scales)).cast("B")
                        stream.write(scale_raw)
                        digest.update(scale_raw)
                        overall.update(scale_raw)
                        parameter_hash.update(scale_raw)
                    item.update(metric)
                    item["checkpoint_key"] = provider.key(logical)
                    item["packed_sha256"] = parameter_hash.hexdigest()
                if stream.tell() != segment["image_size_bytes"]:
                    raise ValueError("INT8 image segment size differs from plan")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        segment["image_sha256"] = digest.hexdigest()
    result = dict(layout)
    result.update(checkpoint_config_sha256=spec["source_sha256"],
                  checkpoint_files_sha256=files,
                  checkpoint_weight_sha256=hashlib.sha256(
                      json.dumps(files, sort_keys=True).encode()).hexdigest(),
                  image_sha256=overall.hexdigest(), checkpoint_weights=True,
                  roundtrip_changed_elements=sum(item["roundtrip_changed_elements"]
                                                 for item in layout["parameters"]),
                  roundtrip_max_abs_error=max(item["roundtrip_max_abs_error"]
                                              for item in layout["parameters"]))
    manifest = output_dir / "weights-image.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    verify_image(manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--out-dir", type=Path,
                        default=HERE / "generated/checkpoint-int8")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    layout = plan(args.layers)
    if args.plan_only:
        print(json.dumps({key: value for key, value in layout.items()
                          if key != "parameters"}, indent=2))
        return
    if args.checkpoint_dir is None:
        parser.error("--checkpoint-dir is required for export")
    result = export(layout, args.checkpoint_dir, args.out_dir / MODEL)
    print(f"Gemma: {result['distinct_parameters']} parameters, "
          f"{result['image_size_bytes']} bytes, SHA-256 {result['image_sha256']}")


if __name__ == "__main__":
    main()
