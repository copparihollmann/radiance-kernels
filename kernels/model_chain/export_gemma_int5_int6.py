#!/usr/bin/env python3
"""Pack Gemma with a tied group-64 INT5 table and INT6 feed-forward layers.

The image is split around the device ELF: 15 early feed-forward matrices go
below 0x10000000, and the remaining parameters begin at 0x14200000. The
exporter checks the pinned checkpoint and every image segment hash.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from export_decoder_int8 import export
from export_decoder_weights import ALIGNMENT, parameter_specs
from split_decoder_weights import STACK_BOTTOM
from stitch import build, model_specs


MODEL = "gemma_2_2b_it"
LOW_BASE = 0x01000000
LOW_LIMIT = 0x10000000
HIGH_BASE = 0x14200000
HIGH_LIMIT = STACK_BOTTOM
GPU_SIZE = 0x80000000
LOW_FFN_MATRICES = 15


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def plan(layers: int = 26) -> dict:
    spec = dict(model_specs()[MODEL])
    if layers != spec["num_hidden_layers"]:
        raise ValueError("this compact Gemma format requires all 26 layers")
    graph = build(MODEL, prefill=1, decode_steps=1)
    parameters, uses = parameter_specs(graph)
    segments = [
        {"index": 0, "image_file": "weights-low.bin",
         "gpu_base_address": LOW_BASE, "image_size_bytes": 0},
        {"index": 1, "image_file": "weights-high.bin",
         "gpu_base_address": HIGH_BASE, "image_size_bytes": 0},
    ]
    low_used = 0
    embedding = None
    for item in parameters:
        logical, operation = item["logical_name"], item["operation"]
        if logical == "lm_head":
            if embedding is None:
                raise ValueError("tied Gemma output has no input embedding")
            item.update({key: embedding[key] for key in (
                "weight_size_bytes", "scale_count", "size_bytes", "segment_index",
                "offset_bytes", "gpu_address", "scale_offset_bytes",
                "scale_gpu_address", "overlaps_warp_stacks")})
            item.update(storage_dtype="int5_tied", alias_of="embed_tokens")
            continue
        ffn = logical.endswith(("gate_proj", "up_proj", "down_proj"))
        attention = logical.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))
        if operation not in ("embedding", "linear", "rmsnorm", "bias_add"):
            raise ValueError(f"{logical}: unsupported parameter operation")
        storage = ("int5_group64" if operation == "embedding" else
                   "int6_scaled" if ffn else
                   "int8_scaled" if attention else "fp32")
        elements = math.prod(item["shape"])
        weight_bytes = (elements * 4 if storage == "fp32" else
                        elements if storage == "int8_scaled" else
                        (elements * (5 if storage == "int5_group64" else 6) + 7) // 8)
        scale_count = (item["shape"][0] * (item["shape"][1] // 64)
                       if operation == "embedding" else
                       item["shape"][1] if operation == "linear" else 0)
        relative_scale = align(weight_bytes) if scale_count else None
        length = relative_scale + 4 * scale_count if scale_count else weight_bytes
        segment_index = 0 if ffn and low_used < LOW_FFN_MATRICES else 1
        if segment_index == 0:
            low_used += 1
        segment = segments[segment_index]
        offset = align(segment["image_size_bytes"])
        base = segment["gpu_base_address"]
        item.update(storage_dtype=storage, weight_size_bytes=weight_bytes,
                    scale_count=scale_count, size_bytes=length,
                    segment_index=segment_index, offset_bytes=offset,
                    gpu_address=base + offset,
                    scale_offset_bytes=offset + relative_scale
                    if scale_count else None,
                    scale_gpu_address=base + offset + relative_scale
                    if scale_count else None,
                    overlaps_warp_stacks=False)
        segment["image_size_bytes"] = offset + length
        if logical == "embed_tokens":
            embedding = item
    if low_used != LOW_FFN_MATRICES:
        raise ValueError("compact image did not select all low FFN matrices")
    if (LOW_BASE + segments[0]["image_size_bytes"] > LOW_LIMIT or
            HIGH_BASE + segments[1]["image_size_bytes"] > HIGH_LIMIT or
            HIGH_BASE + segments[1]["image_size_bytes"] > GPU_SIZE):
        raise ValueError("compact Gemma image exceeds available GPU regions")
    return {
        "model": MODEL, "family": spec["family"], "layers": layers,
        "pinned_model_layers": layers, "graph_stages": len(graph.stages),
        "parameter_uses": uses, "distinct_parameters": len(parameters),
        "physical_parameters": len(parameters) - 1,
        "dtype": "int5_group64_int6_tied", "linear_layout": "row_major_in_out",
        "alignment_bytes": ALIGNMENT, "gpu_base_address": LOW_BASE,
        "gpu_end_address_exclusive": max(segment["gpu_base_address"] +
                                         segment["image_size_bytes"]
                                         for segment in segments),
        "image_size_bytes": sum(segment["image_size_bytes"] for segment in segments),
        "fits_32_bit_address_space": True,
        "warp_stack_reserved_range": [STACK_BOTTOM, 0x7F000000],
        "stack_overlap_parameters": [], "segments": segments,
        "parameters": parameters, "checkpoint_weights": False,
        "device_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    layout = plan()
    if args.plan_only:
        print({key: value for key, value in layout.items() if key != "parameters"})
        return
    if args.checkpoint_dir is None:
        parser.error("--checkpoint-dir is required for export")
    result = export(layout, args.checkpoint_dir, args.out_dir / MODEL)
    print(f"Gemma: {result['physical_parameters']} physical parameters, "
          f"{result['image_size_bytes']} bytes, SHA-256 {result['image_sha256']}")


if __name__ == "__main__":
    main()
