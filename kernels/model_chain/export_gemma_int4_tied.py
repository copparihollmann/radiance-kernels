#!/usr/bin/env python3
"""Pack full Gemma with INT8 embedding/attention and INT4 feed-forward layers.

This is a lossy software format for the 2 GiB Radiance GPU region. Norms and
biases remain FP32. Check generated tokens against the pinned FP32 checkpoint.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from export_decoder_int8 import export
from export_decoder_weights import ALIGNMENT, parameter_specs
from split_decoder_weights import STACK_BOTTOM, STACK_TOP
from stitch import build, model_specs


MODEL = "gemma_2_2b_it"
# The full greedy program's checked LOAD regions end above 0x14000000.
BASE = 0x14200000
GPU_SIZE = 0x80000000


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def plan(layers: int, base: int = BASE) -> dict:
    spec = dict(model_specs()[MODEL])
    if not 1 <= layers <= spec["num_hidden_layers"]:
        raise ValueError("layer count exceeds pinned Gemma depth")
    if base < 0 or base % 4096 or base >= STACK_BOTTOM:
        raise ValueError("invalid page-aligned checkpoint base")
    spec["num_hidden_layers"] = layers
    graph = build(MODEL, prefill=1, decode_steps=1, specs={MODEL: spec})
    parameters, uses = parameter_specs(graph)
    size = 0
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
            item.update(storage_dtype="int8_tied", alias_of="embed_tokens")
            continue
        if operation not in ("embedding", "linear", "rmsnorm", "bias_add"):
            raise ValueError(f"{logical}: unsupported parameter operation")
        storage = ("int8_scaled" if operation == "embedding" or
                   logical.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else
                   "fp32" if operation in ("rmsnorm", "bias_add") else
                   "int4_scaled")
        elements = math.prod(item["shape"])
        weight_bytes = (elements * 4 if storage == "fp32" else
                        elements if storage == "int8_scaled" else
                        (elements + 1) // 2)
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
        if logical == "embed_tokens":
            embedding = item
    if base + size > STACK_BOTTOM or base + size > GPU_SIZE:
        raise ValueError("Gemma INT4/INT8 checkpoint exceeds the 2 GiB GPU region")
    return {
        "model": MODEL, "family": spec["family"], "layers": layers,
        "pinned_model_layers": model_specs()[MODEL]["num_hidden_layers"],
        "graph_stages": len(graph.stages), "parameter_uses": uses,
        "distinct_parameters": len(parameters), "physical_parameters": len(parameters) - 1,
        "dtype": "int4_int8_tied", "linear_layout": "row_major_in_out",
        "alignment_bytes": ALIGNMENT, "gpu_base_address": base,
        "gpu_end_address_exclusive": base + size, "image_size_bytes": size,
        "fits_32_bit_address_space": True,
        "warp_stack_reserved_range": [STACK_BOTTOM, STACK_TOP],
        "stack_overlap_parameters": [],
        "segments": [{"index": 0, "image_file": "weights-0.bin",
                      "gpu_base_address": base, "image_size_bytes": size}],
        "parameters": parameters, "checkpoint_weights": False,
        "device_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "generated/checkpoint-int4")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    layout = plan(args.layers)
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
