#!/usr/bin/env python3
"""Pack the pinned SmolVLA checkpoint for the loop-aware model schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from export_decoder_weights import ALIGNMENT, sha256_file
from split_decoder_weights import STACK_BOTTOM, STACK_TOP, verify_image


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "kernels/evaluation/llm"))
from stitch import build, model_specs  # noqa: E402


FIRST_BASE = 0x30000000
SECOND_BASE = 0x80000000
SECOND_LIMIT = 0xFF000000


def parameter_specs(graph) -> tuple[list[dict], int]:
    parameters = {}
    uses = 0
    for stage in graph.stages:
        attrs = stage["attrs"]
        binding = dict(attrs.get("parameters", {}))
        if "parameter" in attrs:
            binding[attrs["parameter"]] = attrs.get("checkpoint_shape",
                                                    [stage["shape"][-1]])
        for name, checkpoint_shape in binding.items():
            checkpoint_shape = list(checkpoint_shape)
            runtime_shape = (list(attrs["weight_shape"])
                             if stage["op"] == "linear" else checkpoint_shape)
            item = {
                "logical_name": name, "checkpoint_key": name,
                "checkpoint_shape": checkpoint_shape,
                "shape": runtime_shape, "operation": stage["op"],
                "transpose_checkpoint_matrix": stage["op"] == "linear",
            }
            if name in parameters and parameters[name] != item:
                raise ValueError(f"{name}: inconsistent SmolVLA parameter use")
            parameters[name] = item
            uses += 1
    return list(parameters.values()), uses


def plan(first_base: int = FIRST_BASE, second_base: int = SECOND_BASE) -> dict:
    if (first_base < FIRST_BASE or first_base % ALIGNMENT or
            first_base >= STACK_BOTTOM or second_base < STACK_TOP or
            second_base % ALIGNMENT or second_base >= SECOND_LIMIT):
        raise ValueError("SmolVLA weight bases intersect reserved device memory")
    graph = build("smolvla_base")
    parameters, uses = parameter_specs(graph)
    segments = [
        {"index": 0, "image_file": "weights-0.bin", "gpu_base_address": first_base,
         "image_size_bytes": 0},
        {"index": 1, "image_file": "weights-1.bin", "gpu_base_address": second_base,
         "image_size_bytes": 0},
    ]
    limits = [STACK_BOTTOM, SECOND_LIMIT]
    selected = 0
    for item in parameters:
        size = 4 * math.prod(item["shape"])
        offset = (segments[selected]["image_size_bytes"] + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        if segments[selected]["gpu_base_address"] + offset + size > limits[selected]:
            selected += 1
            if selected == len(segments):
                raise ValueError(f"SmolVLA image exceeds safe GMEM at {item['logical_name']}")
            offset = 0
        item.update(storage_dtype="fp32", size_bytes=size, segment_index=selected,
                    offset_bytes=offset,
                    gpu_address=segments[selected]["gpu_base_address"] + offset,
                    overlaps_warp_stacks=False)
        segments[selected]["image_size_bytes"] = offset + size
    segments = [segment for segment in segments if segment["image_size_bytes"]]
    return {
        "model": "smolvla_base", "family": "smolvla", "dtype": "fp32",
        "graph_stages": len(graph.stages), "parameter_uses": uses,
        "distinct_parameters": len(parameters),
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
        "execution_schedule_sha256": hashlib.sha256(
            json.dumps(graph.execution_schedule, sort_keys=True).encode()).hexdigest(),
    }


def export(layout: dict, checkpoint_dir: Path, output_dir: Path) -> dict:
    from safetensors import safe_open
    import torch

    spec = model_specs()["smolvla_base"]
    weight_path = checkpoint_dir / "model.safetensors"
    if sha256_file(checkpoint_dir / "config.json") != spec["source_sha256"]:
        raise ValueError("SmolVLA config differs from pinned revision")
    if sha256_file(weight_path) != spec["checkpoint_weight_sha256"]:
        raise ValueError("SmolVLA weights differ from pinned checkpoint")
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
                        raise ValueError("invalid SmolVLA image alignment")
                    pad = bytes(gap)
                    stream.write(pad)
                    digest.update(pad)
                    overall.update(pad)
                    with safe_open(str(weight_path), framework="pt", device="cpu") as file:
                        tensor = file.get_tensor(item["checkpoint_key"])
                    if list(tensor.shape) != item["checkpoint_shape"]:
                        raise ValueError(f"{item['checkpoint_key']}: checkpoint shape differs")
                    if item["transpose_checkpoint_matrix"]:
                        tensor = tensor.T
                    value = np.ascontiguousarray(tensor.to(dtype=torch.float32).numpy(),
                                                  dtype="<f4")
                    if list(value.shape) != item["shape"] or value.nbytes != item[
                            "size_bytes"] or not np.isfinite(value).all():
                        raise ValueError(f"{item['logical_name']}: invalid packed parameter")
                    raw = memoryview(value).cast("B")
                    stream.write(raw)
                    digest.update(raw)
                    overall.update(raw)
                    item["packed_sha256"] = hashlib.sha256(raw).hexdigest()
                if stream.tell() != segment["image_size_bytes"]:
                    raise ValueError("SmolVLA segment size differs from plan")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        segment["image_sha256"] = digest.hexdigest()
    result = dict(layout)
    result.update(checkpoint_config_sha256=spec["source_sha256"],
                  checkpoint_weight_sha256=spec["checkpoint_weight_sha256"],
                  image_sha256=overall.hexdigest(), checkpoint_weights=True)
    manifest = output_dir / "weights-image.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    verify_image(manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--out-dir", type=Path,
                        default=HERE / "generated/checkpoint-smolvla")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    layout = plan()
    if args.plan_only:
        print(json.dumps({key: value for key, value in layout.items()
                          if key != "parameters"}, indent=2))
        return
    if args.checkpoint_dir is None:
        parser.error("--checkpoint-dir is required for export")
    result = export(layout, args.checkpoint_dir, args.out_dir / "smolvla_base")
    print(f"SmolVLA: {result['distinct_parameters']} parameters, "
          f"{result['image_size_bytes']} bytes, SHA-256 {result['image_sha256']}")


if __name__ == "__main__":
    main()
