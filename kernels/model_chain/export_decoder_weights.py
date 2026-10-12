#!/usr/bin/env python3
"""Pack checkpoint decoder weights in the layout consumed by model_chain.

The binary stays under generated/ and can be preloaded into Cyclotron GMEM.
The JSON records each parameter's logical name, shape, byte offset and address.
This creates a weight image; it does not compile or execute a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernels/evaluation/llm"))
from checkpoint import SafeTensorWeights  # noqa: E402
from stitch import build, model_specs  # noqa: E402

MODELS = ("tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma_2_2b_it")
ADDRESS_SPACE_SIZE = 1 << 32
ALIGNMENT = 64
# lib/src/mu_start.S places 8 KiB per lane below 0x7F000000. Reserve the
# configured maximum of two cores × eight warps × sixteen lanes.
WARP_STACK_BOTTOM = 0x7EE00000
WARP_STACK_TOP = 0x7F000000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_sha256(model: str) -> str:
    records = json.loads((ROOT / "kernels/evaluation/llm/checkpoint-results.json").read_text())
    matches = [row["checkpoint_sha256"] for row in records["checks"]
               if row["model"] == model and row["layers_checked"] ==
               model_specs()[model]["num_hidden_layers"]]
    if len(matches) != 1:
        raise ValueError(f"{model}: no pinned full-checkpoint hash is recorded")
    return matches[0]


def parameter_specs(graph) -> tuple[list[dict], int]:
    parameters: dict[str, dict] = {}
    uses = 0
    for stage in graph.stages:
        attrs = stage["attrs"]
        name = attrs.get("parameter")
        if name is None:
            continue
        op = stage["op"]
        if op == "embedding":
            shape = (attrs["vocab_size"], stage["shape"][-1])
        elif op == "linear":
            shape = tuple(attrs["weight_shape"])
        elif op == "rmsnorm":
            shape = (stage["shape"][-1],)
        elif op == "bias_add":
            shape = tuple(stage["shape"][2:])
        else:
            raise ValueError(f"{stage['id']}: unsupported checkpoint parameter op {op}")
        item = {"logical_name": name, "operation": op, "shape": list(shape),
                "gemma_weight_offset": bool(attrs.get("gemma_weight_offset", False))}
        if name in parameters and parameters[name] != item:
            raise ValueError(f"{name}: inconsistent use across model stages")
        parameters[name] = item
        uses += 1
    return list(parameters.values()), uses


def plan(model: str, layers: int, base: int) -> dict:
    if model not in MODELS:
        raise ValueError(f"unknown decoder model: {model}")
    spec = dict(model_specs()[model])
    if not 1 <= layers <= spec["num_hidden_layers"]:
        raise ValueError("layer count exceeds pinned model depth")
    if base < 0 or base % ALIGNMENT:
        raise ValueError("GPU base address must be nonnegative and 64-byte aligned")
    spec["num_hidden_layers"] = layers
    graph = build(model, prefill=1, decode_steps=1, specs={model: spec})
    parameters, uses = parameter_specs(graph)
    offset = 0
    for item in parameters:
        offset = (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        item["offset_bytes"] = offset
        item["gpu_address"] = base + offset
        item["size_bytes"] = 4 * math.prod(item["shape"])
        item["overlaps_warp_stacks"] = (
            item["gpu_address"] < WARP_STACK_TOP
            and item["gpu_address"] + item["size_bytes"] > WARP_STACK_BOTTOM)
        offset += item["size_bytes"]
    return {
        "model": model, "family": spec["family"], "layers": layers,
        "pinned_model_layers": model_specs()[model]["num_hidden_layers"],
        "graph_stages": len(graph.stages), "parameter_uses": uses,
        "distinct_parameters": len(parameters), "dtype": "fp32",
        "linear_layout": "row_major_in_out",
        "alignment_bytes": ALIGNMENT, "gpu_base_address": base,
        "image_size_bytes": offset, "gpu_end_address_exclusive": base + offset,
        "fits_32_bit_address_space": base + offset <= ADDRESS_SPACE_SIZE,
        "warp_stack_reserved_range": [WARP_STACK_BOTTOM, WARP_STACK_TOP],
        "stack_overlap_parameters": [item["logical_name"] for item in parameters
                                     if item["overlaps_warp_stacks"]],
        "parameters": parameters,
        "checkpoint_weights": False, "device_execution": False,
    }


def export(plan_data: dict, checkpoint_dir: Path, output_dir: Path) -> dict:
    model = plan_data["model"]
    if not plan_data["fits_32_bit_address_space"]:
        raise ValueError(f"{model}: FP32 image exceeds the 32-bit GMEM address space")
    spec = model_specs()[model]
    config_path = checkpoint_dir / "config.json"
    weight_path = checkpoint_dir / "model.safetensors"
    if sha256_file(config_path) != spec["source_sha256"]:
        raise ValueError("checkpoint config differs from pinned model config")
    expected_weights_sha = checkpoint_sha256(model)
    if sha256_file(weight_path) != expected_weights_sha:
        raise ValueError("checkpoint weights differ from pinned full-model check")
    provider = SafeTensorWeights(weight_path, spec["family"])
    output_dir.mkdir(parents=True, exist_ok=True)
    image = output_dir / "weights.bin"
    temporary = output_dir / "weights.bin.tmp"
    total_digest = hashlib.sha256()
    try:
        with temporary.open("wb") as stream:
            for item in plan_data["parameters"]:
                gap = item["offset_bytes"] - stream.tell()
                if gap < 0 or gap >= ALIGNMENT:
                    raise ValueError("invalid weight-image alignment")
                if gap:
                    pad = bytes(gap)
                    stream.write(pad)
                    total_digest.update(pad)
                logical = item["logical_name"]
                shape = tuple(item["shape"])
                if item["operation"] == "embedding":
                    value = provider.embedding_table(logical, *shape)
                else:
                    value = provider(logical, shape)
                if item["gemma_weight_offset"]:
                    value = value + 1.0
                array = np.ascontiguousarray(value, dtype="<f4")
                if array.shape != shape or array.nbytes != item["size_bytes"]:
                    raise ValueError(f"{logical}: checkpoint value has wrong shape or size")
                raw = memoryview(array).cast("B")
                stream.write(raw)
                total_digest.update(raw)
                item["checkpoint_key"] = provider.key(logical)
                item["packed_sha256"] = hashlib.sha256(raw).hexdigest()
            if stream.tell() != plan_data["image_size_bytes"]:
                raise ValueError("written image does not match planned size")
        os.replace(temporary, image)
    finally:
        if temporary.exists():
            temporary.unlink()
    result = dict(plan_data)
    result.update(checkpoint_config_sha256=spec["source_sha256"],
                  checkpoint_weight_sha256=expected_weights_sha,
                  image_sha256=total_digest.hexdigest(),
                  image_file="weights.bin", checkpoint_weights=True)
    (output_dir / "weights-image.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--base-address", type=lambda value: int(value, 0),
                        default=0x40000000)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--out-dir", type=Path,
                        default=HERE / "generated/checkpoint-weights")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    layout = plan(args.model, args.layers, args.base_address)
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
