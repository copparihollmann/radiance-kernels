#!/usr/bin/env python3
"""Profile OLMo 3 graph shapes without loading or copying its checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from math import prod
from pathlib import Path

from estimate_kv import kv_sizes
from profile_workloads import live_tensor_footprint
from stitch import HERE, build, model_specs


SCENARIOS = (("short_b1", 1, 128), ("short_b8", 8, 128),
             ("window_boundary_b1", 1, 4097), ("long_b1", 1, 8192))


def parameter_shapes(graph) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for stage in graph.stages:
        name = stage["attrs"].get("parameter")
        if name is None:
            continue
        if stage["op"] == "embedding":
            shape = (stage["attrs"]["vocab_size"], stage["shape"][-1])
        elif stage["op"] == "linear":
            shape = tuple(stage["attrs"]["weight_shape"])
        elif stage["op"] == "rmsnorm":
            shape = (stage["shape"][-1],)
        else:
            raise ValueError(f"unknown parameter-bearing stage {stage['id']}")
        if name in shapes and shapes[name] != shape:
            raise ValueError(f"parameter {name} changed shape across phases")
        shapes[name] = shape
    return shapes


def attention_macs(stage: dict) -> int:
    if stage["op"] != "causal_gqa":
        return 0
    batch, queries = stage["shape"][:2]
    past = stage["attrs"]["query_start"]
    window = stage["attrs"]["window"]
    visible_pairs = sum(min(past + q + 1, window or past + q + 1)
                        for q in range(queries))
    # One QK dot and one weighted-V dot per visible query/key/head pair.
    return 2 * batch * stage["attrs"]["q_heads"] * stage["attrs"]["head_dim"] * visible_pairs


def profile(batch: int, prefill: int) -> dict:
    spec = model_specs()["olmo3_7b"]
    graph = build("olmo3_7b", batch=batch, prefill=prefill,
                  decode_steps=2, generation="greedy")
    shapes = parameter_shapes(graph)
    expected_parameters = (2 * spec["vocab_size"] * spec["hidden_size"]
                           + spec["num_hidden_layers"] * (
                               4 * spec["hidden_size"] ** 2
                               + 3 * spec["hidden_size"] * spec["intermediate_size"]
                               + 4 * spec["hidden_size"])
                           + spec["hidden_size"])
    parameter_elements = sum(prod(shape) for shape in shapes.values())
    if parameter_elements != expected_parameters:
        raise ValueError("stitched parameter shapes differ from pinned OLMo 3 block formula")
    source = HERE / "inputs/olmo3-7b.json"
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["source_sha256"]:
        raise ValueError("pinned config hash differs")
    config = json.loads(raw)
    stages = Counter(stage["op"] for stage in graph.stages)
    per_phase = defaultdict(lambda: {"linear_macs": 0, "attention_dot_macs": 0})
    for stage in graph.stages:
        phase = stage["id"].split(".", 1)[0]
        if stage["op"] == "linear":
            per_phase[phase]["linear_macs"] += stage["attrs"]["macs"]
        per_phase[phase]["attention_dot_macs"] += attention_macs(stage)
    _, resident_kv, largest_kv, new_token_kv = kv_sizes(config, batch, prefill)
    graph_kv = (batch * prefill * 2 * spec["num_key_value_heads"]
                * spec["head_dim"] * 2 * spec["num_hidden_layers"])
    logits = graph.tensors[graph.outputs[0]]
    live = live_tensor_footprint(graph, "olmo3_7b")
    return {
        "status": "shape_estimate_only",
        "model": spec["name"],
        "source_config_sha256": spec["source_sha256"],
        "batch": batch, "prefill_tokens": prefill, "decode_steps": 2,
        "generation": "greedy", "stage_count": len(graph.stages),
        "operator_counts": dict(sorted(stages.items())),
        "analytical_parameter_elements": parameter_elements,
        "analytical_bf16_parameter_bytes": 2 * parameter_elements,
        "prefill_logit_shape": logits["shape"],
        "prefill_logit_bf16_bytes": 2 * prod(logits["shape"]),
        "window_retained_kv_bf16_bytes": resident_kv,
        "full_history_graph_kv_bf16_bytes": graph_kv,
        "largest_layer_kv_bf16_bytes": largest_kv,
        "new_token_kv_write_bf16_bytes": new_token_kv,
        "named_tensor_peak_fp32_mapping_bytes": live["peak_bytes"],
        "named_tensor_peak_stage": live["peak_stage"],
        "phase_work": dict(per_phase),
        "scope": "pinned-config and stitched-graph sizes; no checkpoint image, cache misses, measured bandwidth, latency, or device execution",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {"schema": "radiance.olmo3-shape-profiles.v1",
              "scenarios": {name: profile(batch, prefill)
                            for name, batch, prefill in SCENARIOS}}
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
