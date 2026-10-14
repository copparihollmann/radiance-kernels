#!/usr/bin/env python3
"""Summarize the pinned architecture graphs without loading model weights.

These are operation counts and tensor shapes. They are not cache misses,
hardware utilization, transferred bytes, or elapsed cycles.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from math import prod
from pathlib import Path

from estimate_kv import kv_sizes
from inventory import inventory
from profile_workloads import (LOGICAL_DTYPE_BYTES, MAPPED_DTYPE_BYTES,
                               live_tensor_footprint, tensor_bytes)
from stitch import build, model_specs


HERE = Path(__file__).resolve().parent
GRAPH_KEYS = {
    "qwen3-0.6b": "qwen3_0_6b",
    "olmo3-7b": "olmo3_7b",
    "qwen3-30b-a3b": "qwen3_30b_a3b",
    "nemotron3-nano-4b": "nemotron3_nano_4b",
    "mistral-7b-v03": "mistral_7b_v03",
}
SCENARIOS = (("short_b1", 1, 128), ("short_b8", 8, 128),
             ("long_b1", 1, 8192))
DECODE_STEPS = 2


def attention_dot_macs(stage: dict) -> int:
    """Count QK and weighted-V dot products for the causal visible pairs."""
    if stage["op"] != "causal_gqa":
        return 0
    batch, queries = stage["shape"][:2]
    past = stage["attrs"]["query_start"]
    window = stage["attrs"].get("window")
    if window is None:
        pairs = queries * (past + 1) + queries * (queries - 1) // 2
    else:
        unsaturated = min(queries, max(0, window - past))
        pairs = (unsaturated * (past + 1) +
                 unsaturated * (unsaturated - 1) // 2 +
                 (queries - unsaturated) * window)
    return 2 * batch * stage["attrs"]["q_heads"] * stage["attrs"]["head_dim"] * pairs


def expert_macs(stage: dict) -> int:
    if stage["op"] not in ("expert_gate_up", "expert_down"):
        return 0
    shapes = list(stage["attrs"]["parameters"].values())
    if len(shapes) != 1 or stage["shape"][-1] != shapes[0][-2]:
        raise ValueError(f"{stage['id']}: expert output differs from weight shape")
    return prod(stage["shape"]) * shapes[0][-1]


def scenario_profile(model: str, config: dict, batch: int, prefill: int) -> dict:
    if prefill + DECODE_STEPS > int(config["max_position_embeddings"]):
        raise ValueError(f"{model}: scenario exceeds pinned context limit")
    graph = build(model, batch=batch, prefill=prefill,
                  decode_steps=DECODE_STEPS, generation="greedy")
    schedule = graph.execution_schedule
    ordered = (schedule["prefill_stages"] +
               [name for item in schedule["decode_loop"] for name in item["stages"]])
    if (ordered != [stage["id"] for stage in graph.stages] or
            len(schedule["decode_loop"]) != DECODE_STEPS):
        raise ValueError(f"{model}: prefill/decode schedule lost its dependency order")
    if graph.outputs != ["prefill.lm_head", "decode0.lm_head", "decode1.lm_head"]:
        raise ValueError(f"{model}: unexpected output sequence")

    phases = defaultdict(lambda: Counter())
    for stage in graph.stages:
        phase = stage["id"].split(".", 1)[0]
        counts = phases[phase]
        counts["stages"] += 1
        counts["linear_macs"] += (int(stage["attrs"]["macs"])
                                  if stage["op"] == "linear" else 0)
        counts["expert_macs"] += expert_macs(stage)
        counts["attention_dot_macs"] += attention_dot_macs(stage)
        counts["logical_operand_bytes"] += (
            sum(tensor_bytes(graph.tensors[name], LOGICAL_DTYPE_BYTES)
                for name in stage["reads"]) +
            tensor_bytes(graph.tensors[stage["writes"]], LOGICAL_DTYPE_BYTES))
        counts["mapped_operand_bytes"] += (
            sum(tensor_bytes(graph.tensors[name], MAPPED_DTYPE_BYTES)
                for name in stage["reads"]) +
            tensor_bytes(graph.tensors[stage["writes"]], MAPPED_DTYPE_BYTES))
    live = live_tensor_footprint(graph, model)
    largest_name = max(graph.tensors, key=lambda name: tensor_bytes(
        graph.tensors[name], MAPPED_DTYPE_BYTES))
    final_kv = [stage for stage in graph.stages
                if stage["op"] == "kv_append" and stage["id"].startswith("decode1.")]
    layers, window_kv, _, new_token_kv = kv_sizes(
        config, batch, prefill + DECODE_STEPS)
    if len(final_kv) != 2 * layers:
        raise ValueError(f"{model}: final K/V stages differ from pinned attention layers")
    full_history_kv = sum(tensor_bytes(graph.tensors[stage["writes"]],
                                       LOGICAL_DTYPE_BYTES)
                          for stage in final_kv)
    if full_history_kv < window_kv:
        raise ValueError(f"{model}: graph K/V is smaller than window-trimmed lower bound")
    return {
        "batch": batch, "prefill_tokens": prefill,
        "greedy_cached_decode_steps": DECODE_STEPS,
        "stage_count": len(graph.stages),
        "operator_counts": dict(sorted(Counter(stage["op"] for stage in graph.stages).items())),
        "phase_work": {name: dict(phases[name])
                       for name in ("prefill", "decode0", "decode1")},
        "prefill_logits_shape": graph.tensors["prefill.lm_head"]["shape"],
        "prefill_logits_mapped_bytes": tensor_bytes(
            graph.tensors["prefill.lm_head"], MAPPED_DTYPE_BYTES),
        "largest_mapped_tensor": largest_name,
        "largest_mapped_tensor_bytes": tensor_bytes(
            graph.tensors[largest_name], MAPPED_DTYPE_BYTES),
        "peak_live_named_mapped_tensor_bytes": live["peak_bytes"],
        "peak_live_stage": live["peak_stage"],
        "attention_layers": layers,
        "final_full_history_kv_logical_bytes": full_history_kv,
        "window_trimmed_kv_logical_bytes": window_kv,
        "new_token_kv_write_logical_bytes": new_token_kv,
    }


def matrix() -> dict:
    catalog = inventory(HERE / "inputs")
    specs = model_specs()
    sources = {source["case"]: source for source in _sources()}
    if set(GRAPH_KEYS) != {item["case"] for item in catalog["cases"]}:
        raise ValueError("architecture graph list differs from pinned inventory")
    cases = []
    for item in catalog["cases"]:
        key = GRAPH_KEYS[item["case"]]
        spec = specs[key]
        if spec["source_sha256"] != item["config_sha256"]:
            raise ValueError(f"{key}: graph and inventory use different configs")
        config_path = HERE / "inputs" / sources[item["case"]]["config_file"]
        raw = config_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["config_sha256"]:
            raise ValueError(f"{key}: config changed during profiling")
        config = json.loads(raw)
        scenarios = {name: scenario_profile(key, config, batch, prefill)
                     for name, batch, prefill in SCENARIOS}
        cases.append({
            "case": item["case"], "graph_key": key,
            "architecture_class": item["class"],
            "model": item["model"], "revision": item["revision"],
            "config_sha256": item["config_sha256"],
            "layer_counts": item["counts"],
            "checkpoint_use_cache": bool(config.get("use_cache", True)),
            "scenarios": scenarios,
        })
    return {
        "schema": "radiance.llm.architecture_shapes.v1",
        "scope": "pinned-config and stitched-graph shape estimates; no weights or device execution",
        "interpretation": {
            "linear_macs": "dense linear stages only",
            "expert_macs": "selected expert gate/up and down projections only",
            "attention_dot_macs": "idealized visible-pair QK and weighted-V dot products; excludes masked pairs and implementation overhead",
            "operand_bytes": "sum of stage-boundary tensor sizes, not cache misses or HBM transactions",
            "peak_live_named_mapped_tensor_bytes": "distinct named FP32/32-bit graph tensors, without scratch or in-place reuse",
            "final_full_history_kv_logical_bytes": "stitched graph retains all K/V positions, including sliding-attention layers",
            "window_trimmed_kv_logical_bytes": "analytical BF16 K/V with local windows evicted; requires a cache policy the graph does not implement",
            "mamba_state": "excluded from K/V estimates; see nemotron-state-profiles.json",
            "cached_decode": "a proposed graph mapping when the checkpoint config has use_cache=false; not a checkpoint execution claim",
            "performance_measurement": False,
        },
        "cases": cases,
    }


def _sources() -> list[dict]:
    import csv
    with (HERE / "inputs/sources.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = json.dumps(matrix(), indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
