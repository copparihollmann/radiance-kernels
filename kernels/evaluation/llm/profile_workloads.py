#!/usr/bin/env python3
"""Record shape-derived stage traces for the pinned model workloads.

The byte counts are operand sizes, not cache misses or measured DRAM traffic.
Linear MACs exclude attention, normalization, and other arithmetic.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from math import prod
from pathlib import Path

from stitch import build, model_specs, SPECS


LOGICAL_DTYPE_BYTES = {"bf16": 2, "fp32": 4, "int32": 4, "bool": 1}
MAPPED_DTYPE_BYTES = {"bf16": 4, "fp32": 4, "int32": 4, "bool": 4}


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def stage_phase(model: str, stage_id: str) -> str:
    first = stage_id.split(".", 1)[0]
    if model == "smolvla_base":
        return first if first.startswith("denoise") else "prefix"
    return first


def tensor_bytes(tensor: dict, dtype_bytes: dict[str, int]) -> int:
    return prod(tensor["shape"]) * dtype_bytes[tensor["dtype"]]


def profile(model: str, batch: int, prefill: int, decode_steps: int) -> dict:
    if batch <= 0 or prefill <= 0 or decode_steps < 0:
        raise ValueError("batch and prefill must be positive; decode_steps cannot be negative")
    specs = model_specs()
    graph = build(model, batch=batch, prefill=prefill,
                  decode_steps=decode_steps, generation="greedy")
    smolvla = model == "smolvla_base"
    stages = []
    phases = defaultdict(lambda: {"stages": 0, "linear_macs": 0,
                                  "logical_operand_bytes": 0,
                                  "mapped_operand_bytes": 0, "ops": Counter()})
    produced = set(graph.tensors) - {stage["writes"] for stage in graph.stages}
    for index, stage in enumerate(graph.stages):
        if any(name not in produced for name in stage["reads"]):
            raise ValueError(f"{stage['id']}: reads a tensor before it is produced")
        produced.add(stage["writes"])
        reads = [graph.tensors[name] for name in stage["reads"]]
        writes = graph.tensors[stage["writes"]]
        logical_read = sum(tensor_bytes(item, LOGICAL_DTYPE_BYTES) for item in reads)
        logical_write = tensor_bytes(writes, LOGICAL_DTYPE_BYTES)
        mapped_read = sum(tensor_bytes(item, MAPPED_DTYPE_BYTES) for item in reads)
        mapped_write = tensor_bytes(writes, MAPPED_DTYPE_BYTES)
        macs = 0
        if stage["op"] == "linear":
            k, n = stage["attrs"]["weight_shape"]
            if prod(writes["shape"][2:]) != n:
                raise ValueError(f"{stage['id']}: linear output width differs from weight")
            macs = prod(writes["shape"]) * k
            if "macs" in stage["attrs"] and stage["attrs"]["macs"] != macs:
                raise ValueError(f"{stage['id']}: declared MAC count differs from shape")
        phase = stage_phase(model, stage["id"])
        record = {"index": index, "id": stage["id"], "phase": phase,
                  "op": stage["op"], "reads": stage["reads"],
                  "writes": stage["writes"], "shape": stage["shape"],
                  "attrs": stage["attrs"],
                  "logical_read_bytes": logical_read,
                  "logical_write_bytes": logical_write,
                  "mapped_read_bytes": mapped_read,
                  "mapped_write_bytes": mapped_write,
                  "linear_macs": macs}
        stages.append(record)
        summary = phases[phase]
        summary["stages"] += 1
        summary["linear_macs"] += macs
        summary["logical_operand_bytes"] += logical_read + logical_write
        summary["mapped_operand_bytes"] += mapped_read + mapped_write
        summary["ops"][stage["op"]] += 1
    schedule = graph.execution_schedule
    if smolvla:
        prefix = schedule["prefix_once_per_refill"]["stages"]
        iterations = schedule["denoise_loop"]["iterations"]
        flattened = prefix + [name for item in iterations for name in item["stages"]]
        if (flattened != [stage["id"] for stage in graph.stages] or
                len(iterations) != specs[model]["num_denoise_steps"] or
                schedule["policy_select_action"]["queued_actions_per_refill"] !=
                specs[model]["n_action_steps"]):
            raise ValueError("SmolVLA trace lost its action-refill loop structure")
    phase_summary = {name: {**values, "ops": dict(values["ops"])}
                     for name, values in phases.items()}
    largest_tensor = max(graph.tensors, key=lambda name: tensor_bytes(
        graph.tensors[name], MAPPED_DTYPE_BYTES))
    return {
        "schema": "radiance.llm.shape_trace.v1",
        "model": model, "source_spec_sha256": hashlib.sha256(SPECS.read_bytes()).hexdigest(),
        "graph_sha256": canonical_sha256(graph.report()),
        "scenario": {"batch": batch, "prefill_tokens": None if smolvla else prefill,
                     "cached_decode_steps": None if smolvla else decode_steps,
                     "scope": ("one_action_queue_refill" if smolvla else
                               "one_prefill_plus_cached_greedy_decode")},
        "interpretation": {
            "operand_bytes": "tensor shape times dtype width for each stage read and write; cache reuse and transactions are not modeled",
            "logical_dtypes": LOGICAL_DTYPE_BYTES,
            "mapped_buffer_dtypes": MAPPED_DTYPE_BYTES,
            "mapped_buffer_scope": "current FP32/32-bit software storage, excluding weights and temporary scratch",
            "linear_macs": "linear stages only; attention and other arithmetic excluded",
            "device_execution": False, "performance_measurement": False,
        },
        "outputs": graph.outputs,
        "execution_schedule": schedule,
        "stage_count": len(stages),
        "phase_summary": phase_summary,
        "totals": {
            "linear_macs": sum(item["linear_macs"] for item in stages),
            "logical_operand_bytes": sum(item["logical_read_bytes"] + item["logical_write_bytes"] for item in stages),
            "mapped_operand_bytes": sum(item["mapped_read_bytes"] + item["mapped_write_bytes"] for item in stages),
            "largest_mapped_tensor": largest_tensor,
            "largest_mapped_tensor_bytes": tensor_bytes(
                graph.tensors[largest_tensor], MAPPED_DTYPE_BYTES),
        },
        "stages": stages,
    }


def matrix(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    baseline = {"tinyllama": 2, "deepseek_r1_distill_qwen_1_5b": 1,
                "gemma_2_2b_it": 2}
    for model, pinned_prefill in baseline.items():
        cases.extend((model, batch, prefill, decode, label) for
                     batch, prefill, decode, label in (
                         (1, pinned_prefill, 2, "fixed_handoff_shape"),
                         (1, 128, 1, "medium_context"),
                         (8, 128, 1, "high_batch"),
                         (1, 1024, 1, "long_context")))
    cases.append(("smolvla_base", 1, 1, 0, "one_action_queue_refill"))
    rows = []
    for model, batch, prefill, decode, label in cases:
        result = profile(model, batch, prefill, decode)
        name = f"{model}-{label}.json"
        path = out_dir / name
        path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
        rows.append({"model": model, "case": label, "trace": name,
                     "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "scenario": result["scenario"],
                     "graph_sha256": result["graph_sha256"],
                     "stage_count": result["stage_count"],
                     "totals": result["totals"],
                     "phase_summary": result["phase_summary"]})
    index = {"schema": "radiance.llm.shape_trace_matrix.v1",
             "source_spec_sha256": hashlib.sha256(SPECS.read_bytes()).hexdigest(),
             "scope": "shape-derived software schedules; no measured utilization, cache, memory, or latency",
             "cases": rows}
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    index = matrix(args.out_dir)
    print(f"wrote {len(index['cases'])} shape traces to {args.out_dir}")


if __name__ == "__main__":
    main()
