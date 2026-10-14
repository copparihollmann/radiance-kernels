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
WEIGHT_MANIFESTS = {
    "tinyllama": "tinyllama-2g-int8-image.json",
    "deepseek_r1_distill_qwen_1_5b": "deepseek-2g-int8-image.json",
    "gemma_2_2b_it": "gemma-2g-int5-int6-image.json",
    "smolvla_base": "smolvla-2g-source-weights-image.json",
}
WEIGHT_ROOT = Path(__file__).resolve().parents[2] / "model_chain/evaluation"


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


def packed_footprint(names: set[str], parameters: dict[str, dict]) -> int:
    """Count unique GPU address intervals, including exact tied aliases once."""
    ranges = defaultdict(list)
    for name in names:
        item = parameters[name]
        ranges[item.get("segment_index", 0)].append(
            (item["gpu_address"], item["gpu_address"] + item["size_bytes"]))
    total = 0
    for intervals in ranges.values():
        end = -1
        for low, high in sorted(intervals):
            total += high - max(low, end) if high > end else 0
            end = max(end, high)
    return total


def live_tensor_footprint(graph, model: str) -> dict:
    """Count named tensors alive in schedule order with graph outputs retained."""
    sizes = {name: tensor_bytes(item, MAPPED_DTYPE_BYTES)
             for name, item in graph.tensors.items()}
    remaining = Counter(name for stage in graph.stages for name in stage["reads"])
    retained = set(graph.outputs)
    written = {stage["writes"] for stage in graph.stages}
    live = {name for name in graph.tensors if name not in written and
            (remaining[name] or name in retained)}
    current = sum(sizes[name] for name in live)
    peak = current
    peak_stage = "initial_inputs"
    phase_peaks = {}
    for stage in graph.stages:
        name = stage["writes"]
        if name in live or any(source not in live for source in stage["reads"]):
            raise ValueError(f"{stage['id']}: invalid live-tensor dependency")
        live.add(name)
        current += sizes[name]
        phase = stage_phase(model, stage["id"])
        if current > phase_peaks.get(phase, (0, ""))[0]:
            phase_peaks[phase] = (current, stage["id"])
        if current > peak:
            peak, peak_stage = current, stage["id"]
        for source in stage["reads"]:
            remaining[source] -= 1
            if remaining[source] == 0 and source not in retained:
                live.remove(source)
                current -= sizes[source]
        if remaining[name] == 0 and name not in retained:
            live.remove(name)
            current -= sizes[name]
    if not retained.issubset(live):
        raise ValueError("a graph output was released before completion")
    return {"peak_bytes": peak, "peak_stage": peak_stage,
            "phase_peaks": phase_peaks, "retained_output_bytes": sum(
                sizes[name] for name in retained)}


def profile(model: str, batch: int, prefill: int, decode_steps: int) -> dict:
    if batch <= 0 or prefill <= 0 or decode_steps < 0:
        raise ValueError("batch and prefill must be positive; decode_steps cannot be negative")
    specs = model_specs()
    graph = build(model, batch=batch, prefill=prefill,
                  decode_steps=decode_steps, generation="greedy")
    weight_path = WEIGHT_ROOT / WEIGHT_MANIFESTS[model]
    weight_manifest = json.loads(weight_path.read_text())
    if (weight_manifest["model"] != model or
            weight_manifest["checkpoint_config_sha256"] != specs[model]["source_sha256"]):
        raise ValueError(f"{model}: packed weight manifest differs from pinned model")
    parameters = {item["logical_name"]: item
                  for item in weight_manifest["parameters"]}
    if len(parameters) != len(weight_manifest["parameters"]):
        raise ValueError(f"{model}: duplicate logical weight names")
    smolvla = model == "smolvla_base"
    stages = []
    phases = defaultdict(lambda: {"stages": 0, "linear_macs": 0,
                                  "logical_operand_bytes": 0,
                                  "mapped_operand_bytes": 0, "ops": Counter(),
                                  "parameter_names": set()})
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
        parameter_names = set(stage["attrs"].get("parameters", {}))
        if stage["attrs"].get("parameter") is not None:
            parameter_names.add(stage["attrs"]["parameter"])
        if parameter_names - parameters.keys():
            raise ValueError(f"{stage['id']}: weight image lacks a graph parameter")
        phase = stage_phase(model, stage["id"])
        record = {"index": index, "id": stage["id"], "phase": phase,
                  "op": stage["op"], "reads": stage["reads"],
                  "writes": stage["writes"], "shape": stage["shape"],
                  "attrs": stage["attrs"],
                  "logical_read_bytes": logical_read,
                  "logical_write_bytes": logical_write,
                  "mapped_read_bytes": mapped_read,
                  "mapped_write_bytes": mapped_write,
                  "linear_macs": macs,
                  "packed_parameter_names": sorted(parameter_names),
                  "packed_parameter_storage_bytes": sum(
                      parameters[name]["size_bytes"] for name in parameter_names)}
        stages.append(record)
        summary = phases[phase]
        summary["stages"] += 1
        summary["linear_macs"] += macs
        summary["logical_operand_bytes"] += logical_read + logical_write
        summary["mapped_operand_bytes"] += mapped_read + mapped_write
        summary["ops"][stage["op"]] += 1
        summary["parameter_names"].update(parameter_names)
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
    phase_summary = {
        name: {**{key: value for key, value in values.items()
                  if key != "parameter_names"},
               "ops": dict(values["ops"]),
               "unique_packed_parameter_count": len(values["parameter_names"]),
               "unique_packed_parameter_footprint_bytes": packed_footprint(
                   values["parameter_names"], parameters)}
        for name, values in phases.items()}
    used_parameters = set().union(*(item["parameter_names"]
                                    for item in phases.values()))
    largest_tensor = max(graph.tensors, key=lambda name: tensor_bytes(
        graph.tensors[name], MAPPED_DTYPE_BYTES))
    largest_bytes = tensor_bytes(graph.tensors[largest_tensor], MAPPED_DTYPE_BYTES)
    liveness = live_tensor_footprint(graph, model)
    for phase, (peak_bytes, peak_stage) in liveness["phase_peaks"].items():
        phase_summary[phase]["peak_live_mapped_tensor_bytes"] = peak_bytes
        phase_summary[phase]["peak_live_stage"] = peak_stage
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
            "packed_weight_bytes": "storage footprint of named parameters, with tied GPU address intervals counted once; not bytes fetched from HBM",
            "live_mapped_tensor_bytes": "sum of distinct named tensor sizes alive in graph order, with initial graph inputs resident, all graph outputs retained, and each result allocated before its last input is released; in-place reuse, device buffers, scratch, and cache behavior are not modeled",
            "device_execution": False, "performance_measurement": False,
        },
        "packed_weights": {
            "manifest": "kernels/model_chain/evaluation/" + WEIGHT_MANIFESTS[model],
            "manifest_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
            "image_sha256": weight_manifest["image_sha256"],
            "format": weight_manifest["dtype"],
            "image_size_bytes": weight_manifest["image_size_bytes"],
            "used_logical_parameter_count": len(used_parameters),
            "used_packed_parameter_footprint_bytes": packed_footprint(
                used_parameters, parameters),
            "raw_image_bytes_verified_here": False,
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
            "largest_mapped_tensor_bytes": largest_bytes,
            "peak_live_mapped_tensor_bytes": liveness["peak_bytes"],
            "peak_live_stage": liveness["peak_stage"],
            "retained_output_bytes": liveness["retained_output_bytes"],
            "packed_image_plus_peak_live_mapped_tensor_bytes": (
                weight_manifest["image_size_bytes"] + liveness["peak_bytes"]),
            "packed_image_plus_largest_tensor_bytes": (
                weight_manifest["image_size_bytes"] + largest_bytes),
            "packed_image_plus_largest_tensor_exceeds_2gib": (
                weight_manifest["image_size_bytes"] + largest_bytes > 2**31),
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
                     "packed_weights": result["packed_weights"],
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
