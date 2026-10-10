#!/usr/bin/env python3
"""Assign shared GMEM buffers to a stitched graph using tensor lifetimes.

The graph's execution order and nested schedule remain unchanged. This is a
storage plan, not a claim that every graph operation has a device backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ALIGNMENT = 64
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "kernels/evaluation/llm"))
from stitch import build, model_specs  # noqa: E402


def size_bytes(tensor: dict) -> int:
    return ((math.prod(tensor["shape"]) * 4 + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT)


def plan(graph) -> dict:
    writes = {stage["writes"] for stage in graph.stages}
    lifetime = {
        name: {"birth": -1 if name not in writes else None,
               "last_use": -1, "size_bytes": size_bytes(tensor)}
        for name, tensor in graph.tensors.items()
    }
    for index, stage in enumerate(graph.stages):
        for name in stage["reads"]:
            lifetime[name]["last_use"] = max(lifetime[name]["last_use"], index)
        lifetime[stage["writes"]]["birth"] = index
    for name in graph.outputs:
        lifetime[name]["last_use"] = len(graph.stages)
    dead_stage_outputs = []
    for name, item in lifetime.items():
        if item["birth"] is not None and item["last_use"] < item["birth"]:
            dead_stage_outputs.append(name)
            item["last_use"] = item["birth"]
    free: list[tuple[int, int]] = []
    active: dict[str, tuple[int, int]] = {}
    arena_end = 0
    live_peak = 0

    def release_before(index: int) -> None:
        for name in list(active):
            if lifetime[name]["last_use"] < index:
                free.append(active.pop(name))

    def assign(name: str) -> None:
        nonlocal arena_end, live_peak
        needed = lifetime[name]["size_bytes"]
        candidates = [(size, i) for i, (_, size) in enumerate(free) if size >= needed]
        if candidates:
            _, selected = min(candidates)
            offset, available = free.pop(selected)
            if available > needed:
                free.append((offset + needed, available - needed))
        else:
            offset = arena_end
            arena_end += needed
        lifetime[name]["offset_bytes"] = offset
        active[name] = (offset, needed)
        live_peak = max(live_peak, sum(size for _, size in active.values()))

    for name in graph.tensors:
        if name not in writes:
            assign(name)
    for index, stage in enumerate(graph.stages):
        release_before(index)
        assign(stage["writes"])
    for name in graph.tensors:
        item = lifetime[name]
        if item["birth"] is None or "offset_bytes" not in item:
            raise ValueError(f"{name}: missing buffer allocation")
    # The physical intervals of tensors alive at the same stage must not
    # overlap, including the stage's input and output buffers.
    for index in range(len(graph.stages)):
        active_at_stage = sorted((item["offset_bytes"], item["offset_bytes"] +
                                  item["size_bytes"], name)
                                 for name, item in lifetime.items()
                                 if item["birth"] <= index <= item["last_use"])
        for (_, previous_end, previous_name), (start, _, name) in zip(
                active_at_stage, active_at_stage[1:]):
            if start < previous_end:
                raise ValueError(f"{previous_name} and {name} overlap while live")
    schedule = graph.execution_schedule
    return {
        "model": graph.model, "stage_count": len(graph.stages),
        "tensor_count": len(graph.tensors), "arena_bytes": arena_end,
        "peak_live_bytes": live_peak,
        "unreused_bytes": sum(item["size_bytes"] for item in lifetime.values()),
        "unused_stage_outputs": dead_stage_outputs,
        "execution_schedule_sha256": hashlib.sha256(
            json.dumps(schedule, sort_keys=True).encode()).hexdigest()
            if schedule is not None else None,
        "allocation": lifetime, "device_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=model_specs(), required=True)
    parser.add_argument("--out", type=Path,
                        default=HERE / "evaluation/buffer-plan.json")
    args = parser.parse_args()
    result = plan(build(args.model))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.model}: {result['tensor_count']} tensors, "
          f"{result['arena_bytes']} arena bytes, "
          f"{result['peak_live_bytes']} peak live bytes")


if __name__ == "__main__":
    main()
