"""Evaluate a Qwen3 greedy graph with a last-token prefill projection."""

from __future__ import annotations

from collections import Counter

import numpy as np

import reference
from stitch import Graph


def execute(graph: Graph, inputs: dict[str, np.ndarray],
            *, weights: reference.WeightProvider | None = None,
            retain: tuple[str, ...] = (),
            max_tensor_elements: int = 1_000_000) -> dict[str, np.ndarray]:
    if (graph.spec["family"] != "qwen3" or
            graph.execution_schedule.get("prefill_logits_policy") != "last"):
        raise ValueError("expected a Qwen3 greedy graph with last prefill logits")
    if max_tensor_elements <= 0:
        raise ValueError("max_tensor_elements must be positive")
    keep = set(graph.outputs) | set(retain)
    for name, value in inputs.items():
        if name not in graph.tensors or tuple(value.shape) != tuple(graph.tensors[name]["shape"]):
            raise ValueError(f"invalid input {name}")
    remaining = Counter(source for stage in graph.stages for source in stage["reads"])
    values = dict(inputs)
    for stage in graph.stages:
        name = stage["writes"]
        if int(np.prod(stage["shape"])) > max_tensor_elements:
            raise ValueError(f"{name}: tensor exceeds numerical control limit")
        args = [values[source] for source in stage["reads"]]
        if stage["op"] == "select_last_token":
            value = args[0][:, -1:, :].copy()
        else:
            one = Graph(graph.model, graph.spec)
            one.tensors = graph.tensors
            one.stages = [stage]
            value = reference.execute(one, dict(zip(stage["reads"], args)),
                                      max_tensor_elements=max_tensor_elements,
                                      weights=weights)[name]
        if tuple(value.shape) != tuple(stage["shape"]):
            raise ValueError(f"{name}: wrong output shape")
        values[name] = value
        for source in stage["reads"]:
            remaining[source] -= 1
            if remaining[source] == 0 and source not in keep:
                values.pop(source, None)
        if remaining[name] == 0 and name not in keep:
            values.pop(name, None)
    return {name: values[name] for name in sorted(keep)}
