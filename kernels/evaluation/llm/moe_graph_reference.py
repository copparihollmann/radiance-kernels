"""Low-memory synthetic numerical control for a Qwen3 MoE decoder graph.

This composes the existing dense NumPy stage evaluator with the sparse expert
stages. It is a graph-order control, not a checkpoint or device execution.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

import moe_ops
import reference
from stitch import Graph


def execute(graph: Graph, inputs: dict[str, np.ndarray],
            *, retain: tuple[str, ...] = (),
            max_tensor_elements: int = 1_000_000) -> dict[str, np.ndarray]:
    if graph.spec["family"] != "qwen3_moe":
        raise ValueError("this executor requires a Qwen3 MoE graph")
    if max_tensor_elements <= 0:
        raise ValueError("max_tensor_elements must be positive")
    keep = set(graph.outputs) | set(retain)
    for name, value in inputs.items():
        if name not in graph.tensors or tuple(value.shape) != tuple(graph.tensors[name]["shape"]):
            raise ValueError(f"invalid input {name}")
    for name in keep:
        if name not in graph.tensors:
            raise ValueError(f"unknown retained tensor {name}")
    remaining = Counter(source for stage in graph.stages for source in stage["reads"])
    values = dict(inputs)
    def load_weight(name: str, shape: tuple[int, ...]) -> np.ndarray:
        if int(np.prod(shape)) > max_tensor_elements:
            raise ValueError(f"{name}: weight exceeds numerical control limit")
        return reference.parameter(name, shape)
    for stage in graph.stages:
        name = stage["writes"]
        if int(np.prod(stage["shape"])) > max_tensor_elements:
            raise ValueError(f"{name}: tensor exceeds numerical control limit")
        args = [values[source] for source in stage["reads"]]
        if stage["op"] in moe_ops.OPS:
            value = moe_ops.execute_stage(stage, args, load_weight)
        else:
            one = Graph(graph.model, graph.spec)
            one.tensors = graph.tensors
            one.stages = [stage]
            stage_inputs = dict(zip(stage["reads"], args))
            value = reference.execute(one, stage_inputs,
                                      max_tensor_elements=max_tensor_elements)[name]
        values[name] = value
        for source in stage["reads"]:
            remaining[source] -= 1
            if remaining[source] == 0 and source not in keep:
                values.pop(source, None)
        if remaining[name] == 0 and name not in keep:
            values.pop(name, None)
    return {name: values[name] for name in sorted(keep)}
