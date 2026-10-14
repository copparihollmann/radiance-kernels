"""NumPy control for the sparse FFN stages emitted by the Qwen3 MoE graph.

The stage inputs and outputs follow stitch.py. Expert projections iterate over
selected experts so the control does not gather the entire expert weight bank.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


OPS = frozenset({"softmax", "topk_indices", "topk_scores", "expert_group",
                 "expert_gate_up", "expert_swiglu", "expert_down",
                 "expert_weighted_reduce"})
WeightLoader = Callable[[str, tuple[int, ...]], np.ndarray]


def execute_stage(stage: dict, args: list[np.ndarray],
                  load_weight: WeightLoader) -> np.ndarray:
    """Execute one graph MoE stage with FP32 values and explicit weights."""
    op, attrs = stage["op"], stage["attrs"]
    if op not in OPS:
        raise ValueError(f"unsupported sparse expert operation: {op}")
    if op == "softmax":
        shifted = args[0] - np.max(args[0], axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        value = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    elif op == "topk_indices":
        value = np.argsort(-args[0], axis=-1, kind="stable")[..., :attrs["k"]]
    elif op == "topk_scores":
        value = np.take_along_axis(args[0], args[1], axis=-1)
        if attrs["renormalize"]:
            value = value / np.sum(value, axis=-1, keepdims=True)
    elif op == "expert_group":
        hidden, indices = args
        if np.any(indices < 0) or np.any(indices >= attrs["expert_count"]):
            raise ValueError("selected expert index is out of range")
        value = np.broadcast_to(hidden[..., None, :], (*indices.shape, hidden.shape[-1]))
    elif op in ("expert_gate_up", "expert_down"):
        inputs, indices = args
        (name, dimensions), = attrs["parameters"].items()
        weight = load_weight(name, tuple(dimensions))
        if weight.shape != tuple(dimensions):
            raise ValueError(f"{name}: incorrect expert weight shape")
        if np.any(indices < 0) or np.any(indices >= weight.shape[0]):
            raise ValueError("selected expert index is out of range")
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        flat_indices = indices.reshape(-1)
        flat_output = np.empty((flat_indices.size, weight.shape[1]), dtype=np.float32)
        for slot, expert in enumerate(flat_indices):
            flat_output[slot] = weight[int(expert)] @ flat_inputs[slot]
        value = flat_output.reshape(*indices.shape, weight.shape[1])
    elif op == "expert_swiglu":
        gate, up = np.split(args[0], 2, axis=-1)
        value = (gate / (1.0 + np.exp(-gate))) * up
    else:  # expert_weighted_reduce
        expert_output, scores = args
        if expert_output.shape[-2] != attrs["selected_experts"]:
            raise ValueError("expert output has the wrong selected-expert count")
        value = np.sum(expert_output * scores[..., None], axis=-2,
                       dtype=np.float32)
    expected = tuple(stage["shape"])
    if value.shape != expected or not np.isfinite(value).all():
        raise ValueError(f"{stage['id']}: invalid output {value.shape}; expected {expected}")
    return value.astype(np.int32 if op == "topk_indices" else np.float32,
                        copy=False)
