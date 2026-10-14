"""Execute OLMo 3 graph stages, including projection reshape and YaRN RoPE."""

from __future__ import annotations

import math

import numpy as np

import reference
from stitch import Graph


def yarn_frequencies(scaling: dict, theta: float, head_dim: int) -> np.ndarray:
    original = float(scaling["original_max_position_embeddings"])

    def correction(rotations: float) -> float:
        return head_dim * math.log(original / (rotations * 2.0 * math.pi)) / (
            2.0 * math.log(theta))

    low = max(math.floor(correction(float(scaling["beta_fast"]))), 0)
    high = min(math.ceil(correction(float(scaling["beta_slow"]))), head_dim - 1)
    if high == low:
        high += 0.001
    dimensions = np.arange(head_dim // 2, dtype=np.float32)
    ramp = np.clip((dimensions - low) / (high - low), 0.0, 1.0)
    power = np.arange(0, head_dim, 2, dtype=np.float32) / head_dim
    return (((1.0 - ramp) + ramp / float(scaling["factor"])) /
            (np.float32(theta) ** power)).astype(np.float32)


def execute(graph: Graph, inputs: dict[str, np.ndarray],
            *, weights: reference.WeightProvider | None = None,
            max_tensor_elements: int = 1_000_000) -> dict[str, np.ndarray]:
    if graph.spec["family"] != "olmo3":
        raise ValueError("expected an OLMo 3 graph")
    values = dict(inputs)
    for stage in graph.stages:
        name = stage["writes"]
        shape = tuple(stage["shape"])
        if int(np.prod(shape)) > max_tensor_elements:
            raise ValueError(f"{name}: tensor exceeds numerical control limit")
        args = [values[source] for source in stage["reads"]]
        if stage["op"] == "reshape_heads":
            value = args[0].reshape(shape).copy()
        elif stage["op"] == "rope_yarn":
            width = shape[-1]
            half = width // 2
            attrs = stage["attrs"]
            inverse = yarn_frequencies(attrs["scaling"], attrs["theta"], width)
            positions = np.arange(attrs["position_start"],
                                  attrs["position_start"] + shape[1],
                                  dtype=np.float32)
            angles = positions[:, None] * inverse[None, :]
            trig = np.concatenate((angles, angles), axis=-1)
            factor = np.float32(attrs["scaling"]["attention_factor"])
            cosine = (np.cos(trig).astype(np.float32) * factor)[None, :, None, :]
            sine = (np.sin(trig).astype(np.float32) * factor)[None, :, None, :]
            rotated = np.concatenate((-args[0][..., half:],
                                      args[0][..., :half]), axis=-1)
            value = args[0] * cosine + rotated * sine
        else:
            one = Graph(graph.model, graph.spec)
            one.tensors = graph.tensors
            one.stages = [stage]
            value = reference.execute(one, dict(zip(stage["reads"], args)),
                                      max_tensor_elements=max_tensor_elements,
                                      weights=weights)[name]
        if tuple(value.shape) != shape or not np.isfinite(value).all():
            raise ValueError(f"{name}: invalid output")
        values[name] = value.astype(np.int32 if graph.tensors[name]["dtype"] == "int32"
                                    else np.float32, copy=False)
    return values
