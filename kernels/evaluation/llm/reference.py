#!/usr/bin/env python3
"""Small deterministic numerical control for the PR #1 decoder schedules.

The weights are generated for this control. They are not checkpoint weights,
and this NumPy path is not a Radiance timing or precision model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from stitch import Graph, build, model_specs


def parameter(name: str, shape: tuple[int, ...], scale: float = 0.05) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    return (np.random.default_rng(seed).standard_normal(shape) * scale).astype(np.float32)


class WeightProvider(Protocol):
    def __call__(self, name: str, shape: tuple[int, ...]) -> np.ndarray: ...
    def embedding_row(self, name: str, row: int, width: int) -> np.ndarray: ...


def rope(x: np.ndarray, start: int) -> np.ndarray:
    half = x.shape[-1] // 2
    frequencies = 10000.0 ** (-np.arange(half, dtype=np.float32) / half)
    angle = np.arange(start, start + x.shape[1], dtype=np.float32)[:, None] * frequencies
    cosine, sine = np.cos(angle)[None, :, None, :], np.sin(angle)[None, :, None, :]
    left, right = x[..., :half], x[..., half:]
    return np.concatenate((left * cosine - right * sine,
                           right * cosine + left * sine), axis=-1)


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, attrs: dict) -> np.ndarray:
    batch, queries, q_heads, width = q.shape
    kv_heads = k.shape[2]
    if q_heads % kv_heads or k.shape != v.shape:
        raise ValueError("invalid grouped-query cache")
    result = np.empty_like(q)
    query_positions = attrs["query_start"] + np.arange(queries)
    key_positions = np.arange(k.shape[1])
    allowed = key_positions[None, :] <= query_positions[:, None]
    if attrs["window"] is not None:
        allowed &= key_positions[None, :] > (query_positions[:, None] - attrs["window"])
    for head in range(q_heads):
        kv = head // (q_heads // kv_heads)
        scores = np.einsum("bsd,btd->bst", q[:, :, head], k[:, :, kv]) / np.sqrt(width)
        if attrs["softcap"] is not None:
            cap = attrs["softcap"]
            scores = cap * np.tanh(scores / cap)
        scores = np.where(allowed[None], scores, -1.0e30)
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        result[:, :, head] = np.einsum("bst,btd->bsd", weights, v[:, :, kv])
    return result.reshape(batch, queries, q_heads * width)


def execute(graph: Graph, inputs: dict[str, np.ndarray],
            max_tensor_elements: int = 1_000_000,
            weights: WeightProvider | None = None) -> dict[str, np.ndarray]:
    """Execute a decoder graph using shared NumPy tensors.

    The default generated weights are for small controls. An explicit provider
    may bind real checkpoint tensors; this path is still not MX/Radiance math.
    """
    if graph.spec["family"] == "smolvla":
        raise ValueError("SmolVLA's vision and action stages lack a numerical backend")
    values: dict[str, np.ndarray] = {}
    for name, value in inputs.items():
        if name not in graph.tensors or tuple(value.shape) != tuple(graph.tensors[name]["shape"]):
            raise ValueError(f"invalid input {name}")
        values[name] = value
    for stage in graph.stages:
        name, op, attrs = stage["id"], stage["op"], stage["attrs"]
        shape = tuple(stage["shape"])
        if np.prod(shape, dtype=np.int64) > max_tensor_elements:
            raise ValueError(f"{name}: numerical control is limited to small shapes")
        args = [values[source] for source in stage["reads"]]
        if op == "token_select":
            value = np.argmax(args[0][:, -1, :], axis=-1).astype(np.int32)[:, None]
        elif op == "embedding":
            ids = args[0]
            if ids.min() < 0 or ids.max() >= attrs["vocab_size"]:
                raise ValueError("token id outside vocabulary")
            rows = [weights.embedding_row(attrs["parameter"], int(token), shape[-1])
                    if weights is not None else
                    parameter(f"{attrs['parameter']}.{int(token)}", (shape[-1],))
                    for token in ids.flat]
            value = np.stack(rows).reshape(shape) * attrs["scale"]
        elif op == "rmsnorm":
            if weights is not None:
                gamma = weights(attrs["parameter"], (shape[-1],))
                if attrs["gemma_weight_offset"]:
                    gamma = gamma + 1.0
            else:
                gamma = parameter(attrs["parameter"], (shape[-1],)) + 1.0
            value = args[0] / np.sqrt(np.mean(args[0] ** 2, axis=-1,
                                               keepdims=True) + attrs["epsilon"]) * gamma
        elif op == "linear":
            weight_shape = tuple(attrs["weight_shape"])
            weight = (weights(attrs["parameter"], weight_shape) if weights is not None
                      else parameter(attrs["parameter"], weight_shape,
                                     1.0 / np.sqrt(weight_shape[0])))
            value = (args[0] @ weight).reshape(shape)
        elif op == "bias_add":
            bias = (weights(attrs["parameter"], shape[2:]) if weights is not None
                    else parameter(attrs["parameter"], shape[2:]))
            value = args[0] + bias
        elif op == "rope":
            value = rope(args[0], attrs["position_start"])
        elif op == "kv_append":
            value = np.concatenate(args, axis=1) if len(args) == 2 else args[0]
        elif op == "causal_gqa":
            value = attention(*args, attrs)
        elif op == "gated_activation":
            gate, up = args
            if attrs["function"] == "silu":
                value = (gate / (1.0 + np.exp(-gate))) * up
            elif attrs["function"] == "gelu_tanh":
                value = (0.5 * gate * (1.0 + np.tanh(
                    np.sqrt(2.0 / np.pi) * (gate + 0.044715 * gate ** 3)))) * up
            else:
                raise ValueError(attrs["function"])
        elif op == "add":
            value = args[0] + args[1]
        elif op == "softcap":
            value = attrs["cap"] * np.tanh(args[0] / attrs["cap"])
        else:
            raise ValueError(f"no numerical implementation for {op}")
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name}: invalid output {value.shape}, expected {shape}")
        target_dtype = np.int32 if graph.tensors[name]["dtype"] == "int32" else np.float32
        values[name] = value.astype(target_dtype, copy=False)
    return values


def reduced_spec(spec: dict) -> dict:
    """Preserve operation topology and GQA ratio while shrinking the control."""
    result = dict(spec)
    ratio = spec["num_attention_heads"] // spec["num_key_value_heads"]
    result.update(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=ratio, num_key_value_heads=1,
                  head_dim=8, vocab_size=64)
    if spec["family"] == "gemma2":
        result["sliding_window"] = 2
    return result


def check_small(model: str, layers: int = 2) -> dict:
    """A prefill followed by two cached decode steps must equal full prefill."""
    if layers <= 0:
        raise ValueError("layers must be positive")
    spec = reduced_spec(model_specs()[model])
    if spec["family"] == "smolvla":
        raise ValueError("SmolVLA needs a separate vision/action reference")
    spec["num_hidden_layers"] = layers
    graph = build(model, prefill=3, decode_steps=2, specs={model: spec})
    token_ids = np.arange(1, 6, dtype=np.int32)
    inputs = {"prefill.token_ids": token_ids[:3][None],
              "decode0.token_ids": token_ids[3:4][None],
              "decode1.token_ids": token_ids[4:5][None]}
    incremental = execute(graph, inputs)[graph.outputs[-1]]
    full = build(model, prefill=5, decode_steps=0, specs={model: spec})
    full_logits = execute(full, {"prefill.token_ids": token_ids[None]})[full.outputs[0]][:, -1:]
    maximum_error = float(np.max(np.abs(incremental - full_logits)))
    if not np.allclose(incremental, full_logits, rtol=1e-4, atol=1e-5):
        raise AssertionError(f"{model}: cached decode differs from full causal pass")
    changed_inputs = dict(inputs)
    changed_inputs["decode1.token_ids"] = token_ids[4:5][None] + 1
    changed_logits = execute(graph, changed_inputs)[graph.outputs[-1]]
    changed_token_delta = float(np.max(np.abs(incremental - changed_logits)))
    if changed_token_delta <= 0:
        raise AssertionError(f"{model}: final logits ignored the changed decode token")
    return {"model": model, "layers": layers, "prefill_tokens": 3,
            "decode_steps": 2, "graph_stages": len(graph.stages),
            "control": "synthetic_small_cached_decode_vs_full_prefill",
            "passed": True, "maximum_absolute_error": maximum_error,
            "changed_decode_token_max_abs_delta": changed_token_delta,
            "output_sha256": hashlib.sha256(incremental.tobytes()).hexdigest(),
            "device_execution": False, "checkpoint_weights": False,
            "original_model_dimensions": False,
            "upstream_execution_equivalent": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=model_specs(), required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = check_small(args.model, args.layers)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
