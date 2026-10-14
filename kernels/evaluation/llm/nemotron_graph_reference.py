"""Execute a reduced Nemotron H decoder graph with explicit Mamba state."""

from __future__ import annotations

import numpy as np

import reference
from stitch import Graph


def control_spec(pinned: dict, layers: int) -> dict:
    """Keep the pinned block prefix with the checked small Mamba dimensions."""
    if pinned["family"] != "nemotron_h" or not 1 <= layers <= len(pinned["layer_kinds"]):
        raise ValueError("invalid Nemotron H control depth")
    spec = reference.reduced_spec(pinned)
    spec.update(num_hidden_layers=layers,
                layer_kinds=spec["layer_kinds"][:layers],
                mamba_num_heads=4, mamba_head_dim=8,
                n_groups=2, ssm_state_size=4, chunk_size=8)
    return spec


def _weight(weights: reference.WeightProvider | None, name: str,
            shape: tuple[int, ...], *, scale: float = 0.05) -> np.ndarray:
    value = (weights(name, shape) if weights is not None else
             reference.parameter(name, shape, scale))
    if value.shape != shape:
        raise ValueError(f"{name}: weight shape {value.shape} != {shape}")
    return np.asarray(value, dtype=np.float32)


def _silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-value))


def _conv_state(raw: np.ndarray, previous: np.ndarray | None,
                kernel_width: int) -> np.ndarray:
    batch, _, channels = raw.shape
    state = (np.zeros((batch, channels, kernel_width), dtype=np.float32)
             if previous is None else previous.copy())
    for token in range(raw.shape[1]):
        state[:, :, :-1] = state[:, :, 1:]
        state[:, :, -1] = raw[:, token, :]
    return state


def _convolve(raw: np.ndarray, previous: np.ndarray | None,
              weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    state = (np.zeros((raw.shape[0], raw.shape[2], weight.shape[-1]),
                      dtype=np.float32) if previous is None else previous.copy())
    output = np.empty_like(raw)
    for token in range(raw.shape[1]):
        state[:, :, :-1] = state[:, :, 1:]
        state[:, :, -1] = raw[:, token, :]
        output[:, token, :] = _silu(
            np.sum(state * weight[:, 0, :][None], axis=-1) + bias[None])
    return output


def _scan(x: np.ndarray, b: np.ndarray, c: np.ndarray, raw_dt: np.ndarray,
          previous: np.ndarray | None, a_log: np.ndarray, d: np.ndarray,
          dt_bias: np.ndarray, dt_limit: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    batch, tokens, heads, width = x.shape
    groups, state_width = b.shape[2:]
    if heads % groups or b.shape != c.shape or raw_dt.shape != (batch, tokens, heads):
        raise ValueError("invalid Nemotron scan inputs")
    state = (np.zeros((batch, heads, width, state_width), dtype=np.float32)
             if previous is None else previous.copy())
    output = np.empty((batch, tokens, heads, width), dtype=np.float32)
    decay_rate = -np.exp(a_log)
    for token in range(tokens):
        dt = np.logaddexp(np.float32(0), raw_dt[:, token] + dt_bias[None])
        dt = np.clip(dt, *dt_limit)
        for head in range(heads):
            group = head // (heads // groups)
            decay = np.exp(dt[:, head] * decay_rate[head])[:, None, None]
            state[:, head] = (state[:, head] * decay +
                              (dt[:, head, None, None] *
                               x[:, token, head, :, None] *
                               b[:, token, group, None, :]))
            output[:, token, head] = (
                np.sum(state[:, head] * c[:, token, group, None, :], axis=-1)
                + x[:, token, head] * d[head])
    return output.reshape(batch, tokens, heads * width), state


def execute(graph: Graph, inputs: dict[str, np.ndarray],
            *, weights: reference.WeightProvider | None = None,
            max_tensor_elements: int = 1_000_000) -> dict[str, np.ndarray]:
    """Return every named stage output; reject controls above the size limit."""
    if graph.spec["family"] != "nemotron_h":
        raise ValueError("expected a Nemotron H graph")
    values: dict[str, np.ndarray] = {}
    for name, value in inputs.items():
        if (name not in graph.tensors or
                tuple(value.shape) != tuple(graph.tensors[name]["shape"])):
            raise ValueError(f"{name}: invalid graph input")
        values[name] = value
    paired_scan_state: dict[str, np.ndarray] = {}
    spec = graph.spec
    inner = spec["mamba_num_heads"] * spec["mamba_head_dim"]
    group_channels = spec["n_groups"] * spec["ssm_state_size"]
    conv_channels = inner + 2 * group_channels
    for stage in graph.stages:
        name, op, attrs = stage["id"], stage["op"], stage["attrs"]
        shape = tuple(stage["shape"])
        if int(np.prod(shape)) > max_tensor_elements:
            raise ValueError(f"{name}: tensor exceeds numerical control limit")
        args = [values[source] for source in stage["reads"]]
        if op == "embedding":
            ids = args[0]
            if ids.min() < 0 or ids.max() >= attrs["vocab_size"]:
                raise ValueError(f"{name}: token outside vocabulary")
            rows = [(weights.embedding_row(attrs["parameter"], int(token), shape[-1])
                     if weights is not None else
                     reference.parameter(f"{attrs['parameter']}.{int(token)}",
                                         (shape[-1],))) for token in ids.flat]
            value = np.stack(rows).reshape(shape)
        elif op == "rmsnorm":
            gamma = _weight(weights, attrs["parameter"], (shape[-1],))
            if weights is None:
                gamma = gamma + 1.0
            value = (args[0] / np.sqrt(
                np.mean(args[0] * args[0], axis=-1, keepdims=True) +
                attrs["epsilon"])) * gamma
        elif op == "mamba_projection_slice":
            start, stop = {"gate": (0, inner),
                           "conv_input": (inner, inner + conv_channels),
                           "dt": (inner + conv_channels,
                                  inner + conv_channels + spec["mamba_num_heads"])
                           }[attrs["part"]]
            value = args[0][..., start:stop].copy()
        elif op == "mamba_depthwise_conv":
            params = attrs["parameters"]
            conv_name = next(key for key in params if key.endswith(".weight"))
            bias_name = next(key for key in params if key.endswith(".bias"))
            weight = _weight(weights, conv_name, tuple(params[conv_name]))
            bias = _weight(weights, bias_name, tuple(params[bias_name]))
            value = _convolve(args[0], args[1] if len(args) == 2 else None,
                              weight, bias)
        elif op == "mamba_conv_state":
            value = _conv_state(args[0], args[1] if len(args) == 3 else None,
                                attrs["kernel_width"])
        elif op == "mamba_channel_slice":
            start, stop = {"x": (0, inner),
                           "B": (inner, inner + group_channels),
                           "C": (inner + group_channels, conv_channels)}[attrs["part"]]
            value = args[0][..., start:stop].reshape(shape).copy()
        elif op == "mamba_selective_scan":
            a_name, d_name, bias_name = attrs["parameters"]
            heads = spec["mamba_num_heads"]
            value, state = _scan(
                *args[:4], args[4] if len(args) == 5 else None,
                _weight(weights, a_name, (heads,)),
                _weight(weights, d_name, (heads,)),
                _weight(weights, bias_name, (heads,)),
                tuple(spec.get("time_step_limit", (0.0, float("inf")))))
            paired_scan_state[name] = state
        elif op == "mamba_scan_state":
            if stage["reads"][-1] != attrs["paired_scan"]:
                raise ValueError(f"{name}: missing paired scan dependency")
            value = paired_scan_state.pop(attrs["paired_scan"])
        elif op == "mamba_gated_rmsnorm":
            group = attrs["group_size"]
            if inner % group or attrs["norm_before_gate"]:
                raise ValueError(f"{name}: unsupported gated norm layout")
            gated = args[0] * _silu(args[1])
            grouped = gated.reshape(*gated.shape[:-1], inner // group, group)
            normalized = grouped / np.sqrt(
                np.mean(grouped * grouped, axis=-1, keepdims=True) +
                attrs["epsilon"])
            gamma = _weight(weights, attrs["parameter"], (inner,), scale=0.05)
            if weights is None:
                gamma = gamma + 1.0
            value = normalized.reshape(shape) * gamma
        elif op == "relu_squared":
            value = np.maximum(args[0], 0.0) ** 2
        elif op == "select_last_token":
            value = args[0][:, -1:, :].copy()
        else:
            one = Graph(graph.model, graph.spec)
            one.tensors = graph.tensors
            one.stages = [stage]
            value = reference.execute(one, dict(zip(stage["reads"], args)),
                                      max_tensor_elements=max_tensor_elements,
                                      weights=weights)[name]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name}: invalid output {value.shape}, expected {shape}")
        values[name] = value.astype(np.int32 if graph.tensors[name]["dtype"] == "int32"
                                    else np.float32, copy=False)
    return values
