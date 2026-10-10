#!/usr/bin/env python3
"""Bind the decoder schedule's logical parameters to Hugging Face safetensors.

Requires optional torch and safetensors packages. The checkpoint stays in its
original location; no checkpoint data is copied into radiance-kernels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class SafeTensorWeights:
    def __init__(self, path: Path, family: str):
        if family not in ("llama", "qwen2", "gemma2"):
            raise ValueError(f"unsupported checkpoint family {family}")
        try:
            from safetensors import safe_open
            import torch
        except ImportError as error:
            raise RuntimeError("checkpoint binding requires safetensors and torch") from error
        self.path = Path(path)
        self.family = family
        self.safe_open = safe_open
        self.torch = torch
        if self.path.is_dir():
            index = json.loads((self.path / "model.safetensors.index.json").read_text())
            self._key_to_file = {}
            for key, filename in index["weight_map"].items():
                if Path(filename).name != filename or not (self.path / filename).is_file():
                    raise ValueError(f"invalid checkpoint shard for {key}")
                self._key_to_file[key] = self.path / filename
            self.keys = set(self._key_to_file)
        else:
            with self.safe_open(str(self.path), framework="pt", device="cpu") as file:
                self.keys = set(file.keys())
            self._key_to_file = {key: self.path for key in self.keys}
        self._cached_name: str | None = None
        self._cached_value: np.ndarray | None = None

    def key(self, logical: str) -> str:
        if logical in ("embed_tokens", "final_norm", "lm_head"):
            key = {"embed_tokens": "model.embed_tokens.weight",
                   "final_norm": "model.norm.weight",
                   "lm_head": "lm_head.weight"}[logical]
            if logical == "lm_head" and self.family == "gemma2" and key not in self.keys:
                return "model.embed_tokens.weight"  # tied output embedding
            return key
        parts = logical.split(".")
        if len(parts) != 3 or parts[0] != "layers":
            raise KeyError(logical)
        index, part = int(parts[1]), parts[2]
        norm_parts = {
            "attn_norm": "input_layernorm",
            "ffn_norm": "pre_feedforward_layernorm" if self.family == "gemma2"
                        else "post_attention_layernorm",
            "post_attn_norm": "post_attention_layernorm",
            "post_ffn_norm": "post_feedforward_layernorm",
        }
        if part in norm_parts:
            return f"model.layers.{index}.{norm_parts[part]}.weight"
        if part.endswith("_bias"):
            return f"model.layers.{index}.self_attn.{part[0]}_proj.bias"
        if part in ("q_proj", "k_proj", "v_proj", "o_proj"):
            return f"model.layers.{index}.self_attn.{part}.weight"
        if part in ("gate_proj", "up_proj", "down_proj"):
            return f"model.layers.{index}.mlp.{part}.weight"
        raise KeyError(logical)

    def _tensor(self, key: str) -> np.ndarray:
        if key not in self.keys:
            raise KeyError(f"missing checkpoint tensor {key}")
        if self._cached_name == key:
            return self._cached_value
        with self.safe_open(str(self._key_to_file[key]), framework="pt", device="cpu") as file:
            value = file.get_tensor(key).to(dtype=self.torch.float32).numpy()
        self._cached_name, self._cached_value = key, value
        return value

    def __call__(self, logical: str, shape: tuple[int, ...]) -> np.ndarray:
        key = self.key(logical)
        value = self._tensor(key)
        # PyTorch linear weights are [out, in]; the graph uses [in, out].
        if logical.endswith("_bias"):
            value = value.reshape(shape)
        elif len(shape) == 2:
            value = value.T
        if value.shape != shape:
            raise ValueError(f"{key}: got {value.shape}, expected {shape}")
        return value

    def embedding_row(self, logical: str, row: int, width: int) -> np.ndarray:
        key = self.key(logical)
        value = self._tensor(key)
        if row < 0 or row >= value.shape[0] or value.shape[1] != width:
            raise ValueError(f"{key}: invalid row {row} or width {width}")
        return value[row]

    def embedding_table(self, logical: str, vocab: int, width: int) -> np.ndarray:
        key = self.key(logical)
        value = self._tensor(key)
        if value.shape != (vocab, width):
            raise ValueError(f"{key}: got {value.shape}, expected {(vocab, width)}")
        return value

    def check_bindings(self, graph) -> dict:
        bound = set()
        for stage in graph.stages:
            logical = stage["attrs"].get("parameter")
            if logical is not None:
                key = self.key(logical)
                if key not in self.keys:
                    raise KeyError(f"missing checkpoint tensor {key}")
                bound.add(key)
        return {"checkpoint": str(self.path), "bound_tensors": len(bound),
                "missing_tensors": 0}


class MixedFP16Weights:
    """Numerical reference for the device's FP16 linear/embedding storage."""

    def __init__(self, source: SafeTensorWeights):
        self.source = source

    def __call__(self, logical: str, shape: tuple[int, ...]) -> np.ndarray:
        value = self.source(logical, shape)
        if logical.endswith(("norm", "_bias")):
            return value
        return value.astype(np.float16).astype(np.float32)

    def embedding_row(self, logical: str, row: int, width: int) -> np.ndarray:
        value = self.source.embedding_row(logical, row, width)
        return value.astype(np.float16).astype(np.float32)

    def key(self, logical: str) -> str:
        return self.source.key(logical)


def quantize_per_channel(value: np.ndarray, operation: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Symmetric signed INT8 with a separate FP32 scale per output/embedding row."""
    if operation not in ("linear", "embedding") or value.ndim != 2:
        raise ValueError("per-channel INT8 requires a linear or embedding matrix")
    if not np.isfinite(value).all():
        raise ValueError("checkpoint contains nonfinite weights")
    maxima = np.zeros(value.shape[1 if operation == "linear" else 0], dtype=np.float32)
    for start in range(0, value.shape[0], 64):
        block = value[start:start + 64]
        absolute = np.abs(block)
        if operation == "linear":
            maxima = np.maximum(maxima, absolute.max(axis=0))
        else:
            maxima[start:start + len(block)] = absolute.max(axis=1)
    scales = np.where(maxima == 0.0, 1.0, maxima / 127.0).astype("<f4")
    packed = np.empty(value.shape, dtype=np.int8)
    maximum_error = 0.0
    changed = 0
    for start in range(0, value.shape[0], 64):
        stop = min(start + 64, value.shape[0])
        block = value[start:stop]
        scale = scales if operation == "linear" else scales[start:stop, None]
        part = np.rint(block / scale).clip(-127, 127).astype(np.int8)
        packed[start:stop] = part
        delta = np.abs(part.astype(np.float32) * scale - block)
        maximum_error = max(maximum_error, float(delta.max()))
        changed += int(np.count_nonzero(delta))
    return packed, scales, {
        "roundtrip_changed_elements": changed,
        "roundtrip_max_abs_error": maximum_error,
    }


class QuantizedINT8Weights:
    """Numerical reference for the device's per-channel INT8 checkpoint image."""

    def __init__(self, source: SafeTensorWeights):
        self.source = source

    def __call__(self, logical: str, shape: tuple[int, ...]) -> np.ndarray:
        value = self.source(logical, shape)
        if logical.endswith(("norm", "_bias")):
            return value
        packed, scales, _ = quantize_per_channel(value, "linear")
        return packed.astype(np.float32) * scales

    def embedding_row(self, logical: str, row: int, width: int) -> np.ndarray:
        value = self.source.embedding_row(logical, row, width)
        maximum = float(np.max(np.abs(value)))
        scale = np.float32(maximum / 127.0 if maximum else 1.0)
        packed = np.rint(value / scale).clip(-127, 127).astype(np.int8)
        return packed.astype(np.float32) * scale

    def key(self, logical: str) -> str:
        return self.source.key(logical)


class TiedFP16INT8GemmaWeights(QuantizedINT8Weights):
    """Gemma INT8 body with a shared FP16 input/output embedding table."""

    def __call__(self, logical: str, shape: tuple[int, ...]) -> np.ndarray:
        if logical == "lm_head":
            return self.source(logical, shape).astype(np.float16).astype(np.float32)
        return super().__call__(logical, shape)

    def embedding_row(self, logical: str, row: int, width: int) -> np.ndarray:
        return self.source.embedding_row(logical, row, width).astype(np.float16).astype(np.float32)
