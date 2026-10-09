#!/usr/bin/env python3
"""Bind the decoder schedule's logical parameters to Hugging Face safetensors.

Requires optional torch and safetensors packages. The checkpoint stays in its
original location; no checkpoint data is copied into radiance-kernels.
"""

from __future__ import annotations

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
        with self.safe_open(str(self.path), framework="pt", device="cpu") as file:
            self.keys = set(file.keys())
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
        with self.safe_open(str(self.path), framework="pt", device="cpu") as file:
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
