#!/usr/bin/env python3
"""Bind the decoder schedule's logical parameters to Hugging Face safetensors.

Requires optional torch and safetensors packages. The checkpoint stays in its
original location; no checkpoint data is copied into radiance-kernels.
"""

from __future__ import annotations

import json
import math
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


def quantize_int4_per_channel(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pack signed [-7, 7] nibbles from row-major W[K,N], low nibble first."""
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("per-column INT4 requires a finite linear matrix")
    maxima = np.zeros(value.shape[1], dtype=np.float32)
    for start in range(0, value.shape[0], 64):
        maxima = np.maximum(maxima,
                            np.abs(value[start:start + 64]).max(axis=0))
    scales = np.where(maxima == 0.0, 1.0, maxima / 7.0).astype("<f4")
    signed = np.empty(value.shape, dtype=np.int8)
    maximum_error = 0.0
    changed = 0
    for start in range(0, value.shape[0], 64):
        stop = min(start + 64, value.shape[0])
        block = value[start:stop]
        part = np.rint(block / scales).clip(-7, 7).astype(np.int8)
        signed[start:stop] = part
        delta = np.abs(part.astype(np.float32) * scales - block)
        maximum_error = max(maximum_error, float(delta.max()))
        changed += int(np.count_nonzero(delta))
    flat = signed.reshape(-1)
    packed = np.zeros((flat.size + 1) // 2, dtype=np.uint8)
    packed[:flat[::2].size] = (flat[::2] & 0x0f).astype(np.uint8)
    packed[:flat[1::2].size] |= ((flat[1::2] & 0x0f) << 4).astype(np.uint8)
    return packed, scales, {
        "roundtrip_changed_elements": changed,
        "roundtrip_max_abs_error": maximum_error,
    }


def unpack_int4(packed: np.ndarray, count: int) -> np.ndarray:
    raw = np.empty(packed.size * 2, dtype=np.uint8)
    raw[::2] = packed & 0x0f
    raw[1::2] = packed >> 4
    signed = raw[:count].astype(np.int8)
    signed[signed >= 8] -= 16
    return signed


def quantize_int4_groupwise(value: np.ndarray, group_size: int = 32
                            ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Signed INT4 W[K,N], with a separate scale per K group and output."""
    if value.ndim != 2 or group_size <= 0 or not np.isfinite(value).all():
        raise ValueError("groupwise INT4 requires a finite linear matrix")
    k, n = value.shape
    scales = np.empty(((k + group_size - 1) // group_size, n), dtype="<f4")
    signed = np.empty(value.shape, dtype=np.int8)
    maximum_error = 0.0
    changed = 0
    for group, start in enumerate(range(0, k, group_size)):
        block = value[start:start + group_size]
        maxima = np.abs(block).max(axis=0)
        scale = np.where(maxima == 0.0, 1.0, maxima / 7.0).astype("<f4")
        scales[group] = scale
        part = np.rint(block / scale).clip(-7, 7).astype(np.int8)
        signed[start:start + group_size] = part
        delta = np.abs(part.astype(np.float32) * scale - block)
        maximum_error = max(maximum_error, float(delta.max()))
        changed += int(np.count_nonzero(delta))
    flat = signed.reshape(-1)
    packed = np.zeros((flat.size + 1) // 2, dtype=np.uint8)
    packed[:flat[::2].size] = (flat[::2] & 0x0f).astype(np.uint8)
    packed[:flat[1::2].size] |= ((flat[1::2] & 0x0f) << 4).astype(np.uint8)
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


class TiedINT8INT4GemmaWeights(QuantizedINT8Weights):
    """Reference for INT8 embedding/attention and INT4 feed-forward weights."""

    def __call__(self, logical: str, shape: tuple[int, ...]) -> np.ndarray:
        if logical == "lm_head":
            table = self.source("lm_head", shape).T
            packed, scales, _ = quantize_per_channel(table, "embedding")
            return (packed.astype(np.float32) * scales[:, None]).T
        if logical.endswith(("norm", "_bias")):
            return self.source(logical, shape)
        value = self.source(logical, shape)
        if logical.endswith(("q_proj", "k_proj", "v_proj", "o_proj")):
            packed, scales, _ = quantize_per_channel(value, "linear")
            return packed.astype(np.float32) * scales
        packed, scales, _ = quantize_int4_per_channel(value)
        signed = unpack_int4(packed, value.size).reshape(value.shape)
        return signed.astype(np.float32) * scales


class TiedINT8GroupINT4GemmaWeights(TiedINT8INT4GemmaWeights):
    """Reference for INT8 attention and group-32 INT4 feed-forward weights."""

    def __call__(self, logical: str, shape: tuple[int, ...]) -> np.ndarray:
        if not logical.endswith(("gate_proj", "up_proj", "down_proj")):
            return super().__call__(logical, shape)
        value = self.source(logical, shape)
        packed, scales, _ = quantize_int4_groupwise(value)
        signed = unpack_int4(packed, value.size).reshape(value.shape)
        result = np.empty(value.shape, dtype=np.float32)
        for group, start in enumerate(range(0, value.shape[0], 32)):
            stop = min(start + 32, value.shape[0])
            result[start:stop] = signed[start:stop].astype(np.float32) * scales[group]
        return result


def pack_signed_bits(value: np.ndarray, bits: int) -> np.ndarray:
    """Pack signed row-major values low-bit-first into a byte stream."""
    if bits not in (5, 6) or value.dtype != np.int8:
        raise ValueError("packing requires signed INT5 or INT6 values")
    flat = value.reshape(-1)
    per_group = 8 // math.gcd(8, bits)
    bytes_per_group = per_group * bits // 8
    full_groups = flat.size // per_group
    output = np.empty((flat.size * bits + 7) // 8, dtype=np.uint8)

    def write_groups(groups: np.ndarray, destination: np.ndarray) -> None:
        word = np.zeros(len(groups), dtype=np.uint64)
        for col in range(per_group):
            word |= ((groups[:, col].astype(np.uint64) & ((1 << bits) - 1))
                     << (bits * col))
        bytes_view = destination.reshape(len(groups), bytes_per_group)
        for byte in range(bytes_per_group):
            bytes_view[:, byte] = (word >> (8 * byte)).astype(np.uint8)

    groups = flat[:full_groups * per_group].reshape(full_groups, per_group)
    for start in range(0, full_groups, 1 << 17):
        stop = min(start + (1 << 17), full_groups)
        write_groups(groups[start:stop],
                     output[start * bytes_per_group:stop * bytes_per_group])
    if flat.size % per_group:
        tail = np.zeros((1, per_group), dtype=np.int8)
        tail[0, :flat.size % per_group] = flat[full_groups * per_group:]
        raw = np.empty(bytes_per_group, dtype=np.uint8)
        write_groups(tail, raw)
        output[full_groups * bytes_per_group:] = raw[:len(output) -
                                                     full_groups * bytes_per_group]
    return output


def unpack_signed_bits(packed: np.ndarray, count: int, bits: int) -> np.ndarray:
    if bits not in (5, 6) or count < 0 or len(packed) < (count * bits + 7) // 8:
        raise ValueError("invalid signed packed weight stream")
    per_group = 8 // math.gcd(8, bits)
    bytes_per_group = per_group * bits // 8
    output = np.empty(count, dtype=np.int8)
    full_groups = count // per_group
    groups = packed[:full_groups * bytes_per_group].reshape(full_groups,
                                                            bytes_per_group)
    values = output[:full_groups * per_group].reshape(full_groups, per_group)
    for start in range(0, full_groups, 1 << 17):
        stop = min(start + (1 << 17), full_groups)
        word = np.zeros(stop - start, dtype=np.uint64)
        for byte in range(bytes_per_group):
            word |= groups[start:stop, byte].astype(np.uint64) << (8 * byte)
        for col in range(per_group):
            raw = ((word >> (bits * col)) & ((1 << bits) - 1)).astype(np.int8)
            values[start:stop, col] = np.where(raw >= (1 << (bits - 1)),
                                               raw - (1 << bits), raw)
    if count % per_group:
        tail = np.zeros(bytes_per_group, dtype=np.uint8)
        remaining = packed[full_groups * bytes_per_group:]
        tail[:len(remaining)] = remaining
        word = sum(int(byte) << (8 * index) for index, byte in enumerate(tail))
        for col in range(count % per_group):
            raw = (word >> (bits * col)) & ((1 << bits) - 1)
            output[full_groups * per_group + col] = (
                raw - (1 << bits) if raw >= (1 << (bits - 1)) else raw)
    return output


def quantize_int5_group64_block(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if value.ndim != 2 or value.shape[1] % 64 or not np.isfinite(value).all():
        raise ValueError("INT5 embedding needs finite rows divisible by 64")
    block = value.reshape(value.shape[0], -1, 64)
    maxima = np.abs(block).max(axis=2)
    scales = np.where(maxima == 0, 1, maxima / 15).astype("<f4")
    quantized = np.rint(block / scales[:, :, None]).clip(-15, 15).astype(np.int8)
    return quantized.reshape(value.shape), scales


def quantize_int5_group64_embedding(value: np.ndarray
                                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    if value.ndim != 2 or value.shape[1] % 64:
        raise ValueError("INT5 embedding needs a table with width divisible by 64")
    rows, width = value.shape
    output = np.empty((value.size * 5 + 7) // 8, dtype=np.uint8)
    scales = np.empty((rows, width // 64), dtype="<f4")
    cursor = 0
    maximum_error = 0.0
    changed = 0
    for start in range(0, rows, 64):
        block = value[start:start + 64]
        quantized, part_scales = quantize_int5_group64_block(block)
        scales[start:start + len(block)] = part_scales
        raw = pack_signed_bits(quantized, 5)
        output[cursor:cursor + len(raw)] = raw
        cursor += len(raw)
        restored = (quantized.reshape(len(block), -1, 64).astype(np.float32) *
                    part_scales[:, :, None]).reshape(block.shape)
        delta = np.abs(restored - block)
        maximum_error = max(maximum_error, float(delta.max()))
        changed += int(np.count_nonzero(delta))
    if cursor != len(output):
        raise ValueError("INT5 embedding packing did not fill its image")
    return output, scales.reshape(-1), {
        "roundtrip_changed_elements": changed,
        "roundtrip_max_abs_error": maximum_error,
    }


def quantize_int6_per_channel(value: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray, dict]:
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("per-column INT6 requires a finite linear matrix")
    maxima = np.zeros(value.shape[1], dtype=np.float32)
    for start in range(0, value.shape[0], 64):
        maxima = np.maximum(maxima,
                            np.abs(value[start:start + 64]).max(axis=0))
    scales = np.where(maxima == 0, 1, maxima / 31).astype("<f4")
    output = np.empty((value.size * 6 + 7) // 8, dtype=np.uint8)
    cursor = 0
    maximum_error = 0.0
    changed = 0
    for start in range(0, value.shape[0], 64):
        block = value[start:start + 64]
        quantized = np.rint(block / scales).clip(-31, 31).astype(np.int8)
        raw = pack_signed_bits(quantized, 6)
        output[cursor:cursor + len(raw)] = raw
        cursor += len(raw)
        delta = np.abs(quantized.astype(np.float32) * scales - block)
        maximum_error = max(maximum_error, float(delta.max()))
        changed += int(np.count_nonzero(delta))
    if cursor != len(output):
        raise ValueError("INT6 linear packing did not fill its image")
    return output, scales, {
        "roundtrip_changed_elements": changed,
        "roundtrip_max_abs_error": maximum_error,
    }


class TiedINT5INT6GemmaWeights(QuantizedINT8Weights):
    """Numerical reference for grouped INT5 embedding and INT6 FFN."""

    def __call__(self, logical: str, shape: tuple[int, ...]) -> np.ndarray:
        if logical == "lm_head":
            table = self.source(logical, shape).T
            result = np.empty(table.shape, dtype=np.float32)
            for start in range(0, len(table), 64):
                block = table[start:start + 64]
                quantized, scales = quantize_int5_group64_block(block)
                result[start:start + len(block)] = (
                    quantized.reshape(len(block), -1, 64).astype(np.float32) *
                    scales[:, :, None]).reshape(block.shape)
            return result.T
        if logical.endswith(("gate_proj", "up_proj", "down_proj")):
            value = self.source(logical, shape)
            packed, scales, _ = quantize_int6_per_channel(value)
            signed = unpack_signed_bits(packed, value.size, 6).reshape(value.shape)
            return signed.astype(np.float32) * scales
        return super().__call__(logical, shape)

    def embedding_row(self, logical: str, row: int, width: int) -> np.ndarray:
        value = self.source.embedding_row(logical, row, width)[None, :]
        quantized, scales = quantize_int5_group64_block(value)
        return (quantized.reshape(1, -1, 64).astype(np.float32) *
                scales[:, :, None]).reshape(width)
