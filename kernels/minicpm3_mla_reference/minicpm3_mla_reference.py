#!/usr/bin/env python3
"""Independent deterministic oracle for MiniCPM3's KV compression path.

This is intentionally a small, dependency-free mathematical reference.  It
does not import Transformers, PyTorch, the Radiance component implementation,
or checkpoint weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Sequence


OFFICIAL_REVISION = "0a2cd817886710fff15d181edeae4c249443ed67"


@dataclass(frozen=True)
class OfficialConfig:
    vocab_size: int = 73_448
    hidden_size: int = 2_560
    intermediate_size: int = 6_400
    num_hidden_layers: int = 62
    num_attention_heads: int = 40
    q_lora_rank: int = 768
    kv_lora_rank: int = 256
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64


def official_parameter_count(config: OfficialConfig = OfficialConfig()) -> int:
    """Count unique learned scalars for the pinned, tied-embedding config."""
    h = config.hidden_size
    heads = config.num_attention_heads
    q_width = config.qk_nope_head_dim + config.qk_rope_head_dim
    kv_width = config.qk_nope_head_dim + config.v_head_dim
    attention = (
        h * config.q_lora_rank
        + config.q_lora_rank  # q_a RMSNorm
        + config.q_lora_rank * heads * q_width
        + h * (config.kv_lora_rank + config.qk_rope_head_dim)
        + config.kv_lora_rank  # kv_a RMSNorm
        + config.kv_lora_rank * heads * kv_width
        + heads * config.v_head_dim * h
    )
    mlp = 3 * h * config.intermediate_size
    decoder_norms = 2 * h
    per_layer = attention + mlp + decoder_norms
    tied_embedding_and_head = config.vocab_size * h
    final_norm = h
    return (
        tied_embedding_and_head
        + config.num_hidden_layers * per_layer
        + final_norm
    )


@dataclass(frozen=True)
class Dimensions:
    tokens: int
    hidden_size: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    num_heads: int
    qk_nope_head_dim: int
    v_head_dim: int

    def validate(self) -> None:
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def compression_width(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def reconstruction_width(self) -> int:
        return self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)

    @property
    def output_elements(self) -> int:
        return self.tokens * (
            self.kv_lora_rank
            + self.qk_rope_head_dim
            + self.kv_lora_rank
            + self.reconstruction_width
        )


@dataclass(frozen=True)
class Parameters:
    # Row-major [kv_lora_rank + qk_rope_head_dim][hidden_size].
    kv_a_weight: list[float]
    # Learned MiniCPM RMSNorm scale [kv_lora_rank].
    kv_norm_weight: list[float]
    # Row-major [num_heads * (qk_nope_head_dim + v_head_dim)][kv_lora_rank].
    kv_b_weight: list[float]


@dataclass(frozen=True)
class OracleResult:
    latent: list[float]
    rope_key: list[float]
    normalized_latent: list[float]
    reconstructed_kv: list[float]

    def flatten(self) -> list[float]:
        return (
            self.latent
            + self.rope_key
            + self.normalized_latent
            + self.reconstructed_kv
        )


def f32(value: float) -> float:
    """Round one scalar to IEEE binary32, independent of NumPy."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


class Lcg:
    """Specified 32-bit LCG, used only to make bounded synthetic inputs."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def value(self, scale: float, offset: float = 0.0) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        unit = ((self.state >> 8) & 0x00FFFFFF) / float(0x01000000)
        return f32(offset + scale * (2.0 * unit - 1.0))

    def values(self, count: int, scale: float, offset: float = 0.0) -> list[float]:
        return [self.value(scale, offset) for _ in range(count)]


def deterministic_fixture(
    dims: Dimensions, seed: int = 0x4D4C4133
) -> tuple[list[float], Parameters]:
    dims.validate()
    rng = Lcg(seed)
    hidden = rng.values(dims.tokens * dims.hidden_size, 0.50)
    kv_a_weight = rng.values(
        dims.compression_width * dims.hidden_size, 0.125
    )
    kv_norm_weight = rng.values(dims.kv_lora_rank, 0.10, 1.0)
    kv_b_weight = rng.values(
        dims.reconstruction_width * dims.kv_lora_rank, 0.15
    )
    return hidden, Parameters(kv_a_weight, kv_norm_weight, kv_b_weight)


def _check_lengths(
    dims: Dimensions, hidden: Sequence[float], parameters: Parameters
) -> None:
    expected = {
        "hidden": dims.tokens * dims.hidden_size,
        "kv_a_weight": dims.compression_width * dims.hidden_size,
        "kv_norm_weight": dims.kv_lora_rank,
        "kv_b_weight": dims.reconstruction_width * dims.kv_lora_rank,
    }
    actual = {
        "hidden": len(hidden),
        "kv_a_weight": len(parameters.kv_a_weight),
        "kv_norm_weight": len(parameters.kv_norm_weight),
        "kv_b_weight": len(parameters.kv_b_weight),
    }
    for name in expected:
        if actual[name] != expected[name]:
            raise ValueError(
                f"{name} has {actual[name]} elements, expected {expected[name]}"
            )


def run_oracle(
    dims: Dimensions,
    hidden: Sequence[float],
    parameters: Parameters,
    rms_norm_epsilon: float = 1.0e-5,
) -> OracleResult:
    """Run compression, latent RMSNorm, and per-head K/V reconstruction.

    Dot products and reductions use ``math.fsum`` in binary64.  Each public
    stage result is rounded to binary32 before it feeds the next stage.  This
    is deliberately different from the component kernel's sequential FP32
    reductions and provides a non-tautological numerical oracle.
    """
    dims.validate()
    _check_lengths(dims, hidden, parameters)
    if rms_norm_epsilon <= 0.0:
        raise ValueError("rms_norm_epsilon must be positive")

    latent = [0.0] * (dims.tokens * dims.kv_lora_rank)
    rope_key = [0.0] * (dims.tokens * dims.qk_rope_head_dim)
    normalized = [0.0] * len(latent)
    reconstructed = [0.0] * (dims.tokens * dims.reconstruction_width)

    for token in range(dims.tokens):
        hidden_base = token * dims.hidden_size
        for output in range(dims.compression_width):
            weight_base = output * dims.hidden_size
            projected = f32(
                math.fsum(
                    float(hidden[hidden_base + column])
                    * float(parameters.kv_a_weight[weight_base + column])
                    for column in range(dims.hidden_size)
                )
            )
            if output < dims.kv_lora_rank:
                latent[token * dims.kv_lora_rank + output] = projected
            else:
                rope_key[
                    token * dims.qk_rope_head_dim
                    + output
                    - dims.kv_lora_rank
                ] = projected

        latent_base = token * dims.kv_lora_rank
        mean_square = math.fsum(
            float(latent[latent_base + column]) ** 2
            for column in range(dims.kv_lora_rank)
        ) / dims.kv_lora_rank
        inverse_rms = 1.0 / math.sqrt(mean_square + rms_norm_epsilon)
        for column in range(dims.kv_lora_rank):
            normalized[latent_base + column] = f32(
                latent[latent_base + column]
                * inverse_rms
                * parameters.kv_norm_weight[column]
            )

        reconstruction_base = token * dims.reconstruction_width
        for output in range(dims.reconstruction_width):
            weight_base = output * dims.kv_lora_rank
            reconstructed[reconstruction_base + output] = f32(
                math.fsum(
                    float(normalized[latent_base + column])
                    * float(parameters.kv_b_weight[weight_base + column])
                    for column in range(dims.kv_lora_rank)
                )
            )

    return OracleResult(latent, rope_key, normalized, reconstructed)


def split_reconstructed_head(
    dims: Dimensions, result: OracleResult, token: int, head: int
) -> tuple[list[float], list[float]]:
    if not (0 <= token < dims.tokens and 0 <= head < dims.num_heads):
        raise IndexError("token or head out of range")
    per_head = dims.qk_nope_head_dim + dims.v_head_dim
    begin = token * dims.reconstruction_width + head * per_head
    middle = begin + dims.qk_nope_head_dim
    end = middle + dims.v_head_dim
    return result.reconstructed_kv[begin:middle], result.reconstructed_kv[middle:end]


if __name__ == "__main__":
    config = OfficialConfig()
    print(f"official_revision={OFFICIAL_REVISION}")
    print(f"unique_parameters={official_parameter_count(config)}")
