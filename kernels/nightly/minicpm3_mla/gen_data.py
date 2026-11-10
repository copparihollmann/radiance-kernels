#!/usr/bin/env python3
"""Generate the bounded MiniCPM3 MLA KV-path component workload."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "minicpm3_mla_reference"))

from nightly_data import Layout
from minicpm3_mla_reference import (
    Dimensions,
    deterministic_fixture,
    run_oracle,
)


DIMS = Dimensions(
    tokens=7,
    hidden_size=64,
    kv_lora_rank=16,
    qk_rope_head_dim=8,
    num_heads=4,
    qk_nope_head_dim=16,
    v_head_dim=16,
)

hidden, parameters = deterministic_fixture(DIMS)
oracle = run_oracle(DIMS, hidden, parameters)
gold = np.asarray(oracle.flatten(), dtype=np.float32)

latent_offset = 0
rope_offset = latent_offset + DIMS.tokens * DIMS.kv_lora_rank
normalized_offset = rope_offset + DIMS.tokens * DIMS.qk_rope_head_dim
reconstructed_offset = normalized_offset + DIMS.tokens * DIMS.kv_lora_rank
assert reconstructed_offset + DIMS.tokens * DIMS.reconstruction_width == gold.size
assert gold.size == DIMS.output_elements and gold.size % 2 == 0

layout = Layout("minicpm3_mla")
layout.const("MCM3_TOKENS", f"{DIMS.tokens}u")
layout.const("MCM3_HIDDEN", f"{DIMS.hidden_size}u")
layout.const("MCM3_KV_RANK", f"{DIMS.kv_lora_rank}u")
layout.const("MCM3_ROPE_DIM", f"{DIMS.qk_rope_head_dim}u")
layout.const("MCM3_HEADS", f"{DIMS.num_heads}u")
layout.const("MCM3_NOPE_DIM", f"{DIMS.qk_nope_head_dim}u")
layout.const("MCM3_VALUE_DIM", f"{DIMS.v_head_dim}u")
layout.const("MCM3_COMPRESSION_WIDTH", f"{DIMS.compression_width}u")
layout.const("MCM3_RECONSTRUCTION_WIDTH", f"{DIMS.reconstruction_width}u")
layout.const("MCM3_LATENT_OFFSET", f"{latent_offset}u")
layout.const("MCM3_ROPE_OFFSET", f"{rope_offset}u")
layout.const("MCM3_NORMALIZED_OFFSET", f"{normalized_offset}u")
layout.const("MCM3_RECONSTRUCTED_OFFSET", f"{reconstructed_offset}u")
layout.const("MCM3_OUTPUT_ELEMENTS", f"{gold.size}u")

layout.add(
    "MCM3_HIDDEN_INPUT",
    1,
    np.asarray(hidden, dtype=np.float32),
    "synthetic FP32 hidden states [tokens][hidden]",
)
layout.add(
    "MCM3_KV_A_WEIGHT",
    1,
    np.asarray(parameters.kv_a_weight, dtype=np.float32),
    "FP32 compression projection [rank+rope][hidden]",
)
layout.add(
    "MCM3_KV_NORM_WEIGHT",
    1,
    np.asarray(parameters.kv_norm_weight, dtype=np.float32),
    "FP32 learned latent RMSNorm scale [rank]",
)
layout.add(
    "MCM3_KV_B_WEIGHT",
    1,
    np.asarray(parameters.kv_b_weight, dtype=np.float32),
    "FP32 reconstruction projection [heads*(nope+value)][rank]",
)
layout.reserve(
    "MCM3_OUTPUT",
    2,
    4 * gold.size,
    fill=0xEE,
    comment="all FP32 stage outputs, initially poisoned",
)
layout.reserve(
    "MCM3_STATUS",
    2,
    2 * gold.size,
    fill=0xEE,
    comment="one BF16 zero/pass or 1.0/fail code per output",
)
layout.add("MCM3_GOLD", 3, gold, "independent CPU-oracle FP32 stages")
layout.add(
    "MCM3_GOLD_STATUS",
    3,
    np.zeros(gold.size, dtype=np.uint16),
    "exact zero status required for every output",
)
layout.write("data", "minicpm3_mla_data.h")
