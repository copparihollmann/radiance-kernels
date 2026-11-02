#!/usr/bin/env python3
"""Granite 3.1 MoE routing data from the shared, validated CPU oracle."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "granite_moe_reference"))
from nightly_data import Layout
from granite_moe_reference import (
    NUM_ASSIGNMENTS,
    NUM_EXPERTS,
    NUM_TOKENS,
    TOP_K,
    build_oracle,
    deterministic_logits,
)

oracle = build_oracle()
logits = np.asarray(deterministic_logits(), dtype=np.float32)
top = np.asarray(oracle.top_indices, dtype=np.uint32).reshape(-1)
counts = np.asarray(oracle.expert_counts, dtype=np.uint32)
offsets = np.asarray(oracle.expert_offsets, dtype=np.uint32)
packed = np.asarray(
    [assignment.assignment_id for assignment in oracle.packed_assignments],
    dtype=np.uint32,
)
inverse = np.asarray(oracle.assignment_to_packed, dtype=np.uint32)
gold = np.concatenate((top, counts, offsets, packed, inverse))

top_word = 0
counts_word = top_word + top.size
offsets_word = counts_word + counts.size
packed_word = offsets_word + offsets.size
inverse_word = packed_word + packed.size
assert inverse_word + inverse.size == gold.size

layout = Layout("granite_moe_route")
layout.const("GMR_TOKENS", f"{NUM_TOKENS}u")
layout.const("GMR_EXPERTS", f"{NUM_EXPERTS}u")
layout.const("GMR_TOP_K", f"{TOP_K}u")
layout.const("GMR_ASSIGNMENTS", f"{NUM_ASSIGNMENTS}u")
layout.const("GMR_TOP_WORD", f"{top_word}u")
layout.const("GMR_COUNTS_WORD", f"{counts_word}u")
layout.const("GMR_OFFSETS_WORD", f"{offsets_word}u")
layout.const("GMR_PACKED_WORD", f"{packed_word}u")
layout.const("GMR_INVERSE_WORD", f"{inverse_word}u")
layout.const("GMR_OUTPUT_WORDS", f"{gold.size}u")
layout.add("GMR_LOGITS", 1, logits, "synthetic FP32 router logits [tokens][experts]")
layout.reserve(
    "GMR_OUTPUT",
    2,
    4 * gold.size,
    fill=0xEE,
    comment="top-k, counts, offsets, packed IDs and inverse map (poisoned)",
)
layout.add("GMR_GOLD", 3, gold, "exact uint32 routing-structure oracle")
layout.write("data", "granite_moe_route_data.h")
