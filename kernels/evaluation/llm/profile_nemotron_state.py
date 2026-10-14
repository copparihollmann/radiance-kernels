#!/usr/bin/env python3
"""Size the pinned Nemotron hybrid decode state without loading weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from inventory import HERE, layer_kinds


SCENARIOS = (("short_b1", 1, 128), ("short_b8", 8, 128),
             ("long_b1", 1, 8192))
SOURCE_CODE = (
    "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/"
    "blob/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f/modeling_nemotron_h.py"
)


def pinned_config() -> tuple[dict, dict]:
    with (HERE / "inputs/sources.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["case"] == "nemotron3-nano-4b"]
    if len(rows) != 1:
        raise ValueError("expected one pinned Nemotron source")
    row = rows[0]
    raw = (HERE / "inputs" / row["config_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["config_sha256"]:
        raise ValueError("pinned Nemotron config hash changed")
    return json.loads(raw), row


def profile(config: dict, batch: int, context: int, ssm_bytes: int) -> dict:
    """Report resident state after prefill; SSM storage width is explicit."""
    if batch <= 0 or context <= 0 or ssm_bytes not in (2, 4):
        raise ValueError("batch/context must be positive; SSM width must be 2 or 4 bytes")
    kinds = layer_kinds(config)
    mamba_layers = kinds.count("mamba2")
    attention_layers = kinds.count("full_attention")
    if not mamba_layers or not attention_layers:
        raise ValueError("expected both Mamba and attention blocks")
    inner = int(config["mamba_num_heads"]) * int(config["mamba_head_dim"])
    if inner % int(config["n_groups"]):
        raise ValueError("Mamba groups must divide inner width")
    conv_channels = inner + 2 * int(config["n_groups"]) * int(config["ssm_state_size"])
    conv_elements = batch * mamba_layers * conv_channels * int(config["conv_kernel"])
    ssm_elements = batch * mamba_layers * inner * int(config["ssm_state_size"])
    # The revision-pinned HybridMambaAttentionDynamicCache constructor uses
    # expand*hidden_size for both initial buffers. Prefill's update methods
    # replace these with the mixer's conv_dim and head*dim*state tensors.
    initial_width = int(config["expand"]) * int(config["hidden_size"])
    initial_mamba_elements = (batch * mamba_layers * initial_width
                              * (int(config["conv_kernel"]) + int(config["ssm_state_size"])))
    kv_elements = (batch * context * attention_layers * 2
                   * int(config["num_key_value_heads"]) * int(config["head_dim"]))
    conv_bytes = conv_elements * 2
    ssm_state_bytes = ssm_elements * ssm_bytes
    kv_bytes = kv_elements * 2
    return {
        "batch": batch,
        "context_tokens": context,
        "mamba_layers": mamba_layers,
        "attention_layers": attention_layers,
        "mlp_only_layers": kinds.count("mlp_only"),
        "mamba_inner_width": inner,
        "mamba_conv_channels": conv_channels,
        "initial_cache_inner_width": initial_width,
        "initial_mamba_cache_bf16_bytes": 2 * initial_mamba_elements,
        "carried_conv_bf16_bytes": conv_bytes,
        "carried_ssm_bytes": ssm_state_bytes,
        "ssm_element_bytes": ssm_bytes,
        "attention_kv_bf16_bytes": kv_bytes,
        "total_carried_state_bytes": conv_bytes + ssm_state_bytes + kv_bytes,
        "next_token_kv_write_bf16_bytes": (
            batch * attention_layers * 2 * int(config["num_key_value_heads"])
            * int(config["head_dim"]) * 2
        ),
    }


def build() -> dict:
    config, source = pinned_config()
    cases = {}
    for name, batch, context in SCENARIOS:
        cases[name] = {"bf16_ssm": profile(config, batch, context, 2),
                       "fp32_ssm": profile(config, batch, context, 4)}
    return {
        "schema": "radiance.nemotron-hybrid-state.v1",
        "status": "analytical_capacity_only",
        "model": source["model"],
        "revision": source["revision"],
        "config_sha256": source["config_sha256"],
        "pinned_modeling_source": SOURCE_CODE,
        "layer_kinds": layer_kinds(config),
        "state_note": (
            "The pinned cache starts at expand*hidden_size width; prefill replaces "
            "Mamba conv/SSM buffers with mixer-sized states. Conv and attention KV "
            "are counted at BF16. Both BF16 and FP32 SSM storage are shown "
            "because the saved prefill state dtype depends on the execution path. "
            "The pinned NemotronHBlock source does not pass its cache to the "
            "attention mixer; KV here is the intended mapped working set, not "
            "a verified measurement of that source. These are analytical "
            "logical bytes, not measured HBM traffic."
        ),
        "pinned_attention_cache_handoff": "missing_in_block_call",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = json.dumps(build(), indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
