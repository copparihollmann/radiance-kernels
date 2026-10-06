#!/usr/bin/env python3
"""Compute analytical BF16 KV-cache sizes from pinned model configs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
CONDITIONS = (("short_b1", 1, 256), ("short_b8", 8, 256),
              ("long_b1", 1, 8192))


def attention_windows(config: dict) -> list[int | None]:
    """Return one entry per attention layer; None means full context."""
    if "layer_types" in config:
        assert len(config["layer_types"]) == config["num_hidden_layers"]
        windows = []
        for kind in config["layer_types"]:
            if kind == "full_attention":
                windows.append(None)
            elif kind == "sliding_attention":
                windows.append(int(config["sliding_window"]))
            else:
                raise ValueError(f"unknown attention layer type: {kind}")
        return windows
    if "hybrid_override_pattern" in config:
        pattern = config["hybrid_override_pattern"]
        assert len(pattern) == config["num_hidden_layers"]
        if set(pattern) - {"M", "*", "-"}:
            raise ValueError("unknown hybrid layer type")
        return [None for layer in pattern if layer == "*"]
    return [None] * int(config["num_hidden_layers"])


def kv_sizes(config: dict, batch: int, context: int) -> tuple[int, int, int, int]:
    dtype = config.get("dtype", config.get("torch_dtype"))
    if dtype != "bfloat16":
        raise ValueError(f"only BF16 checkpoint KV values are modeled, got {dtype}")
    heads = int(config["num_attention_heads"])
    hidden = int(config["hidden_size"])
    head_dim = int(config.get("head_dim", hidden // heads))
    if head_dim <= 0 or ("head_dim" not in config and hidden % heads):
        raise ValueError("invalid attention head dimension")
    # K and V, each with num_key_value_heads * head_dim BF16 elements.
    bytes_per_token_layer = 2 * int(config["num_key_value_heads"]) * head_dim * 2
    windows = attention_windows(config)
    per_layer = [batch * min(context, window or context) * bytes_per_token_layer
                 for window in windows]
    return (len(windows), sum(per_layer), max(per_layer, default=0),
            batch * len(windows) * bytes_per_token_layer)


def main() -> None:
    rows = []
    with (INPUTS / "sources.csv").open(newline="") as stream:
        sources = list(csv.DictReader(stream))
    for source in sources:
        path = INPUTS / source["config_file"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != source["config_sha256"]:
            raise ValueError(f"pinned config changed: {path}")
        config = json.loads(data)
        for condition, batch, context in CONDITIONS:
            layers, resident, peak, new_write = kv_sizes(config, batch, context)
            rows.append({
                "case": source["case"], "class": source["class"],
                "condition": condition, "batch": batch, "context_tokens": context,
                "attention_layers": layers,
                "resident_kv_bytes": resident,
                "largest_layer_kv_bytes": peak,
                "new_token_kv_write_bytes": new_write,
                "scope": "analytical BF16 KV only; excludes weights, activations and Mamba state",
            })
    path = HERE / "kv-working-set.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} analytical rows to {path}")


if __name__ == "__main__":
    main()
