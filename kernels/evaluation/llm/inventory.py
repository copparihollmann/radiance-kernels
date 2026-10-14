#!/usr/bin/env python3
"""Record the ordered layer structure in the pinned candidate model configs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


def layer_kinds(config: dict) -> list[str]:
    count = int(config["num_hidden_layers"])
    if count <= 0:
        raise ValueError("num_hidden_layers must be positive")
    model_type = config["model_type"]

    if model_type == "nemotron_h":
        pattern = config["hybrid_override_pattern"]
        if len(pattern) != count or set(pattern) - {"M", "*", "-"}:
            raise ValueError("invalid Nemotron layer pattern")
        names = {"M": "mamba2", "*": "full_attention", "-": "mlp_only"}
        return [names[symbol] for symbol in pattern]

    if model_type == "olmo3":
        kinds = config["layer_types"]
        if len(kinds) != count or set(kinds) - {"sliding_attention", "full_attention"}:
            raise ValueError("invalid OLMo layer types")
        return [f"{kind}+dense_ffn" for kind in kinds]

    if model_type == "qwen3_moe":
        experts = int(config["num_experts"])
        selected = int(config["num_experts_per_tok"])
        step = int(config["decoder_sparse_step"])
        mlp_only = set(config["mlp_only_layers"])
        if experts <= 0 or not 0 < selected <= experts or step <= 0:
            raise ValueError("invalid MoE routing dimensions")
        if any(not isinstance(layer, int) or layer < 0 or layer >= count
               for layer in mlp_only):
            raise ValueError("invalid MLP-only layer index")
        # Qwen3-MoE selects experts on 1-based multiples of decoder_sparse_step,
        # except the explicit 0-based mlp_only_layers list.
        return ["full_attention+moe" if i not in mlp_only and (i + 1) % step == 0
                else "full_attention+dense_ffn" for i in range(count)]

    if model_type in {"qwen3", "mistral"}:
        if config.get("sliding_window") not in (None, 0):
            raise ValueError("dense inventory would ignore a sliding window")
        return ["full_attention+dense_ffn"] * count

    raise ValueError(f"unsupported model type: {model_type}")


def inventory(inputs: Path) -> dict:
    cases = []
    with (inputs / "sources.csv").open(newline="") as stream:
        sources = list(csv.DictReader(stream))
    for source in sources:
        raw = (inputs / source["config_file"]).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != source["config_sha256"]:
            raise ValueError(f"pinned config changed: {source['config_file']}")
        config = json.loads(raw)
        kinds = layer_kinds(config)
        counts = Counter(kinds)
        entry = {
            "case": source["case"], "class": source["class"],
            "model": source["model"], "revision": source["revision"],
            "config_sha256": digest, "model_type": config["model_type"],
            "num_hidden_layers": len(kinds),
            "layer_kinds": kinds, "counts": dict(sorted(counts.items())),
        }
        if config["model_type"] == "olmo3":
            entry["sliding_window_tokens"] = int(config["sliding_window"])
            entry["checkpoint_use_cache"] = bool(config["use_cache"])
        if config["model_type"] == "qwen3_moe":
            entry["num_experts"] = int(config["num_experts"])
            entry["experts_per_token"] = int(config["num_experts_per_tok"])
        cases.append(entry)
    return {"scope": "config-derived layer order; no weights, execution, or timing",
            "cases": cases}


def main() -> None:
    result = inventory(HERE / "inputs")
    path = HERE / "architecture-inventory.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {len(result['cases'])} config-derived cases to {path}")


if __name__ == "__main__":
    main()
