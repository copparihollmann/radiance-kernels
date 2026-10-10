#!/usr/bin/env python3
"""Verify the pinned sharded Gemma checkpoint against the decoder graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from checkpoint import SafeTensorWeights
from stitch import build, model_specs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(checkpoint_dir: Path) -> dict:
    from safetensors import safe_open

    spec = model_specs()["gemma_2_2b_it"]
    checkpoint_dir = checkpoint_dir.resolve()
    actual = {name: sha256(checkpoint_dir / name)
              for name in spec["checkpoint_files_sha256"]}
    if actual != spec["checkpoint_files_sha256"]:
        raise ValueError("Gemma checkpoint shard or index hash differs from pinned revision")
    config_hash = sha256(checkpoint_dir / "config.json")
    if config_hash != spec["source_sha256"]:
        raise ValueError("Gemma config differs from pinned revision")
    graph = build("gemma_2_2b_it", prefill=1, decode_steps=1)
    weights = SafeTensorWeights(checkpoint_dir, "gemma2")
    bound = set()
    checked = 0
    for stage in graph.stages:
        logical = stage["attrs"].get("parameter")
        if logical is None:
            continue
        key = weights.key(logical)
        shape = (list(stage["attrs"]["weight_shape"])
                 if stage["op"] == "linear" else
                 [stage["attrs"]["vocab_size"], stage["shape"][-1]]
                 if stage["op"] == "embedding" else
                 [stage["shape"][-1]])
        if stage["op"] == "linear":
            shape.reverse()  # PyTorch checkpoint stores [out, in].
        filename = weights._key_to_file[key]
        with safe_open(str(filename), framework="pt", device="cpu") as file:
            if list(file.get_slice(key).get_shape()) != shape:
                raise ValueError(f"{key}: checkpoint shape differs from graph")
        bound.add(key)
        checked += 1
    if bound != weights.keys:
        raise ValueError("Gemma graph does not bind every checkpoint tensor")
    return {
        "model": "gemma_2_2b_it", "check": "pinned_sharded_checkpoint_graph_binding",
        "passed": True, "checkpoint_revision": spec["checkpoint_revision"],
        "config_sha256": config_hash, "checkpoint_files_sha256": actual,
        "graph_stages": len(graph.stages), "parameter_uses": checked,
        "bound_checkpoint_tensors": len(bound), "unbound_checkpoint_tensors": 0,
        "lm_head_tied_to_embedding": weights.key("lm_head") ==
        "model.embed_tokens.weight", "numerical_execution": False,
        "device_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args.checkpoint_dir)
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
