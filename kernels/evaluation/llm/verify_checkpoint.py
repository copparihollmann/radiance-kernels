#!/usr/bin/env python3
"""Compare checkpoint-bound stitched decoder layers with Transformers.

The independent Transformers forward and the stitched NumPy graph use the
same real checkpoint tensors. Use --full for every layer. This verifies
software wiring, not MX precision or Radiance execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from checkpoint import SafeTensorWeights
from reference import execute
from stitch import build, model_specs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(model_name: str, checkpoint_dir: Path, full: bool = False) -> dict:
    try:
        import torch
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError("install torch, transformers and safetensors to run this check") from error
    if model_name not in ("tinyllama", "deepseek_r1_distill_qwen_1_5b"):
        raise ValueError("checkpoint comparison is currently wired for TinyLlama and DeepSeek")
    checkpoint_dir = Path(checkpoint_dir)
    weights_path = checkpoint_dir / "model.safetensors"
    spec = dict(model_specs()[model_name])
    config_hash = sha256_file(checkpoint_dir / "config.json")
    if config_hash != spec["source_sha256"]:
        raise ValueError("local config differs from pinned model revision")
    if not full:
        spec["num_hidden_layers"] = 1
    provider = SafeTensorWeights(weights_path, spec["family"])
    graph = build(model_name, prefill=3, decode_steps=2, specs={model_name: spec})
    bindings = provider.check_bindings(graph)
    tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    values = execute(graph, {
        "prefill.token_ids": tokens[:3][None],
        "decode0.token_ids": tokens[3:4][None],
        "decode1.token_ids": tokens[4:5][None],
    }, weights=provider)
    stitched = values[graph.outputs[-1]][0, 0]

    config = AutoConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    config.num_hidden_layers = spec["num_hidden_layers"]
    config.use_cache = False
    # Normal construction initializes non-persistent RoPE buffers. Moving a
    # meta model with to_empty leaves those buffers undefined after state load.
    upstream = AutoModelForCausalLM.from_config(config).float().eval()
    with safe_open(str(weights_path), framework="pt", device="cpu") as file:
        state = {name: file.get_tensor(name).float() for name in upstream.state_dict()}
    upstream.load_state_dict(state, strict=True)
    del state
    with torch.inference_mode():
        upstream_outputs = upstream(torch.as_tensor(tokens[None], dtype=torch.long),
                                    use_cache=False, output_hidden_states=True)
        expected = upstream_outputs.logits[0, -1].float().numpy()
    embed_error = float(np.max(np.abs(
        values["prefill.embedding"] -
        upstream_outputs.hidden_states[0][:, :3].float().numpy())))
    hidden_error = float(np.max(np.abs(
        values["prefill.final_norm"] -
        upstream_outputs.hidden_states[-1][:, :3].float().numpy())))
    absolute = np.abs(stitched - expected)
    maximum = float(absolute.max())
    mean = float(absolute.mean())
    passed = bool(np.allclose(stitched, expected, rtol=1e-3, atol=1e-3))
    result = {"model": model_name, "source": spec["source"],
              "config_sha256": config_hash,
              "checkpoint_sha256": sha256_file(weights_path),
              "layers_checked": spec["num_hidden_layers"],
              "prefill_tokens": 3, "cached_decode_tokens": 2,
              "comparison": "last_logits_vs_transformers_full_causal_pass",
              "maximum_absolute_error": maximum, "mean_absolute_error": mean,
              "embedding_maximum_absolute_error": embed_error,
              "prefill_layer_hidden_maximum_absolute_error": hidden_error,
              "rtol": 1e-3, "atol": 1e-3, "passed": passed,
              "device_execution": False, "checkpoint_weights": True,
              "bound_tensors": bindings["bound_tensors"],
              "missing_tensors": bindings["missing_tensors"]}
    if not passed:
        raise AssertionError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=("tinyllama", "deepseek_r1_distill_qwen_1_5b"))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="run all decoder layers")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = check(args.model, args.checkpoint_dir, args.full)
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
