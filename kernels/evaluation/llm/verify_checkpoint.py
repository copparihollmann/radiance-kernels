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
    if model_name not in ("tinyllama", "deepseek_r1_distill_qwen_1_5b",
                          "gemma_2_2b_it"):
        raise ValueError("unknown pinned decoder checkpoint")
    checkpoint_dir = Path(checkpoint_dir)
    weights_path = (checkpoint_dir / "model.safetensors"
                    if (checkpoint_dir / "model.safetensors").is_file()
                    else checkpoint_dir)
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
    if model_name == "gemma_2_2b_it":
        config._attn_implementation = "eager"
    # Normal construction initializes non-persistent RoPE buffers. Moving a
    # meta model with to_empty leaves those buffers undefined after state load.
    upstream = AutoModelForCausalLM.from_config(config).float().eval()
    state = {}
    for name in upstream.state_dict():
        checkpoint_name = ("model.embed_tokens.weight"
                           if model_name == "gemma_2_2b_it" and
                           name == "lm_head.weight" and name not in provider.keys
                           else name)
        if checkpoint_name not in provider.keys:
            raise KeyError(f"upstream parameter missing from checkpoint: {name}")
        source = provider._key_to_file[checkpoint_name]
        with safe_open(str(source), framework="pt", device="cpu") as file:
            state[name] = file.get_tensor(checkpoint_name).float()
    upstream.load_state_dict(state, strict=True)
    del state
    module_outputs = {}
    handles = []
    if model_name == "gemma_2_2b_it":
        monitored = {
            "model.layers.0.input_layernorm": "prefill.layer00.attn_norm",
            "model.layers.0.self_attn.q_proj": "prefill.layer00.q_proj",
            "model.layers.0.self_attn.k_proj": "prefill.layer00.k_proj",
            "model.layers.0.self_attn.v_proj": "prefill.layer00.v_proj",
            "model.layers.0.self_attn.o_proj": "prefill.layer00.o_proj",
            "model.layers.0.post_attention_layernorm": "prefill.layer00.post_attn_norm",
            "model.layers.0.pre_feedforward_layernorm": "prefill.layer00.ffn_norm",
            "model.layers.0.mlp.gate_proj": "prefill.layer00.gate_proj",
            "model.layers.0.mlp.up_proj": "prefill.layer00.up_proj",
            "model.layers.0.mlp.down_proj": "prefill.layer00.down_proj",
            "model.layers.0.post_feedforward_layernorm": "prefill.layer00.post_ffn_norm",
            "model.norm": "prefill.final_norm",
        }
        for name, module in upstream.named_modules():
            if name in monitored:
                stage_name = monitored[name]
                handles.append(module.register_forward_hook(
                    lambda _module, _inputs, output, stage_name=stage_name:
                    module_outputs.__setitem__(stage_name,
                                               output.detach().float().cpu().numpy())))
    with torch.inference_mode():
        upstream_outputs = upstream(torch.as_tensor(tokens[None], dtype=torch.long),
                                    use_cache=False, output_hidden_states=True)
        expected = upstream_outputs.logits[0, -1].float().numpy()
    for handle in handles:
        handle.remove()
    module_errors = {}
    for stage_name, output in module_outputs.items():
        target = values[stage_name]
        actual = output[:, :target.shape[1]].reshape(target.shape)
        module_errors[stage_name] = float(np.max(np.abs(target - actual)))
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
    checkpoint_hashes = ({name: sha256_file(checkpoint_dir / name)
                          for name in spec["checkpoint_files_sha256"]}
                         if model_name == "gemma_2_2b_it" else None)
    if checkpoint_hashes is not None and checkpoint_hashes != spec[
            "checkpoint_files_sha256"]:
        raise ValueError("Gemma checkpoint shards differ from pinned revision")
    result = {"model": model_name, "source": spec["source"],
              "config_sha256": config_hash,
              "checkpoint_sha256": (sha256_file(weights_path)
                                     if weights_path.is_file() else None),
              "checkpoint_files_sha256": checkpoint_hashes,
              "layers_checked": spec["num_hidden_layers"],
              "prefill_tokens": 3, "cached_decode_tokens": 2,
              "comparison": "last_logits_vs_transformers_full_causal_pass",
              "upstream_attention_implementation": config._attn_implementation,
              "maximum_absolute_error": maximum, "mean_absolute_error": mean,
              "embedding_maximum_absolute_error": embed_error,
              "prefill_layer_hidden_maximum_absolute_error": hidden_error,
              "upstream_module_max_abs_error": module_errors,
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
        choices=("tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma_2_2b_it"))
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
