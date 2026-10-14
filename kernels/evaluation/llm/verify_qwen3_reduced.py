#!/usr/bin/env python3
"""Compare a small stitched Qwen3 graph with an independent Transformers run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from reference import execute
from stitch import build, model_specs


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class UpstreamWeights:
    def __init__(self, model: transformers.Qwen3ForCausalLM):
        self.tensors = {name: tensor.detach().cpu().numpy()
                        for name, tensor in model.state_dict().items()}

    def source_name(self, name: str) -> str:
        if name == "embed_tokens":
            return "model.embed_tokens.weight"
        if name == "final_norm":
            return "model.norm.weight"
        prefix, layer, component = name.split(".")
        if prefix != "layers":
            raise ValueError(f"unknown graph parameter: {name}")
        stem = f"model.layers.{layer}."
        if component == "attn_norm":
            return stem + "input_layernorm.weight"
        if component == "ffn_norm":
            return stem + "post_attention_layernorm.weight"
        if component in {"q_proj", "k_proj", "v_proj", "o_proj",
                         "q_norm", "k_norm"}:
            return stem + f"self_attn.{component}.weight"
        if component in {"gate_proj", "up_proj", "down_proj"}:
            return stem + f"mlp.{component}.weight"
        raise ValueError(f"unknown graph parameter: {name}")

    def __call__(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        value = self.tensors[self.source_name(name)]
        if value.shape == shape:
            return value
        if value.ndim == 2 and value.T.shape == shape:
            return value.T
        raise ValueError(f"{name}: checkpoint shape {value.shape} differs from {shape}")

    def embedding_row(self, name: str, row: int, width: int) -> np.ndarray:
        result = self.tensors[self.source_name(name)][row]
        if result.shape != (width,):
            raise ValueError("invalid embedding row width")
        return result


def check() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(19)
    pinned = model_specs()["qwen3_0_6b"]
    small = {**pinned, "hidden_size": 32, "intermediate_size": 64,
             "num_hidden_layers": 2, "num_attention_heads": 2,
             "num_key_value_heads": 1, "head_dim": 8, "vocab_size": 64}
    config = transformers.Qwen3Config(
        vocab_size=small["vocab_size"], hidden_size=small["hidden_size"],
        intermediate_size=small["intermediate_size"],
        num_hidden_layers=small["num_hidden_layers"],
        num_attention_heads=small["num_attention_heads"],
        num_key_value_heads=small["num_key_value_heads"],
        head_dim=small["head_dim"], rms_norm_eps=small["rms_norm_eps"],
        rope_theta=small["rope_theta"], tie_word_embeddings=True,
        attention_bias=False, use_cache=True, use_sliding_window=False,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen3ForCausalLM(config).eval()
    weights = UpstreamWeights(model)
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise AssertionError("upstream small model did not tie LM head to embedding")
    ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    graph = build("qwen3_0_6b", prefill=3, decode_steps=2,
                  specs={"qwen3_0_6b": small})
    inputs = {"prefill.token_ids": ids[:, :3],
              "decode0.token_ids": ids[:, 3:4],
              "decode1.token_ids": ids[:, 4:5]}
    mapped = execute(graph, inputs, weights=weights)
    with torch.no_grad():
        prefix = model(input_ids=torch.from_numpy(ids[:, :3].astype(np.int64)),
                       use_cache=True)
        first = model(input_ids=torch.from_numpy(ids[:, 3:4].astype(np.int64)),
                      past_key_values=prefix.past_key_values, use_cache=True)
        second = model(input_ids=torch.from_numpy(ids[:, 4:5].astype(np.int64)),
                       past_key_values=first.past_key_values, use_cache=True)
        full = model(input_ids=torch.from_numpy(ids.astype(np.int64)),
                     use_cache=False).logits[:, -1:, :]
    reference = [output.logits.detach().cpu().numpy() for output in
                 (prefix, first, second)]
    comparisons = []
    for phase, output_name, expected in zip(
            ("prefill", "decode0", "decode1"), graph.outputs, reference):
        actual = mapped[output_name]
        comparisons.append({
            "phase": phase, "shape": list(actual.shape),
            "max_abs_error": float(np.max(np.abs(actual - expected))),
            "allclose_atol_1e-5_rtol_1e-5": bool(np.allclose(
                actual, expected, atol=1e-5, rtol=1e-5)),
            "graph_output_sha256": digest(actual),
            "upstream_output_sha256": digest(expected),
        })
    full_logits = full.detach().cpu().numpy()
    upstream_cache_error = float(np.max(np.abs(reference[-1] - full_logits)))
    changed = UpstreamWeights(model)
    norm_name = "model.layers.0.self_attn.q_norm.weight"
    changed.tensors[norm_name] = np.zeros_like(changed.tensors[norm_name])
    changed_logits = execute(graph, inputs, weights=changed)[graph.outputs[-1]]
    changed_q_norm_delta = float(np.max(np.abs(changed_logits - reference[-1])))
    passed = (all(row["allclose_atol_1e-5_rtol_1e-5"] for row in comparisons)
              and np.allclose(reference[-1], full_logits, atol=1e-5, rtol=1e-5)
              and changed_q_norm_delta > 1e-4)
    return {
        "status": "passed" if passed else "failed",
        "scope": "two-layer random-weight Qwen3 upstream comparison; not checkpoint execution or device timing",
        "pinned_config_sha256": pinned["source_sha256"],
        "source_model": pinned["name"],
        "small_config": {key: small[key] for key in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "vocab_size", "rms_norm_eps", "rope_theta")},
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "seed": 19, "token_ids": ids.tolist(),
        "comparisons": comparisons,
        "upstream_cached_vs_full_max_abs_error": upstream_cache_error,
        "zero_first_query_norm_max_abs_logit_delta": changed_q_norm_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = check()
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
