#!/usr/bin/env python3
"""Compare the reduced held-out Mistral graph with upstream Transformers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from reference import execute, reduced_spec
from stitch import build, model_specs


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class UpstreamWeights:
    def __init__(self, model: transformers.MistralForCausalLM):
        self.tensors = {name: tensor.detach().cpu().numpy()
                        for name, tensor in model.state_dict().items()}

    @staticmethod
    def source_name(name: str) -> str:
        if name == "embed_tokens":
            return "model.embed_tokens.weight"
        if name == "final_norm":
            return "model.norm.weight"
        if name == "lm_head":
            return "lm_head.weight"
        prefix, layer, component = name.split(".")
        if prefix != "layers":
            raise ValueError(f"unknown graph parameter {name}")
        stem = f"model.layers.{layer}."
        if component == "attn_norm":
            return stem + "input_layernorm.weight"
        if component == "ffn_norm":
            return stem + "post_attention_layernorm.weight"
        if component in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            return stem + f"self_attn.{component}.weight"
        if component in {"gate_proj", "up_proj", "down_proj"}:
            return stem + f"mlp.{component}.weight"
        raise ValueError(f"unknown graph parameter {name}")

    def __call__(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        source = self.tensors[self.source_name(name)]
        value = source.T if source.ndim == 2 else source
        if value.shape != shape:
            raise ValueError(f"{name}: source {source.shape}, graph {shape}")
        return value

    def embedding_row(self, name: str, row: int, width: int) -> np.ndarray:
        value = self.tensors[self.source_name(name)][row]
        if value.shape != (width,):
            raise ValueError("embedding row differs from graph width")
        return value


def check() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(37)
    pinned = model_specs()["mistral_7b_v03"]
    small = reduced_spec(pinned)
    config = transformers.MistralConfig(
        vocab_size=small["vocab_size"], hidden_size=small["hidden_size"],
        intermediate_size=small["intermediate_size"],
        num_hidden_layers=small["num_hidden_layers"],
        num_attention_heads=small["num_attention_heads"],
        num_key_value_heads=small["num_key_value_heads"],
        rms_norm_eps=small["rms_norm_eps"], rope_theta=small["rope_theta"],
        sliding_window=None, tie_word_embeddings=False, use_cache=True,
    )
    config._attn_implementation = "eager"
    model = transformers.MistralForCausalLM(config).eval()
    weights = UpstreamWeights(model)
    ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    phases = (ids[:3], ids[3:4], ids[4:5])
    graph = build("mistral_7b_v03", prefill=3, decode_steps=2,
                  specs={"mistral_7b_v03": small})
    inputs = {f"{phase}.token_ids": tokens[None, :] for phase, tokens in
              zip(("prefill", "decode0", "decode1"), phases)}
    mapped = execute(graph, inputs, weights=weights)
    comparisons = []
    upstream_cache = None
    with torch.no_grad():
        for phase, tokens, output in zip(
                ("prefill", "decode0", "decode1"), phases, graph.outputs):
            result = model(input_ids=torch.from_numpy(tokens[None, :].astype(np.int64)),
                           past_key_values=upstream_cache, use_cache=True)
            upstream_cache = result.past_key_values
            actual = mapped[output]
            expected = result.logits.detach().cpu().numpy()
            comparisons.append({
                "phase": phase, "shape": list(actual.shape),
                "max_abs_error": float(np.max(np.abs(actual - expected))),
                "allclose_atol_1e-5_rtol_1e-5": bool(np.allclose(
                    actual, expected, atol=1e-5, rtol=1e-5)),
                "mapped_sha256": digest(actual),
                "upstream_sha256": digest(expected),
            })
        full = model(input_ids=torch.from_numpy(ids[None, :].astype(np.int64)),
                     use_cache=False).logits[:, -1:, :].detach().cpu().numpy()
        greedy_graph = build("mistral_7b_v03", prefill=3, decode_steps=2,
                             generation="greedy", specs={"mistral_7b_v03": small})
        greedy_values = execute(greedy_graph,
                                {"prefill.token_ids": ids[None, :3]}, weights=weights)
        greedy_result = model(input_ids=torch.from_numpy(ids[None, :3].astype(np.int64)),
                              use_cache=True)
        greedy_error = float(np.max(np.abs(
            greedy_values[greedy_graph.outputs[0]]
            - greedy_result.logits.detach().cpu().numpy())))
        greedy_tokens = []
        for step in range(2):
            token = torch.argmax(greedy_result.logits[:, -1, :], dim=-1)[:, None]
            greedy_tokens.append(int(token.item()))
            if not np.array_equal(greedy_values[f"decode{step}.token_ids"],
                                  token.numpy()):
                raise AssertionError("greedy graph selected a different token")
            greedy_result = model(input_ids=token,
                                  past_key_values=greedy_result.past_key_values,
                                  use_cache=True)
            greedy_error = max(greedy_error, float(np.max(np.abs(
                greedy_values[greedy_graph.outputs[step + 1]]
                - greedy_result.logits.detach().cpu().numpy()))))
    cached_full_error = float(np.max(np.abs(expected - full)))
    changed = dict(inputs)
    changed["decode1.token_ids"] = np.array([[6]], dtype=np.int32)
    changed_logits = execute(graph, changed, weights=weights)[graph.outputs[-1]]
    changed_delta = float(np.max(np.abs(changed_logits - mapped[graph.outputs[-1]])))
    passed = (all(row["allclose_atol_1e-5_rtol_1e-5"] for row in comparisons)
              and cached_full_error < 1e-5 and greedy_error < 1e-5
              and changed_delta > 1e-4)
    return {
        "status": "passed" if passed else "failed",
        "scope": "two-layer random-weight Mistral prefill, cached decode, and greedy token feedback; no checkpoint or Radiance execution",
        "source_model": pinned["name"],
        "pinned_config_sha256": pinned["source_sha256"],
        "small_config": {key: small[key] for key in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "vocab_size", "rms_norm_eps", "rope_theta")},
        "seed": 37, "token_ids": ids.tolist(),
        "graph_stage_count": len(graph.stages),
        "comparisons": comparisons,
        "upstream_cached_vs_full_max_abs_error": cached_full_error,
        "changed_decode_token_max_abs_logit_delta": changed_delta,
        "greedy_graph_stage_count": len(greedy_graph.stages),
        "greedy_selected_tokens": greedy_tokens,
        "greedy_max_abs_logit_error": greedy_error,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
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
