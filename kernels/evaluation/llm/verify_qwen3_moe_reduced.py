#!/usr/bin/env python3
"""Execute a reduced stitched Qwen3 MoE decoder against Transformers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from stitch import build, model_specs


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def parameter_name(name: str) -> str:
    if name == "embed_tokens":
        return "model.embed_tokens.weight"
    if name == "final_norm":
        return "model.norm.weight"
    if name == "lm_head":
        return "lm_head.weight"
    prefix, layer, component = name.split(".", 2)
    if prefix != "layers":
        raise ValueError(f"unknown graph parameter {name}")
    stem = f"model.layers.{layer}."
    if component == "attn_norm":
        return stem + "input_layernorm.weight"
    if component == "ffn_norm":
        return stem + "post_attention_layernorm.weight"
    if component == "router":
        return stem + "mlp.gate.weight"
    if component.startswith("experts."):
        return stem + "mlp." + component
    if component in {"q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"}:
        return stem + f"self_attn.{component}.weight"
    raise ValueError(f"unknown graph parameter {name}")


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


def execute(graph, inputs: dict[str, np.ndarray], weights: dict[str, np.ndarray],
            *, normalize_selected: bool = True) -> dict[str, np.ndarray]:
    values = dict(inputs)
    for stage in graph.stages:
        op, attrs = stage["op"], stage["attrs"]
        reads = [values[name] for name in stage["reads"]]
        if op == "embedding":
            result = weights[parameter_name(attrs["parameter"])][reads[0]]
        elif op == "rmsnorm":
            x = reads[0]
            factor = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True)
                                   + attrs["epsilon"])
            result = x * factor * weights[parameter_name(attrs["parameter"])]
        elif op == "linear":
            result = (reads[0] @ weights[parameter_name(attrs["parameter"])].T)
            result = result.reshape(stage["shape"])
        elif op == "rope":
            x = reads[0]
            head_dim = x.shape[-1]
            inverse = 1.0 / (np.float32(attrs["theta"]) **
                             (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
            positions = np.arange(attrs["position_start"],
                                  attrs["position_start"] + x.shape[1], dtype=np.float32)
            angles = positions[:, None] * inverse[None, :]
            duplicated = np.concatenate((angles, angles), axis=-1)
            cos = np.cos(duplicated).astype(np.float32)[None, :, None, :]
            sin = np.sin(duplicated).astype(np.float32)[None, :, None, :]
            rotated = np.concatenate((-x[..., head_dim // 2:],
                                       x[..., :head_dim // 2]), axis=-1)
            result = x * cos + rotated * sin
        elif op == "kv_append":
            result = reads[0] if len(reads) == 1 else np.concatenate(reads, axis=1)
        elif op == "causal_gqa":
            query, key, value = reads
            repeats = attrs["q_heads"] // attrs["kv_heads"]
            outputs = []
            for q, k, v in zip(query, key, value):
                repeated_k = np.repeat(k, repeats, axis=1)
                repeated_v = np.repeat(v, repeats, axis=1)
                scores = np.einsum("thd,shd->ths", q, repeated_k)
                scores /= np.sqrt(np.float32(attrs["head_dim"]))
                positions = np.arange(attrs["query_start"],
                                      attrs["query_start"] + q.shape[0])
                allowed = np.arange(k.shape[0])[None, :] <= positions[:, None]
                scores = np.where(allowed[:, None, :], scores, -np.inf)
                probabilities = softmax(scores)
                outputs.append(np.einsum("ths,shd->thd", probabilities, repeated_v)
                               .reshape(q.shape[0], -1))
            result = np.stack(outputs, axis=0)
        elif op == "softmax":
            result = softmax(reads[0])
        elif op == "topk_indices":
            result = np.argsort(-reads[0], axis=-1)[..., :attrs["k"]].astype(np.int32)
        elif op == "topk_scores":
            result = np.take_along_axis(reads[0], reads[1], axis=-1)
            if attrs["renormalize"] and normalize_selected:
                result /= np.sum(result, axis=-1, keepdims=True)
        elif op == "expert_group":
            result = np.broadcast_to(reads[0][..., None, :], stage["shape"])
        elif op == "expert_gate_up":
            name = next(iter(attrs["parameters"]))
            selected_weights = weights[parameter_name(name)][reads[1]]
            result = np.einsum("btkh,btkfh->btkf", reads[0], selected_weights)
        elif op == "expert_swiglu":
            gate, up = np.split(reads[0], 2, axis=-1)
            result = (gate / (1.0 + np.exp(-gate))) * up
        elif op == "expert_down":
            name = next(iter(attrs["parameters"]))
            selected_weights = weights[parameter_name(name)][reads[1]]
            result = np.einsum("btkf,btkhf->btkh", reads[0], selected_weights)
        elif op == "expert_weighted_reduce":
            result = np.sum(reads[0] * reads[1][..., None], axis=-2)
        elif op == "add":
            result = reads[0] + reads[1]
        elif op == "token_select":
            result = np.argmax(reads[0][:, -1, :], axis=-1).astype(np.int32)[:, None]
        else:
            raise ValueError(f"unknown MoE graph op {op}")
        if list(result.shape) != stage["shape"] or not np.isfinite(result).all():
            raise AssertionError(f"{stage['id']}: wrong shape or non-finite output")
        values[stage["writes"]] = result
    return values


def check() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(31)
    pinned = model_specs()["qwen3_30b_a3b"]
    small = {**pinned, "vocab_size": 64, "hidden_size": 32,
             "intermediate_size": 64, "num_hidden_layers": 2,
             "num_attention_heads": 2, "num_key_value_heads": 1,
             "head_dim": 16, "moe_intermediate_size": 16,
             "num_experts": 16, "num_experts_per_tok": 8}
    config = transformers.Qwen3MoeConfig(
        vocab_size=small["vocab_size"], hidden_size=small["hidden_size"],
        intermediate_size=small["intermediate_size"],
        num_hidden_layers=small["num_hidden_layers"],
        num_attention_heads=small["num_attention_heads"],
        num_key_value_heads=small["num_key_value_heads"],
        head_dim=small["head_dim"],
        moe_intermediate_size=small["moe_intermediate_size"],
        num_experts=small["num_experts"],
        num_experts_per_tok=small["num_experts_per_tok"],
        decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True,
        rope_theta=small["rope_theta"], rms_norm_eps=small["rms_norm_eps"],
        attention_bias=False, tie_word_embeddings=False, use_cache=True,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen3MoeForCausalLM(config).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.copy_(torch.randn_like(layer.mlp.gate.weight) * 0.15)
            layer.mlp.experts.gate_up_proj.copy_(
                torch.randn_like(layer.mlp.experts.gate_up_proj) * 0.15)
            layer.mlp.experts.down_proj.copy_(
                torch.randn_like(layer.mlp.experts.down_proj) * 0.15)
    weights = {name: tensor.detach().cpu().numpy()
               for name, tensor in model.state_dict().items()}
    ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    phases = (ids[:3], ids[3:4], ids[4:5])
    graph = build("qwen3_30b_a3b", prefill=3, decode_steps=2,
                  specs={"qwen3_30b_a3b": small})
    inputs = {f"{phase}.token_ids": tokens[None, :] for phase, tokens in
              zip(("prefill", "decode0", "decode1"), phases)}
    mapped = execute(graph, inputs, weights)
    wrong_router = execute(graph, inputs, weights, normalize_selected=False)
    selected_from_upstream = {}
    handles = []
    for layer_index, layer in enumerate(model.model.layers):
        def capture(_module, _args, output, *, index=layer_index):
            selected_from_upstream[index] = output[2].detach().cpu().numpy().copy()
        handles.append(layer.mlp.gate.register_forward_hook(capture))
    comparisons = []
    upstream_cache = None
    with torch.no_grad():
        for phase, tokens, output_name in zip(
                ("prefill", "decode0", "decode1"), phases, graph.outputs):
            selected_from_upstream.clear()
            result = model(input_ids=torch.from_numpy(tokens[None, :]),
                           past_key_values=upstream_cache, use_cache=True)
            upstream_cache = result.past_key_values
            actual = mapped[output_name]
            expected = result.logits.detach().cpu().numpy()
            route_matches = all(np.array_equal(
                mapped[f"{phase}.layer{layer:02d}.topk_indices"].reshape(-1, 8),
                selected_from_upstream[layer]) for layer in range(2))
            comparisons.append({
                "phase": phase, "shape": list(actual.shape),
                "max_abs_error": float(np.max(np.abs(actual - expected))),
                "allclose_atol_1e-5_rtol_1e-5": bool(np.allclose(
                    actual, expected, atol=1e-5, rtol=1e-5)),
                "routing_indices_match_upstream": route_matches,
                "mapped_sha256": digest(actual),
                "upstream_sha256": digest(expected),
            })
        full = model(input_ids=torch.from_numpy(ids[None, :]),
                     use_cache=False).logits[:, -1:, :].detach().cpu().numpy()
        greedy_graph = build("qwen3_30b_a3b", prefill=3, decode_steps=2,
                             generation="greedy",
                             specs={"qwen3_30b_a3b": small})
        greedy_values = execute(greedy_graph,
                                {"prefill.token_ids": ids[None, :3]}, weights)
        greedy_reference = model(input_ids=torch.from_numpy(ids[None, :3]),
                                 use_cache=True)
        greedy_tokens = []
        greedy_max_error = float(np.max(np.abs(
            greedy_values[greedy_graph.outputs[0]]
            - greedy_reference.logits.detach().cpu().numpy())))
        for step in range(2):
            token = torch.argmax(greedy_reference.logits[:, -1, :],
                                 dim=-1)[:, None]
            greedy_tokens.append(int(token.item()))
            if not np.array_equal(greedy_values[f"decode{step}.token_ids"],
                                  token.numpy()):
                raise AssertionError("greedy graph selected a different token")
            greedy_reference = model(input_ids=token,
                                     past_key_values=greedy_reference.past_key_values,
                                     use_cache=True)
            greedy_max_error = max(greedy_max_error, float(np.max(np.abs(
                greedy_values[greedy_graph.outputs[step + 1]]
                - greedy_reference.logits.detach().cpu().numpy()))))
    for handle in handles:
        handle.remove()
    cached_full_error = float(np.max(np.abs(expected - full)))
    without_norm_delta = float(np.max(np.abs(
        wrong_router[graph.outputs[-1]] - expected)))
    passed = (all(row["allclose_atol_1e-5_rtol_1e-5"] and
                  row["routing_indices_match_upstream"] for row in comparisons)
              and cached_full_error < 1e-5 and without_norm_delta > 1e-4
              and greedy_max_error < 1e-5)
    return {
        "status": "passed" if passed else "failed",
        "scope": "two-layer random-weight stitched Qwen3 MoE prefill, cached decode, and greedy token feedback; no checkpoint or Radiance execution",
        "pinned_config_sha256": pinned["source_sha256"],
        "source_model": pinned["name"],
        "small_config": {key: small[key] for key in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "num_experts", "num_experts_per_tok", "moe_intermediate_size",
            "vocab_size")},
        "seed": 31, "token_ids": ids.tolist(),
        "graph_stage_count": len(graph.stages),
        "comparisons": comparisons,
        "greedy_graph_stage_count": len(greedy_graph.stages),
        "greedy_selected_tokens": greedy_tokens,
        "greedy_max_abs_logit_error": greedy_max_error,
        "upstream_cached_vs_full_max_abs_error": cached_full_error,
        "without_topk_renormalization_max_abs_logit_delta": without_norm_delta,
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
