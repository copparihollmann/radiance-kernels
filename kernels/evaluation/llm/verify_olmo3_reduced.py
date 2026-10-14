#!/usr/bin/env python3
"""Check an OLMo 3 local/global decoder mapping against upstream Transformers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import transformers

from stitch import build, model_specs
import olmo_graph_reference


CONFIG = Path(__file__).parent / "inputs/olmo3-7b.json"


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return (x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps))) * weight


def yarn_frequencies(config: dict, head_dim: int) -> np.ndarray:
    rope = config["rope_scaling"]
    base = float(config["rope_theta"])
    original = float(rope["original_max_position_embeddings"])
    factor = float(rope["factor"])

    def correction(rotations: float) -> float:
        return head_dim * math.log(original / (rotations * 2.0 * math.pi)) / (2.0 * math.log(base))

    low = max(math.floor(correction(float(rope["beta_fast"]))), 0)
    high = min(math.ceil(correction(float(rope["beta_slow"]))), head_dim - 1)
    if high == low:
        high += 0.001
    dimensions = np.arange(head_dim // 2, dtype=np.float32)
    ramp = np.clip((dimensions - low) / (high - low), 0.0, 1.0)
    power = np.arange(0, head_dim, 2, dtype=np.float32) / head_dim
    base_frequency = np.float32(base) ** power
    return ((1.0 - ramp) + ramp / factor) / base_frequency


def rotary(x: np.ndarray, positions: np.ndarray, inverse: np.ndarray,
           attention_factor: float) -> np.ndarray:
    half = x.shape[-1] // 2
    angles = positions[:, None].astype(np.float32) * inverse[None, :]
    duplicated = np.concatenate((angles, angles), axis=-1)
    cos = np.cos(duplicated).astype(np.float32) * attention_factor
    sin = np.sin(duplicated).astype(np.float32) * attention_factor
    rotated = np.concatenate((-x[..., half:], x[..., :half]), axis=-1)
    return x * cos[:, None, :] + rotated * sin[:, None, :]


def linear(x: np.ndarray, tensors: dict[str, np.ndarray], key: str) -> np.ndarray:
    return x @ tensors[key].T


def mapped_pass(ids: np.ndarray, config: dict, tensors: dict[str, np.ndarray],
                cache: list[tuple[np.ndarray, np.ndarray]] | None,
                *, force_full_attention: bool = False) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    hidden = tensors["model.embed_tokens.weight"][ids]
    head_count = config["num_attention_heads"]
    kv_count = config["num_key_value_heads"]
    head_dim = config["hidden_size"] // head_count
    eps = config["rms_norm_eps"]
    prior_length = 0 if cache is None else cache[0][0].shape[0]
    positions = np.arange(prior_length, prior_length + len(ids), dtype=np.int64)
    inverse = yarn_frequencies(config, head_dim)
    next_cache = []

    for layer, attention_type in enumerate(config["layer_types"]):
        prefix = f"model.layers.{layer}."
        attn = prefix + "self_attn."
        q = linear(hidden, tensors, attn + "q_proj.weight")
        k = linear(hidden, tensors, attn + "k_proj.weight")
        v = linear(hidden, tensors, attn + "v_proj.weight")
        # OLMo 3 normalizes the complete projection before splitting heads.
        q = rms_norm(q, tensors[attn + "q_norm.weight"], eps)
        k = rms_norm(k, tensors[attn + "k_norm.weight"], eps)
        q = rotary(q.reshape(len(ids), head_count, head_dim), positions, inverse,
                   config["rope_scaling"]["attention_factor"])
        k = rotary(k.reshape(len(ids), kv_count, head_dim), positions, inverse,
                   config["rope_scaling"]["attention_factor"])
        v = v.reshape(len(ids), kv_count, head_dim)
        if cache is not None:
            k = np.concatenate((cache[layer][0], k), axis=0)
            v = np.concatenate((cache[layer][1], v), axis=0)
        next_cache.append((k, v))
        repeats = head_count // kv_count
        repeated_k = np.repeat(k, repeats, axis=1)
        repeated_v = np.repeat(v, repeats, axis=1)
        scores = np.einsum("thd,shd->ths", q, repeated_k) / math.sqrt(head_dim)
        key_positions = np.arange(k.shape[0])
        allowed = key_positions[None, :] <= positions[:, None]
        if attention_type == "sliding_attention" and not force_full_attention:
            allowed &= key_positions[None, :] > positions[:, None] - config["sliding_window"]
        scores = np.where(allowed[:, None, :], scores, -np.inf)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= np.sum(probs, axis=-1, keepdims=True)
        context = np.einsum("ths,shd->thd", probs, repeated_v).reshape(len(ids), -1)
        attn_out = linear(context, tensors, attn + "o_proj.weight")
        hidden = hidden + rms_norm(attn_out, tensors[prefix + "post_attention_layernorm.weight"], eps)
        gate = linear(hidden, tensors, prefix + "mlp.gate_proj.weight")
        up = linear(hidden, tensors, prefix + "mlp.up_proj.weight")
        mlp = linear((gate / (1.0 + np.exp(-gate))) * up, tensors,
                     prefix + "mlp.down_proj.weight")
        hidden = hidden + rms_norm(mlp, tensors[prefix + "post_feedforward_layernorm.weight"], eps)

    hidden = rms_norm(hidden, tensors["model.norm.weight"], eps)
    logits = linear(hidden, tensors, "lm_head.weight")
    return logits, next_cache


def parameter_key(name: str) -> str:
    if name == "embed_tokens":
        return "model.embed_tokens.weight"
    if name == "lm_head":
        return "lm_head.weight"
    if name == "final_norm":
        return "model.norm.weight"
    prefix, layer, component = name.split(".")
    if prefix != "layers":
        raise ValueError(f"unknown graph parameter {name}")
    stem = f"model.layers.{layer}."
    if component in {"q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"}:
        return stem + f"self_attn.{component}.weight"
    if component == "post_attn_norm":
        return stem + "post_attention_layernorm.weight"
    if component == "post_ffn_norm":
        return stem + "post_feedforward_layernorm.weight"
    if component in {"gate_proj", "up_proj", "down_proj"}:
        return stem + f"mlp.{component}.weight"
    raise ValueError(f"unknown graph parameter {name}")


class GraphWeights:
    """Present the upstream fixture's row-major tensors to the graph oracle."""

    def __init__(self, tensors: dict[str, np.ndarray]):
        self.tensors = tensors

    def __call__(self, logical: str, shape: tuple[int, ...]) -> np.ndarray:
        tensor = self.tensors[parameter_key(logical)]
        value = tensor.T if tensor.ndim == 2 else tensor
        if value.shape != shape:
            raise ValueError(f"{logical}: fixture shape {value.shape} != {shape}")
        return value

    def embedding_row(self, logical: str, token: int, width: int) -> np.ndarray:
        row = self.tensors[parameter_key(logical)][token]
        if row.shape != (width,):
            raise ValueError(f"{logical}: fixture embedding width differs")
        return row


def execute_graph(graph, inputs: dict[str, np.ndarray],
                  tensors: dict[str, np.ndarray],
                  *, force_full_attention: bool = False) -> dict[str, np.ndarray]:
    """Execute the emitted stages and their named dependencies at reduced size."""
    values = dict(inputs)
    for stage in graph.stages:
        name, op, attrs = stage["id"], stage["op"], stage["attrs"]
        reads = [values[source] for source in stage["reads"]]
        if op == "embedding":
            result = tensors[parameter_key(attrs["parameter"])][reads[0]]
        elif op == "linear":
            result = linear(reads[0], tensors, parameter_key(attrs["parameter"]))
        elif op == "rmsnorm":
            result = rms_norm(reads[0], tensors[parameter_key(attrs["parameter"])],
                              attrs["epsilon"])
        elif op == "reshape_heads":
            result = reads[0].reshape(stage["shape"])
        elif op == "rope_yarn":
            frequency = yarn_frequencies({"rope_scaling": attrs["scaling"],
                                          "rope_theta": attrs["theta"]},
                                         stage["shape"][-1])
            positions = np.arange(attrs["position_start"],
                                  attrs["position_start"] + stage["shape"][1])
            result = np.stack([rotary(batch, positions, frequency,
                                      attrs["scaling"]["attention_factor"])
                               for batch in reads[0]], axis=0)
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
                scores /= math.sqrt(attrs["head_dim"])
                positions = np.arange(attrs["query_start"],
                                      attrs["query_start"] + q.shape[0])
                key_positions = np.arange(k.shape[0])
                allowed = key_positions[None, :] <= positions[:, None]
                if attrs["window"] is not None and not force_full_attention:
                    allowed &= key_positions[None, :] > positions[:, None] - attrs["window"]
                scores = np.where(allowed[:, None, :], scores, -np.inf)
                scores -= np.max(scores, axis=-1, keepdims=True)
                probabilities = np.exp(scores)
                probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
                outputs.append(np.einsum("ths,shd->thd", probabilities, repeated_v)
                               .reshape(q.shape[0], -1))
            result = np.stack(outputs, axis=0)
        elif op == "gated_activation":
            gate, up = reads
            result = (gate / (1.0 + np.exp(-gate))) * up
        elif op == "add":
            result = reads[0] + reads[1]
        elif op == "token_select":
            result = np.argmax(reads[0][:, -1, :], axis=-1).astype(np.int32)[:, None]
        else:
            raise ValueError(f"unsupported OLMo graph stage {name}: {op}")
        if list(result.shape) != stage["shape"] or not np.isfinite(result).all():
            raise AssertionError(f"{name}: invalid output shape or non-finite value")
        values[name] = result
    return values


def check() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(23)
    config_bytes = CONFIG.read_bytes()
    pinned = json.loads(config_bytes)
    small = {**pinned, "vocab_size": 64, "hidden_size": 32,
             "intermediate_size": 64, "num_hidden_layers": 4,
             "num_attention_heads": 2, "num_key_value_heads": 2,
             "sliding_window": 3, "pad_token_id": 0,
             "layer_types": pinned["layer_types"][:4]}
    rope_parameters = {**pinned["rope_scaling"], "rope_theta": pinned["rope_theta"]}
    config = transformers.Olmo3Config(
        vocab_size=small["vocab_size"], hidden_size=small["hidden_size"],
        intermediate_size=small["intermediate_size"],
        num_hidden_layers=small["num_hidden_layers"],
        num_attention_heads=small["num_attention_heads"],
        num_key_value_heads=small["num_key_value_heads"],
        max_position_embeddings=small["max_position_embeddings"],
        rope_parameters=rope_parameters, rms_norm_eps=small["rms_norm_eps"],
        sliding_window=small["sliding_window"], layer_types=small["layer_types"],
        pad_token_id=small["pad_token_id"], tie_word_embeddings=False,
        attention_bias=False, use_cache=True,
    )
    config._attn_implementation = "eager"
    model = transformers.Olmo3ForCausalLM(config).eval()
    tensors = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    yarn_error = float(np.max(np.abs(
        yarn_frequencies(small, small["hidden_size"] // small["num_attention_heads"])
        - model.model.rotary_emb.inv_freq.detach().cpu().numpy())))
    ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    phases = (ids[:4], ids[4:5], ids[5:6])
    small_spec = {**model_specs()["olmo3_7b"], **{
        key: small[key] for key in ("hidden_size", "intermediate_size",
                                    "num_hidden_layers", "num_attention_heads",
                                    "num_key_value_heads", "vocab_size",
                                    "sliding_window", "layer_types")}}
    small_spec["head_dim"] = small["hidden_size"] // small["num_attention_heads"]
    graph = build("olmo3_7b", prefill=4, decode_steps=2,
                  specs={"olmo3_7b": small_spec})
    inputs = {f"{name}.token_ids": phase[None, :] for name, phase in
              zip(("prefill", "decode0", "decode1"), phases)}
    graph_values = execute_graph(graph, inputs, tensors)
    compiler_values = olmo_graph_reference.execute(
        graph, inputs, weights=GraphWeights(tensors))
    compiler_reference_error = max(
        float(np.max(np.abs(compiler_values[stage["writes"]] -
                            graph_values[stage["writes"]])))
        for stage in graph.stages)
    mapped_cache = None
    upstream_cache = None
    comparisons = []
    with torch.no_grad():
        for name, phase, output_name in zip(("prefill", "decode0", "decode1"),
                                            phases, graph.outputs):
            mapped, mapped_cache = mapped_pass(phase, small, tensors, mapped_cache)
            upstream = model(input_ids=torch.from_numpy(phase[None, :]),
                             past_key_values=upstream_cache, use_cache=True)
            upstream_cache = upstream.past_key_values
            expected = upstream.logits[0].detach().cpu().numpy()
            staged = graph_values[output_name][0]
            comparisons.append({
                "phase": name, "shape": list(staged.shape),
                "max_abs_error": float(np.max(np.abs(staged - expected))),
                "allclose_atol_1e-5_rtol_1e-5": bool(np.allclose(
                    staged, expected, atol=1e-5, rtol=1e-5)),
                "graph_vs_direct_max_abs_error": float(np.max(np.abs(staged - mapped))),
                "mapped_sha256": hashlib.sha256(staged.tobytes()).hexdigest(),
                "upstream_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
            })
        full = model(input_ids=torch.from_numpy(ids[None, :]),
                     use_cache=False).logits[0, -1].detach().cpu().numpy()
    incorrect_cache = None
    for phase in phases:
        incorrect, incorrect_cache = mapped_pass(
            phase, small, tensors, incorrect_cache, force_full_attention=True)
    incorrect_delta = float(np.max(np.abs(incorrect[-1] - full)))
    incorrect_graph = execute_graph(graph, inputs, tensors, force_full_attention=True)
    incorrect_graph_delta = float(np.max(np.abs(
        incorrect_graph[graph.outputs[-1]][0, -1] - full)))
    cached_full_error = float(np.max(np.abs(expected[-1] - full)))
    passed = (all(row["allclose_atol_1e-5_rtol_1e-5"] for row in comparisons)
              and compiler_reference_error < 1e-5
              and cached_full_error < 1e-5 and incorrect_delta > 1e-4
              and yarn_error < 1e-6 and incorrect_graph_delta > 1e-4
              and all(row["graph_vs_direct_max_abs_error"] < 1e-5
                      for row in comparisons))
    return {
        "status": "passed" if passed else "failed",
        "scope": "four-layer random-weight OLMo 3 staged graph, compiler reference, direct mapping, and upstream prefill/cached decode; no checkpoint or Radiance execution",
        "pinned_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_model": "allenai/Olmo-3-7B-Instruct",
        "small_config": {key: small[key] for key in (
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "vocab_size",
            "sliding_window", "layer_types")},
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "seed": 23, "token_ids": ids.tolist(),
        "graph_stage_count": len(graph.stages),
        "compiler_reference_stage_max_abs_error": compiler_reference_error,
        "compiler_reference_source_sha256": hashlib.sha256(
            Path(olmo_graph_reference.__file__).read_bytes()).hexdigest(),
        "comparisons": comparisons,
        "upstream_cached_vs_full_max_abs_error": cached_full_error,
        "yarn_inverse_frequency_max_abs_error": yarn_error,
        "force_full_attention_max_abs_logit_delta": incorrect_delta,
        "force_full_attention_graph_max_abs_logit_delta": incorrect_graph_delta,
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
