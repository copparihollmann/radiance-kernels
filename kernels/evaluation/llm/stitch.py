#!/usr/bin/env python3
"""Build checked tensor-dependency schedules for the four PR #1 models.

This is a software dataflow schedule. A listed kernel is an existing primitive,
not an executable shared-buffer implementation of the stage.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
KERNELS = HERE.parents[1]
SPECS = HERE / "pr1-models.json"


def model_specs() -> dict:
    return json.loads(SPECS.read_text())


class Graph:
    def __init__(self, model: str, spec: dict):
        self.model = model
        self.spec = spec
        self.tensors: dict[str, dict] = {}
        self.stages: list[dict] = []
        self.outputs: list[str] = []

    def input(self, name: str, shape: tuple[int, ...], dtype: str = "bf16") -> str:
        self._tensor(name, shape, dtype)
        return name

    def _tensor(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
        if name in self.tensors or not shape or any(n <= 0 for n in shape):
            raise ValueError(f"invalid or duplicate tensor {name}: {shape}")
        self.tensors[name] = {"shape": list(shape), "dtype": dtype}

    def add(self, name: str, op: str, reads: tuple[str, ...],
            shape: tuple[int, ...], kernel: str | None = None,
            attrs: dict | None = None, dtype: str = "bf16") -> str:
        for source in reads:
            if source not in self.tensors:
                raise ValueError(f"{name} reads undefined tensor {source}")
        if kernel is not None and not (KERNELS / kernel / "Makefile").exists():
            raise ValueError(f"{name} names unknown kernel {kernel}")
        self._tensor(name, shape, dtype)
        self.stages.append({
            "id": name, "op": op, "reads": list(reads), "writes": name,
            "shape": list(shape), "kernel": kernel,
            "status": "primitive_only" if kernel else "missing_device_stage",
            "attrs": attrs or {},
        })
        return name

    def report(self) -> dict:
        counts = Counter(stage["status"] for stage in self.stages)
        return {
            "model": self.model, "source": self.spec["source"],
            "mode": "tensor_dataflow_plan",
            "generation": getattr(self, "generation", None),
            "device_execution": False,
            "measured_cycles": None, "outputs": self.outputs,
            "stage_count": len(self.stages), "stage_status": dict(counts),
            "tensors": self.tensors, "stages": self.stages,
        }


def _decoder_pass(graph: Graph, spec: dict, phase: str, batch: int,
                  tokens: int, past: int, caches: dict[int, tuple[str, str]],
                  token_ids: str | None = None) -> tuple[str, dict]:
    h, f = spec["hidden_size"], spec["intermediate_size"]
    qh, kvh, d = (spec[key] for key in
                  ("num_attention_heads", "num_key_value_heads", "head_dim"))
    family = spec["family"]
    ids = token_ids or graph.input(f"{phase}.token_ids", (batch, tokens), "int32")
    if graph.tensors[ids]["shape"] != [batch, tokens]:
        raise ValueError(f"{phase}: invalid token input shape")
    hidden = graph.add(f"{phase}.embedding", "embedding", (ids,), (batch, tokens, h),
                       "embed_scale" if family == "gemma2" else None,
                       {"parameter": "embed_tokens", "vocab_size": spec["vocab_size"],
                        "scale": h ** 0.5 if family == "gemma2" else 1.0})
    next_caches = {}

    def norm(name: str, source: str, parameter: str) -> str:
        return graph.add(name, "rmsnorm", (source,), (batch, tokens, h),
                         "rmsnorm_gemma" if family == "gemma2" else "rmsnorm_qkv_fused",
                         {"parameter": parameter, "epsilon": spec["rms_norm_eps"],
                          "gemma_weight_offset": family == "gemma2"})

    def linear(name: str, source: str, shape: tuple[int, ...],
               parameter: str) -> str:
        in_width = graph.tensors[source]["shape"][-1]
        out_width = 1
        for width in shape[2:]:
            out_width *= width
        return graph.add(name, "linear", (source,), shape, "gemm_mxgemmini",
                         {"parameter": parameter, "weight_shape": [in_width, out_width],
                          "macs": batch * tokens * in_width * out_width})

    for layer in range(spec["num_hidden_layers"]):
        base = f"{phase}.layer{layer:02d}"
        attn_in = norm(f"{base}.attn_norm", hidden, f"layers.{layer}.attn_norm")
        projections = []
        for kind, heads in (("q", qh), ("k", kvh), ("v", kvh)):
            value = linear(f"{base}.{kind}_proj", attn_in,
                           (batch, tokens, heads, d), f"layers.{layer}.{kind}_proj")
            if spec.get("qkv_bias"):
                value = graph.add(f"{base}.{kind}_bias", "bias_add", (value,),
                                  (batch, tokens, heads, d), "qkv_bias",
                                  {"parameter": f"layers.{layer}.{kind}_bias"})
            projections.append(value)
        q, k, v = projections
        q = graph.add(f"{base}.q_rope", "rope", (q,), (batch, tokens, qh, d),
                      "rope_qkv_fused", {"position_start": past})
        k = graph.add(f"{base}.k_rope", "rope", (k,), (batch, tokens, kvh, d),
                      "rope_qkv_fused", {"position_start": past})
        old_k, old_v = caches.get(layer, (None, None))
        k_cache = graph.add(f"{base}.k_cache", "kv_append",
                            (k,) if old_k is None else (old_k, k),
                            (batch, past + tokens, kvh, d))
        v_cache = graph.add(f"{base}.v_cache", "kv_append",
                            (v,) if old_v is None else (old_v, v),
                            (batch, past + tokens, kvh, d))
        next_caches[layer] = (k_cache, v_cache)
        attn_kernel = "flash_attention_mx_gemma" if family == "gemma2" else (
            "flash_attention_mx_gqa" if d == 64 else None)
        attn = graph.add(f"{base}.attention", "causal_gqa",
                         (q, k_cache, v_cache), (batch, tokens, qh * d), attn_kernel,
                         {"query_start": past, "q_heads": qh, "kv_heads": kvh,
                          "head_dim": d,
                          "window": spec.get("sliding_window") if family == "gemma2"
                          and layer % 2 == 0 else None,
                          "softcap": spec.get("attention_softcap")})
        attn_out = linear(f"{base}.o_proj", attn, (batch, tokens, h),
                          f"layers.{layer}.o_proj")
        if family == "gemma2":
            attn_out = norm(f"{base}.post_attn_norm", attn_out,
                            f"layers.{layer}.post_attn_norm")
        hidden_after_attn = graph.add(f"{base}.attn_residual", "add",
                                      (hidden, attn_out), (batch, tokens, h), "vecadd")
        ffn_in = norm(f"{base}.ffn_norm", hidden_after_attn,
                      f"layers.{layer}.ffn_norm")
        gate = linear(f"{base}.gate_proj", ffn_in, (batch, tokens, f),
                      f"layers.{layer}.gate_proj")
        up = linear(f"{base}.up_proj", ffn_in, (batch, tokens, f),
                    f"layers.{layer}.up_proj")
        act = graph.add(f"{base}.activation", "gated_activation", (gate, up),
                        (batch, tokens, f), "geglu" if family == "gemma2" else "swiglu",
                        {"function": spec["activation"]})
        down = linear(f"{base}.down_proj", act, (batch, tokens, h),
                      f"layers.{layer}.down_proj")
        if family == "gemma2":
            down = norm(f"{base}.post_ffn_norm", down,
                        f"layers.{layer}.post_ffn_norm")
        hidden = graph.add(f"{base}.ffn_residual", "add",
                           (hidden_after_attn, down), (batch, tokens, h), "vecadd")
    hidden = norm(f"{phase}.final_norm", hidden, "final_norm")
    logits = linear(f"{phase}.lm_head", hidden,
                    (batch, tokens, spec["vocab_size"]), "lm_head")
    if family == "gemma2":
        logits = graph.add(f"{phase}.logit_softcap", "softcap", (logits,),
                           (batch, tokens, spec["vocab_size"]), "logit_softcap",
                           {"cap": spec["final_logit_softcap"]})
    return logits, next_caches


def decoder_graph(model: str, spec: dict, batch: int, prefill: int,
                  decode_steps: int, generation: str = "teacher_forced") -> Graph:
    if min(batch, prefill) <= 0 or decode_steps < 0:
        raise ValueError("batch and prefill must be positive; decode_steps must be nonnegative")
    if spec["num_attention_heads"] % spec["num_key_value_heads"]:
        raise ValueError("query heads must be a multiple of KV heads")
    if generation not in ("teacher_forced", "greedy"):
        raise ValueError("generation must be teacher_forced or greedy")
    graph = Graph(model, spec)
    graph.generation = generation
    logits, caches = _decoder_pass(graph, spec, "prefill", batch, prefill, 0, {})
    graph.outputs.append(logits)
    for step in range(decode_steps):
        token_ids = None
        if generation == "greedy":
            token_ids = graph.add(f"decode{step}.token_ids", "token_select", (logits,),
                                  (batch, 1), attrs={"method": "argmax_last_position"},
                                  dtype="int32")
        logits, caches = _decoder_pass(graph, spec, f"decode{step}",
                                       batch, 1, prefill + step, caches, token_ids)
        graph.outputs.append(logits)
    return graph


def smolvla_graph(model: str, spec: dict, batch: int) -> Graph:
    if batch <= 0:
        raise ValueError("batch must be positive")
    graph = Graph(model, spec)
    image_size, patch = spec["image_size"], spec["patch_size"]
    patches = (image_size // patch) ** 2
    image_tokens = patches // 16  # SmolVLM2 pixel shuffle factor 4 in each spatial axis.
    prefix_parts = []
    for camera in range(spec["image_cameras"]):
        image = graph.input(f"camera{camera}.image", (batch, 3, image_size, image_size), "fp32")
        graph.input(f"camera{camera}.valid", (batch,), "bool")
        tokens = graph.add(f"camera{camera}.patch_embed", "patch_embed", (image,),
                           (batch, patches, spec["vision_hidden_size"]), "patch_embed")
        tokens = graph.add(f"camera{camera}.vision_encoder", "vision_encoder", (tokens,),
                           (batch, patches, spec["vision_hidden_size"]))
        tokens = graph.add(f"camera{camera}.connector", "pixel_shuffle_connector",
                           (tokens,), (batch, image_tokens, spec["vlm_hidden_size"]))
        prefix_parts.append(tokens)
    language = graph.input("language.ids", (batch, spec["language_tokens"]), "int32")
    graph.input("language.valid", (batch, spec["language_tokens"]), "bool")
    language = graph.add("language.embedding", "embedding", (language,),
                         (batch, spec["language_tokens"], spec["vlm_hidden_size"]))
    state = graph.input("robot.state", (batch, spec["action_dim"]), "fp32")
    state = graph.add("robot.state_proj", "linear", (state,),
                      (batch, 1, spec["vlm_hidden_size"]), "gemm_mxgemmini")
    prefix_length = spec["image_cameras"] * image_tokens + spec["language_tokens"] + 1
    prefix = graph.add("prefix.merge", "multimodal_merge",
                       tuple(prefix_parts + [language, state] +
                             [f"camera{camera}.valid" for camera in range(spec["image_cameras"])] +
                             ["language.valid"]),
                       (batch, prefix_length, spec["vlm_hidden_size"]))
    # The VLM prefill saves K/V separately at every layer. The cross-attention
    # expert layers read these saved tensors without appending suffix tokens.
    cache_shape = (batch, prefix_length, spec["vlm_num_key_value_heads"],
                   spec["vlm_head_dim"])
    prefix_caches = {}
    for layer in range(spec["num_vlm_layers"]):
        layer_input = prefix
        prefix = graph.add(f"vlm.layer{layer:02d}", "vlm_decoder_layer", (prefix,),
                           (batch, prefix_length, spec["vlm_hidden_size"]))
        key = graph.add(f"vlm.layer{layer:02d}.k_cache", "cache_capture", (layer_input,),
                        cache_shape, attrs={"component": "k", "layer": layer})
        value = graph.add(f"vlm.layer{layer:02d}.v_cache", "cache_capture", (layer_input,),
                          cache_shape, attrs={"component": "v", "layer": layer})
        prefix_caches[layer] = (key, value)
    action = graph.input("action.noise", (batch, spec["chunk_size"],
                                          spec["action_dim"]), "fp32")
    for step in range(spec["num_denoise_steps"]):
        timestep = graph.input(f"denoise{step}.timestep", (batch,), "fp32")
        expert = graph.add(f"denoise{step}.action_embed", "action_time_embed",
                           (action, timestep),
                           (batch, spec["chunk_size"], spec["expert_hidden_size"]),
                           attrs={"time": 1.0 - step / spec["num_denoise_steps"]})
        for layer in range(spec["num_expert_layers"]):
            mode = "self" if layer % spec["self_attn_every_n_layers"] == 0 else "cross"
            key, value = prefix_caches[layer]
            if mode == "self":
                extended_shape = (batch, prefix_length + spec["chunk_size"],
                                  spec["vlm_num_key_value_heads"], spec["vlm_head_dim"])
                key = graph.add(f"denoise{step}.expert{layer:02d}.k_append",
                                "suffix_kv_append", (key, expert), extended_shape,
                                attrs={"component": "k", "layer": layer})
                value = graph.add(f"denoise{step}.expert{layer:02d}.v_append",
                                  "suffix_kv_append", (value, expert), extended_shape,
                                  attrs={"component": "v", "layer": layer})
            expert = graph.add(f"denoise{step}.expert{layer:02d}",
                               f"expert_{mode}_attention_layer", (expert, key, value),
                               (batch, spec["chunk_size"], spec["expert_hidden_size"]),
                               attrs={"layer": layer, "cache_mode":
                                      "temporary_suffix" if mode == "self" else "read_prefix"})
            if mode == "self":
                # LeRobot crops the in-place cache to its prefix length after
                # each denoising step, so the next step sees the same prefix.
                prefix_caches[layer] = (
                    graph.add(f"denoise{step}.expert{layer:02d}.k_crop",
                              "cache_crop", (key, expert), cache_shape,
                              attrs={"prefix_length": prefix_length}),
                    graph.add(f"denoise{step}.expert{layer:02d}.v_crop",
                              "cache_crop", (value, expert), cache_shape,
                              attrs={"prefix_length": prefix_length}),
                )
        velocity = graph.add(f"denoise{step}.action_out", "linear", (expert,),
                             (batch, spec["chunk_size"], spec["action_dim"]),
                             "gemm_mxgemmini")
        action = graph.add(f"denoise{step}.euler", "euler_step", (action, velocity),
                           (batch, spec["chunk_size"], spec["action_dim"]),
                           attrs={"step_size": -1 / spec["num_denoise_steps"]})
    graph.outputs.append(action)
    return graph


def build(model: str, batch: int = 1, prefill: int = 16,
          decode_steps: int = 1, specs: dict | None = None,
          generation: str = "teacher_forced") -> Graph:
    specs = model_specs() if specs is None else specs
    spec = specs[model]
    if spec["family"] == "smolvla":
        return smolvla_graph(model, spec, batch)
    return decoder_graph(model, spec, batch, prefill, decode_steps, generation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=model_specs(), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prefill", type=int, default=16)
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--generation", choices=("teacher_forced", "greedy"),
                        default="teacher_forced")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = build(args.model, args.batch, args.prefill, args.decode_steps,
                   generation=args.generation).report()
    output = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
