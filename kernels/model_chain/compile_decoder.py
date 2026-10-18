#!/usr/bin/env python3
"""Compile a connected decoder graph into a Radiance SoC ELF.

The default uses reduced dimensions and deterministic weights to check stage
handoffs. A checkpoint build uses pinned full dimensions and an external
weight image; it requires a separate Cyclotron run for device validation.
Neither mode measures hardware performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernels/evaluation/llm"))
import reference  # noqa: E402
import stitch  # noqa: E402
import moe_graph_reference  # noqa: E402
import last_token_graph_reference  # noqa: E402
import olmo_graph_reference  # noqa: E402
import nemotron_graph_reference  # noqa: E402
from checkpoint import (MixedFP16Weights, QuantizedINT8Weights,  # noqa: E402
                        SafeTensorWeights, TiedFP16INT8GemmaWeights,
                        TiedINT8INT4GemmaWeights, TiedINT5INT6GemmaWeights)
from export_decoder_weights import WARP_STACK_BOTTOM, WARP_STACK_TOP  # noqa: E402
from split_decoder_weights import image_segments, verify_image, verify_image_placement  # noqa: E402
from mx_vpu import reference as mx_reference  # noqa: E402
from mx_vpu import fp32_reference  # noqa: E402


MODELS = ("tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma_2_2b_it",
          "qwen3_0_6b", "qwen3_30b_a3b", "mistral_7b_v03", "olmo3_7b",
          "nemotron3_nano_4b")


def symbol(name: str) -> str:
    return "v_" + re.sub(r"[^a-zA-Z0-9_]", "_", name)


def literal(value: float) -> str:
    text = format(float(np.float32(value)), ".9g")
    if "e" not in text and "." not in text:
        text += ".0"
    return text + "f"


def array_decl(name: str, data: np.ndarray | None, count: int,
               dtype: str = "float") -> str:
    allocated = (count + 15) // 16 * 16  # 16 × 4 B = one Muon cache line.
    if data is None:
        return f"alignas(64) __global {dtype} {name}[{allocated}] = {{0}};\n"
    flat = data.ravel()
    if dtype == "float":
        values = ", ".join(map(literal, flat))
    else:
        values = ", ".join(str(int(v)) for v in flat)
    return f"alignas(64) __global {dtype} {name}[{allocated}] = {{{values}}};\n"


def write_if_changed(path: Path, content: str) -> None:
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_backend_sources(manifest: dict) -> None:
    for name, expected in (manifest.get("fallback_call_source_sha256") or {}).items():
        if name not in ("fallback_abi.hpp", "fallback_runtime.cpp", "fallback_codegen.py") or sha256_file(HERE / name) != expected:
            raise ValueError("pure helper call source changed since generation; regenerate this target")
    if manifest.get("execution_backend") == "rakanic_mx_vpu":
        for name, expected in manifest["backend_source_sha256"].items():
            if sha256_file(HERE / "mx_vpu" / name) != expected:
                raise ValueError(f"backend source changed since generation: {name}; regenerate this target")


def param_data(stage: dict) -> tuple[str, np.ndarray] | None:
    op, attrs = stage["op"], stage["attrs"]
    if op == "embedding":
        width, vocab = stage["shape"][-1], attrs["vocab_size"]
        data = (reference.parameter(attrs["parameter"], (vocab, width))
                if attrs.get("tied_lm_head") else
                np.stack([reference.parameter(f"{attrs['parameter']}.{i}", (width,))
                          for i in range(vocab)]))
    elif op in ("rmsnorm", "mamba_gated_rmsnorm"):
        width = stage["shape"][-1]
        data = reference.parameter(attrs["parameter"], (width,)) + 1.0
    elif op == "linear":
        shape = tuple(attrs["weight_shape"])
        data = reference.parameter(attrs["parameter"], shape,
                                   0.05 if attrs.get("weight_transposed") else
                                   1.0 / np.sqrt(shape[0]))
    elif op == "bias_add":
        data = reference.parameter(attrs["parameter"], tuple(stage["shape"][2:]))
    elif op in ("expert_gate_up", "expert_down"):
        (name, shape), = attrs["parameters"].items()
        data = reference.parameter(name, tuple(shape))
        return symbol("parameter." + name), data
    else:
        return None
    return symbol("parameter." + attrs["parameter"]), data.astype(np.float32)


def extra_param_data(stage: dict) -> list[tuple[str, np.ndarray]]:
    attrs = stage["attrs"]
    if stage["op"] == "mamba_depthwise_conv":
        items = attrs["parameters"].items()
    elif stage["op"] == "mamba_selective_scan":
        heads = attrs["state_shape"][1]
        items = ((name, (heads,)) for name in attrs["parameters"])
    else:
        return []
    return [(symbol("parameter." + name),
             reference.parameter(name, tuple(shape))) for name, shape in items]


def extra_param_symbols(stage: dict, graph: stitch.Graph) -> list[str]:
    if stage["op"] == "mamba_depthwise_conv":
        names = stage["attrs"]["parameters"]
    elif stage["op"] == "mamba_selective_scan":
        names = stage["attrs"]["parameters"]
    elif stage["op"] == "mamba_scan_state":
        paired = next(s for s in graph.stages
                      if s["id"] == stage["attrs"]["paired_scan"])
        names = paired["attrs"]["parameters"]
    else:
        return []
    return [symbol("parameter." + name) for name in names]


def emit_stage(index: int, stage: dict, graph: stitch.Graph,
               parameter: str | None, parameter_dtype: str = "fp32",
               scale_parameter: str | None = None,
               batch_clusters: int = 1,
               execution_backend: str = "muon", mx_linear_policy: str = "all",
               fallback_call_abi: bool = False) -> str:
    op, attrs, shape = stage["op"], stage["attrs"], stage["shape"]
    def tensor_pointer(name: str) -> str:
        base = symbol(name)
        if batch_clusters == 1:
            return base
        size = int(np.prod(graph.tensors[name]["shape"]))
        stride = (size + 15) // 16 * 16
        return f"({base} + replica * {stride}u)"

    ins = [tensor_pointer(name) for name in stage["reads"]]
    out = tensor_pointer(stage["writes"])
    batch = shape[0]
    count = int(np.prod(shape))
    if op == "embedding":
        if parameter_dtype == "int5_group64":
            body = f"embedding_i5_group64({ins[0]}, {parameter}, {scale_parameter}, {out}, {count // shape[-1]}, {shape[-1]}, {literal(attrs['scale'])}, tid, tpb)"
        elif parameter_dtype == "int8_scaled":
            body = f"embedding_i8({ins[0]}, {parameter}, {scale_parameter}, {out}, {count // shape[-1]}, {shape[-1]}, {literal(attrs['scale'])}, tid, tpb)"
        else:
            method = "embedding_f16" if parameter_dtype == "fp16" else "embedding"
            body = f"{method}({ins[0]}, {parameter}, {out}, {count // shape[-1]}, {shape[-1]}, {literal(attrs.get('scale', 1.0))}, tid, tpb)"
    elif op == "rmsnorm":
        body = f"rmsnorm({ins[0]}, {parameter}, {out}, {count // shape[-1]}, {shape[-1]}, {literal(attrs['epsilon'])}, tid, tpb)"
    elif op == "linear":
        k = graph.tensors[stage["reads"][0]]["shape"][-1]
        n = int(np.prod(shape[2:]))
        if attrs.get("weight_transposed") and parameter_dtype == "fp32":
            body = f"linear_tied({ins[0]}, {parameter}, {out}, {count // n}, {k}, {n}, tid, tpb)"
        elif execution_backend == "muon" and attrs.get("weight_transposed") and parameter_dtype not in (
                "fp16", "fp16_tied", "int8_tied"):
            raise ValueError(f"{stage['id']}: unsupported tied weight format")
        elif parameter_dtype == "int6_scaled":
            body = f"linear_i6({ins[0]}, {parameter}, {scale_parameter}, {out}, {count // n}, {k}, {n}, tid, tpb)"
        elif parameter_dtype == "int5_tied":
            body = f"linear_i5_tied({ins[0]}, {parameter}, {scale_parameter}, {out}, {count // n}, {k}, {n}, tid, tpb)"
        elif parameter_dtype == "int4_scaled":
            body = f"linear_i4({ins[0]}, {parameter}, {scale_parameter}, {out}, {count // n}, {k}, {n}, tid, tpb)"
        elif parameter_dtype == "int8_tied":
            body = f"linear_i8_tied({ins[0]}, {parameter}, {scale_parameter}, {out}, {count // n}, {k}, {n}, tid, tpb)"
        elif parameter_dtype == "int8_scaled":
            body = f"linear_i8({ins[0]}, {parameter}, {scale_parameter}, {out}, {count // n}, {k}, {n}, tid, tpb)"
        elif parameter_dtype == "fp16_tied" or (attrs.get("weight_transposed")
                                                and parameter_dtype == "fp16"):
            body = f"linear_f16_tied({ins[0]}, {parameter}, {out}, {count // n}, {k}, {n}, tid, tpb)"
        elif parameter_dtype == "fp16":
            body = f"linear_f16({ins[0]}, {parameter}, {out}, {count // n}, {k}, {n}, tid, tpb)"
        else:
            body = f"linear({ins[0]}, {parameter}, nullptr, {out}, {count // n}, {k}, {n}, tid, tpb)"
    elif op == "bias_add":
        width = int(np.prod(shape[2:]))
        body = f"bias_add({ins[0]}, {parameter}, {out}, {count}, {width}, tid, tpb)"
    elif op == "rope":
        tokens, heads, width = shape[1:]
        body = f"rope({ins[0]}, v_rope_cos, v_rope_sin, {out}, {batch}, {tokens}, {heads}, {width}, {attrs['position_start']}, tid, tpb)"
    elif op == "reshape_heads":
        body = f"reshape_heads({ins[0]}, {out}, {count}, tid, tpb)"
    elif op == "rope_yarn":
        tokens, heads, width = shape[1:]
        body = f"rope({ins[0]}, v_olmo_cos, v_olmo_sin, {out}, {batch}, {tokens}, {heads}, {width}, {attrs['position_start']}, tid, tpb)"
    elif op == "kv_append":
        tokens = graph.tensors[stage["reads"][-1]]["shape"][1]
        past = shape[1] - tokens
        heads, width = shape[2:]
        old = ins[0] if len(ins) == 2 else "nullptr"
        current = ins[-1]
        body = f"kv_append({old}, {current}, {out}, {batch}, {past}, {tokens}, {heads}, {width}, tid, tpb)"
    elif op == "causal_gqa":
        q_tokens = graph.tensors[stage["reads"][0]]["shape"][1]
        kv_tokens = graph.tensors[stage["reads"][1]]["shape"][1]
        qh, kvh, width = attrs["q_heads"], attrs["kv_heads"], attrs["head_dim"]
        window = attrs["window"] or 0
        softcap = literal(attrs["softcap"] or 0.0)
        body = (f"causal_gqa({ins[0]}, {ins[1]}, {ins[2]}, {out}, {batch}, "
                f"{q_tokens}, {kv_tokens}, {qh}, {kvh}, {width}, "
                f"{attrs['query_start']}, {window}, {softcap}, tid, tpb)")
    elif op == "gated_activation":
        gelu = "true" if attrs["function"] == "gelu_tanh" else "false"
        body = f"gated_activation({ins[0]}, {ins[1]}, {out}, {count}, {gelu}, tid, tpb)"
    elif op == "add":
        body = f"residual({ins[0]}, {ins[1]}, {out}, {count}, tid, tpb)"
    elif op == "softcap":
        body = f"softcap({ins[0]}, {out}, {count}, {literal(attrs['cap'])}, tid, tpb)"
    elif op == "token_select":
        in_shape = graph.tensors[stage["reads"][0]]["shape"]
        body = f"token_select({ins[0]}, {out}, {batch}, {in_shape[1]}, {in_shape[2]}, tid, tpb)"
    elif op == "select_last_token":
        in_shape = graph.tensors[stage["reads"][0]]["shape"]
        body = f"select_last_token({ins[0]}, {out}, {batch}, {in_shape[1]}, {in_shape[2]}, tid, tpb)"
    elif op == "softmax":
        body = f"moe_softmax({ins[0]}, {out}, {count // shape[-1]}, {shape[-1]}, tid, tpb)"
    elif op == "topk_indices":
        experts = graph.tensors[stage["reads"][0]]["shape"][-1]
        body = f"moe_topk_indices({ins[0]}, {out}, {count // shape[-1]}, {experts}, {shape[-1]}, tid, tpb)"
    elif op == "topk_scores":
        experts = graph.tensors[stage["reads"][0]]["shape"][-1]
        normalize = "true" if attrs["renormalize"] else "false"
        body = f"moe_topk_scores({ins[0]}, {ins[1]}, {out}, {count // shape[-1]}, {experts}, {shape[-1]}, {normalize}, tid, tpb)"
    elif op == "expert_group":
        body = f"moe_expert_group({ins[0]}, {ins[1]}, {out}, {count // (shape[-2] * shape[-1])}, {shape[-2]}, {shape[-1]}, {attrs['expert_count']}, tid, tpb)"
    elif op == "expert_gate_up":
        hidden = graph.tensors[stage["reads"][0]]["shape"][-1]
        body = f"moe_expert_gate_up({ins[0]}, {ins[1]}, {parameter}, {out}, {count // (shape[-2] * shape[-1])}, {shape[-2]}, {hidden}, {shape[-1] // 2}, tid, tpb)"
    elif op == "expert_swiglu":
        body = f"moe_expert_swiglu({ins[0]}, {out}, {count}, {shape[-1]}, tid, tpb)"
    elif op == "expert_down":
        intermediate = graph.tensors[stage["reads"][0]]["shape"][-1]
        body = f"moe_expert_down({ins[0]}, {ins[1]}, {parameter}, {out}, {count // (shape[-2] * shape[-1])}, {shape[-2]}, {intermediate}, {shape[-1]}, tid, tpb)"
    elif op == "expert_weighted_reduce":
        body = f"moe_expert_reduce({ins[0]}, {ins[1]}, {out}, {count // shape[-1]}, {attrs['selected_experts']}, {shape[-1]}, tid, tpb)"
    elif op == "mamba_projection_slice":
        spec = graph.spec
        inner = spec["mamba_num_heads"] * spec["mamba_head_dim"]
        conv = inner + 2 * spec["n_groups"] * spec["ssm_state_size"]
        start = {"gate": 0, "conv_input": inner, "dt": inner + conv}[attrs["part"]]
        in_width = graph.tensors[stage["reads"][0]]["shape"][-1]
        body = f"mamba_slice({ins[0]}, {out}, {batch}, {shape[1]}, {in_width}, {start}, {shape[-1]}, tid, tpb)"
    elif op == "mamba_depthwise_conv":
        params = attrs["parameters"]
        weight = symbol("parameter." + next(k for k in params if k.endswith(".weight")))
        bias = symbol("parameter." + next(k for k in params if k.endswith(".bias")))
        old = ins[1] if len(ins) == 2 else "nullptr"
        body = f"mamba_depthwise_conv({ins[0]}, {old}, {weight}, {bias}, {out}, {batch}, {shape[1]}, {shape[2]}, {attrs['kernel_width']}, tid, tpb)"
    elif op == "mamba_conv_state":
        old = ins[1] if len(ins) == 3 else "nullptr"
        tokens = graph.tensors[stage["reads"][0]]["shape"][1]
        body = f"mamba_conv_state({ins[0]}, {old}, {out}, {batch}, {tokens}, {shape[1]}, {shape[2]}, tid, tpb)"
    elif op == "mamba_channel_slice":
        spec = graph.spec
        inner = spec["mamba_num_heads"] * spec["mamba_head_dim"]
        grouped = spec["n_groups"] * spec["ssm_state_size"]
        start = {"x": 0, "B": inner, "C": inner + grouped}[attrs["part"]]
        in_width = graph.tensors[stage["reads"][0]]["shape"][-1]
        body = f"mamba_slice({ins[0]}, {out}, {batch}, {shape[1]}, {in_width}, {start}, {count // (batch * shape[1])}, tid, tpb)"
    elif op in ("mamba_selective_scan", "mamba_scan_state"):
        scan = (stage if op == "mamba_selective_scan" else
                next(s for s in graph.stages if s["id"] == attrs["paired_scan"]))
        a_name, d_name, bias_name = scan["attrs"]["parameters"]
        x_shape = graph.tensors[stage["reads"][0]]["shape"]
        b_shape = graph.tensors[stage["reads"][1]]["shape"]
        old = (ins[4] if len(ins) == (5 if op == "mamba_selective_scan" else 6)
               else "nullptr")
        result = out if op == "mamba_selective_scan" else "nullptr"
        final = out if op == "mamba_scan_state" else "nullptr"
        body = (f"mamba_scan({ins[0]}, {ins[1]}, {ins[2]}, {ins[3]}, {old}, "
                f"{symbol('parameter.' + a_name)}, {symbol('parameter.' + d_name)}, "
                f"{symbol('parameter.' + bias_name)}, {result}, {final}, "
                f"{batch}, {x_shape[1]}, {x_shape[2]}, {x_shape[3]}, "
                f"{b_shape[2]}, {b_shape[3]}, 0.0f, 3.402823466e38f, tid, tpb)")
    elif op == "mamba_gated_rmsnorm":
        body = f"mamba_gated_norm({ins[0]}, {ins[1]}, {parameter}, {out}, {batch * shape[1]}, {shape[-1]}, {attrs['group_size']}, {literal(attrs['epsilon'])}, tid, tpb)"
    elif op == "relu_squared":
        body = f"relu_squared({ins[0]}, {out}, {count}, tid, tpb)"
    else:
        raise ValueError(f"unsupported operation {op} at {stage['id']}")
    if execution_backend == "rakanic_mx_vpu":
        if op == "linear" and (mx_linear_policy == "all" or attrs.get("parameter", "").endswith(".q_proj")):
            tied = attrs.get("weight_transposed", False)
            storage = {"fp32": "F32_TIED" if tied else "F32", "fp16": "F16_TIED" if tied else "F16",
                       "fp16_tied": "F16_TIED", "int8_scaled": "I8", "int4_scaled": "I4",
                       "int6_scaled": "I6", "int8_tied": "I8_TIED", "int5_tied": "I5_TIED"}[parameter_dtype]
            k = graph.tensors[stage["reads"][0]]["shape"][-1]
            n = int(np.prod(shape[2:]))
            body = (f"mx_vpu_linear({ins[0]}, {parameter}, {scale_parameter or 'nullptr'}, {out}, "
                    f"model_chain::mx_model::{storage}, {count // n}, {k}, {n}, tid, tpb)")
        elif op in ("add", "bias_add"):
            right = parameter if op == "bias_add" else ins[1]
            right_width = int(np.prod(shape[2:])) if op == "bias_add" else count
            body = f"mx_vpu_add({ins[0]}, {right}, {out}, {count}, {right_width}, tid, tpb)"
    descriptor, code = "", "model_chain::" + body + ";"
    if fallback_call_abi:
        from fallback_codegen import emit_call
        pure_body = body
        strides = {}
        if batch_clusters == 2:
            for name in (*stage["reads"], stage["writes"]):
                size = int(np.prod(graph.tensors[name]["shape"]))
                strides[symbol(name)] = (size + 15) // 16 * 16 * 4
                pure_body = pure_body.replace(tensor_pointer(name), symbol(name))
        wrapped = emit_call(pure_body + ";", f"rk_desc_stage_{index}",
                            context="context", replica_strides=strides)
        if wrapped is not None:
            descriptor, code = wrapped
    return (descriptor + f"static void stage_{index}(void* context, uint32_t tid, uint32_t tpb, uint32_t) {{\n"
            + ("  const uint32_t replica = (uint32_t)(uintptr_t)context;\n"
               if batch_clusters != 1 else "") +
            "#ifdef RADIANCE_DEVICE\n"
            "  // Each stage consumes the previous stage's GMEM output. Muon L0d is\n"
            "  // private per core, so keep this correctness build on core 0.\n"
            "  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return;\n"
            "  tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS\n"
            "      + tid % MU_NUM_THREADS;\n"
            "  tpb /= MU_NUM_CORES;\n"
            "#endif\n"
            f"  {code}\n}}\n")


def generate(model: str, layers: int, prefill: int, decode: int,
             generation: str, out_root: Path,
             device_check_all_stages: bool = False,
             device_opt: str = "O3", stages_per_object: int = 0,
             checkpoint_dir: Path | None = None,
             weight_image: Path | None = None,
             stage_limit: int | None = None,
             token_ids: list[int] | None = None,
             batch_clusters: int = 1,
             prefill_logits: str = "all",
             control_sliding_window: int | None = None,
             native_rtol: float = 1e-4,
             native_atol: float = 1e-5,
             execution_backend: str = "muon", mx_linear_policy: str = "all",
             fallback_call_abi: bool = False) -> Path:
    if execution_backend not in ("muon", "rakanic_mx_vpu"):
        raise ValueError("unknown execution backend")
    use_mx_vpu = execution_backend == "rakanic_mx_vpu"
    if fallback_call_abi and not use_mx_vpu:
        raise ValueError("pure helper ABI requires the explicit MX/VPU backend")
    if mx_linear_policy not in ("all", "q_proj"):
        raise ValueError("unknown MX linear policy")
    def mx_selected(stage):
        return stage["op"] == "linear" and (mx_linear_policy == "all" or
               stage["attrs"].get("parameter", "").endswith(".q_proj"))
    if use_mx_vpu and model not in MODELS[:3]:
        raise ValueError("MX+VPU decoder backend currently supports TinyLlama, DeepSeek and Gemma")

    def backend_source(source):
        if fallback_call_abi:
            source = '#include "fallback_abi.hpp"\n' + source
        return source.replace('#include "pipeline_math.hpp"\n',
                              '// MX variant: sequential FP32 fallbacks, FP contraction disabled.\n'
                              '#include "pipeline_math.hpp"\n#include "mx_vpu/backend.hpp"\n') if use_mx_vpu else source
    if device_opt not in ("O1", "O2", "O3"):
        raise ValueError(f"unsupported device optimization level: {device_opt}")
    if stages_per_object < 0:
        raise ValueError("stages per object cannot be negative")
    if batch_clusters not in (1, 2):
        raise ValueError("batch clusters must be one or two")
    if prefill_logits not in ("all", "last"):
        raise ValueError("prefill logits must be all or last")
    if use_mx_vpu and prefill_logits != "all":
        raise ValueError("MX+VPU format reference currently requires all prefill logits")
    if not (0 < native_rtol <= 5e-3 and 0 < native_atol <= 5e-4):
        raise ValueError("native tolerance must be positive and no looser than device checks")
    if control_sliding_window is not None and (
            model != "olmo3_7b" or control_sliding_window <= 0):
        raise ValueError("the positive control sliding window is only for OLMo 3")
    external = checkpoint_dir is not None or weight_image is not None
    if external and (checkpoint_dir is None or weight_image is None):
        raise ValueError("checkpoint directory and weight image are both required")
    if external and model == "qwen3_30b_a3b":
        raise ValueError("Qwen3 MoE checkpoint weight-image export is not implemented")
    if external and model == "olmo3_7b":
        raise ValueError("OLMo 3 checkpoint weight-image export is not implemented")
    if external and model == "nemotron3_nano_4b":
        raise ValueError("Nemotron H checkpoint weight-image export is not implemented")
    if external and control_sliding_window is not None:
        raise ValueError("the control sliding window requires synthetic weights")
    if stage_limit is not None and not external:
        raise ValueError("stage limit is only for checkpoint device probes")
    if external and (model not in MODELS or stages_per_object <= 0
                     or (batch_clusters == 1 and not device_check_all_stages)):
        raise ValueError("checkpoint builds require a pinned model, sharded objects, "
                         "and all-stage checks for one-cluster runs")
    specs = stitch.model_specs()
    spec = (dict(specs[model]) if external else
            nemotron_graph_reference.control_spec(specs[model], layers)
            if model == "nemotron3_nano_4b" else reference.reduced_spec(specs[model]))
    if model == "qwen3_30b_a3b" and not external:
        spec.update(num_experts=16, num_experts_per_tok=8,
                    moe_intermediate_size=16)
    if token_ids is not None:
        expected_ids = prefill + (decode if generation == "teacher_forced" else 0)
        if len(token_ids) != expected_ids:
            raise ValueError("explicit token IDs require one ID per prefill token "
                             "and, for teacher-forced generation, one per decode step")
        if any(token < 0 or token >= spec["vocab_size"] for token in token_ids):
            raise ValueError("token ID is outside the model vocabulary")
    if not 1 <= layers <= specs[model]["num_hidden_layers"]:
        raise ValueError("layer count exceeds the pinned model")
    spec["num_hidden_layers"] = layers
    if model == "olmo3_7b":
        spec["layer_types"] = spec["layer_types"][:layers]
        if len(spec["layer_types"]) != layers:
            raise ValueError("OLMo 3 layer-type sequence is shorter than requested depth")
        if control_sliding_window is not None:
            spec["sliding_window"] = control_sliding_window
    if model == "nemotron3_nano_4b":
        spec["layer_kinds"] = spec["layer_kinds"][:layers]
        if (len(spec["layer_kinds"]) != layers or spec["conv_kernel"] != 4
                or spec["ssm_state_size"] > 128):
            raise ValueError("unsupported Nemotron H layer or state dimensions")
    graph = stitch.build(model, prefill=prefill, decode_steps=decode,
                         specs={model: spec}, generation=generation,
                         prefill_logits=prefill_logits)
    if stage_limit is not None:
        if not 1 <= stage_limit <= len(graph.stages):
            raise ValueError("stage limit is outside the decoder graph")
        graph.stages = graph.stages[:stage_limit]
        graph.outputs = [graph.stages[-1]["writes"]]
    image_manifest = None
    addresses = {}
    provider = None
    if external:
        image_manifest = json.loads(weight_image.read_text())
        if (image_manifest["model"] != model or image_manifest["layers"] != layers
                or image_manifest["dtype"] not in ("fp32", "mixed_fp16", "int8_scaled",
                                                   "int8_fp16_tied", "int4_int8_tied",
                                                   "int5_group64_int6_tied")
                or (image_manifest["dtype"] == "int5_group64_int6_tied" and
                    model != "gemma_2_2b_it")
                or not image_manifest["checkpoint_weights"]
                or image_manifest["gpu_end_address_exclusive"] > (1 << 32)):
            raise ValueError("weight image does not match the full-dimension graph")
        image_parameters = {item["logical_name"]: item
                            for item in image_manifest["parameters"]}
        for stage in graph.stages:
            logical = stage["attrs"].get("parameter")
            if logical is None:
                continue
            if logical not in image_parameters:
                raise ValueError(f"weight image lacks {logical}")
            item = image_parameters[logical]
            begin = item["gpu_address"]
            end = begin + item["size_bytes"]
            if begin < WARP_STACK_TOP and end > WARP_STACK_BOTTOM:
                raise ValueError(
                    f"{logical}: weight [{begin:#x}, {end:#x}) overlaps "
                    f"Muon warp stacks [{WARP_STACK_BOTTOM:#x}, "
                    f"{WARP_STACK_TOP:#x}); choose a different placement")
        verify_image(weight_image)
        if sha256_file(checkpoint_dir / "config.json") != spec["source_sha256"]:
            raise ValueError("checkpoint config differs from pinned model")
        if "checkpoint_files_sha256" in image_manifest:
            actual_files = {name: sha256_file(checkpoint_dir / name)
                            for name in image_manifest["checkpoint_files_sha256"]}
            if actual_files != image_manifest["checkpoint_files_sha256"] or actual_files != spec[
                    "checkpoint_files_sha256"]:
                raise ValueError("checkpoint shards differ from pinned image source")
        elif sha256_file(checkpoint_dir / "model.safetensors") != image_manifest[
                "checkpoint_weight_sha256"]:
            raise ValueError("checkpoint weights differ from image source")
        addresses = {item["logical_name"]: item["gpu_address"]
                     for item in image_manifest["parameters"]}
        source_provider = SafeTensorWeights(
            checkpoint_dir if "checkpoint_files_sha256" in image_manifest
            else checkpoint_dir / "model.safetensors",
                                            spec["family"])
        provider = (MixedFP16Weights(source_provider)
                    if image_manifest["dtype"] == "mixed_fp16" else
                    TiedINT5INT6GemmaWeights(source_provider)
                    if image_manifest["dtype"] == "int5_group64_int6_tied" else
                    TiedINT8INT4GemmaWeights(source_provider)
                    if image_manifest["dtype"] == "int4_int8_tied" else
                    TiedFP16INT8GemmaWeights(source_provider)
                    if image_manifest["dtype"] == "int8_fp16_tied" else
                    QuantizedINT8Weights(source_provider)
                    if image_manifest["dtype"] == "int8_scaled" else source_provider)
    inputs = {name: np.arange(1, shape["shape"][1] + 1, dtype=np.int32)[None]
              for name, shape in graph.tensors.items()
              if name.endswith(".token_ids") and name.startswith(("prefill", "decode"))
              and name not in {stage["writes"] for stage in graph.stages}}
    if token_ids is not None:
        inputs["prefill.token_ids"] = np.asarray(token_ids[:prefill],
                                                  dtype=np.int32)[None]
        for step in range(decode if generation == "teacher_forced" else 0):
            name = f"decode{step}.token_ids"
            if name not in inputs:
                raise ValueError(f"graph has no teacher-forced input {name}")
            inputs[name] = np.asarray([token_ids[prefill + step]],
                                      dtype=np.int32)[None]
    def evaluate(weights, accelerator=False):
        if model == "qwen3_30b_a3b":
            return moe_graph_reference.execute(
                graph, inputs, retain=tuple(stage["writes"] for stage in graph.stages))
        if model == "olmo3_7b":
            return olmo_graph_reference.execute(graph, inputs, weights=weights)
        if model == "nemotron3_nano_4b":
            return nemotron_graph_reference.execute(graph, inputs, weights=weights)
        if prefill_logits == "last":
            return last_token_graph_reference.execute(
                graph, inputs, weights=weights,
                retain=tuple(stage["writes"] for stage in graph.stages))
        return reference.execute(graph, inputs, weights=weights,
                                 linear_fn=mx_reference.linear if accelerator else None,
                                 add_fn=mx_reference.add if accelerator else None,
                                 linear_stage_filter=mx_selected if accelerator else None,
                                 fallbacks=fp32_reference.FALLBACKS if accelerator else None)

    values = evaluate(provider)
    format_comparison = None
    if use_mx_vpu:
        baseline = values
        values = evaluate(provider, accelerator=True)
        selected = [stage["writes"] for stage in graph.stages if stage["op"] == "token_select"]
        format_comparison = dict(
            reference="independent NumPy block-scaled E4M3 mesh and BF16 add",
            reference_sha256=sha256_file(HERE / "mx_vpu/reference.py"),
            fallback_reference_sha256=sha256_file(HERE / "mx_vpu/fp32_reference.py"),
            greedy_tokens={name: dict(original=baseline[name].tolist(), variant=values[name].tolist(),
                                      exact_equal=bool(np.array_equal(baseline[name], values[name]))) for name in selected},
            outputs={name: dict(max_abs_error=float(np.max(np.abs(values[name]-baseline[name]))),
                                original_sha256=hashlib.sha256(baseline[name].tobytes()).hexdigest(),
                                variant_sha256=hashlib.sha256(values[name].tobytes()).hexdigest()) for name in graph.outputs})
        format_comparison["original_greedy_exact"] = (
            all(np.array_equal(baseline[name], values[name]) for name in selected)
            if generation == "greedy" else None)
    quantization_comparison = None
    if external and image_manifest["dtype"] in ("mixed_fp16", "int8_scaled",
                                                 "int8_fp16_tied", "int4_int8_tied",
                                                 "int5_group64_int6_tied"):
        unquantized = evaluate(source_provider)
        if use_mx_vpu:
            token_names = [s["writes"] for s in graph.stages if s["op"] == "token_select"]
            format_comparison["original_checkpoint_greedy_tokens"] = {
                name: dict(original=unquantized[name].tolist(), variant=values[name].tolist(),
                           exact_equal=bool(np.array_equal(unquantized[name], values[name]))) for name in token_names}
            format_comparison["original_checkpoint_greedy_exact"] = (
                all(np.array_equal(unquantized[name], values[name]) for name in token_names)
                if generation == "greedy" else None)
        quantization_comparison = {}
        for name in graph.outputs:
            actual = values[name].astype(np.float64)
            original = unquantized[name].astype(np.float64)
            delta = actual - original
            quantization_comparison[name] = {
                "max_abs_error": float(np.max(np.abs(delta))),
                "rms_error": float(np.sqrt(np.mean(delta * delta))),
                "reference_sha256": hashlib.sha256(
                    unquantized[name].tobytes()).hexdigest(),
                "quantized_sha256": hashlib.sha256(
                    values[name].tobytes()).hexdigest(),
                "top1_equal": bool(np.argmax(actual) == np.argmax(original)),
            }
    target = (out_root / model).resolve()
    target.mkdir(parents=True, exist_ok=True)
    declarations = []
    for name, tensor in graph.tensors.items():
        data = inputs.get(name)
        dtype = "int32_t" if tensor["dtype"] == "int32" else "float"
        count = int(np.prod(tensor["shape"]))
        stride = (count + 15) // 16 * 16
        if data is not None and batch_clusters == 2:
            padded = np.zeros(stride, dtype=data.dtype)
            padded[:count] = data.ravel()
            data = np.tile(padded, batch_clusters)
        declarations.append(array_decl(symbol(name), data,
                                       stride * batch_clusters, dtype))
    if not external:
        params = {}
        for stage in graph.stages:
            item = param_data(stage)
            items = [item] if item is not None else []
            items.extend(extra_param_data(stage))
            for name, data in items:
                if name in params:
                    if not np.array_equal(params[name], data):
                        raise ValueError(f"inconsistent parameter {name}")
                else:
                    params[name] = data
        for name, data in params.items():
            declarations.append(array_decl(name, data, data.size))

    def parameter_symbol(stage: dict) -> str | None:
        if external:
            logical = stage["attrs"].get("parameter")
            if logical is None:
                return None
            if logical not in addresses:
                raise ValueError(f"weight image lacks {logical}")
            return symbol("parameter." + logical)
        item = param_data(stage)
        return item[0] if item else None

    def scale_symbol(stage: dict) -> str | None:
        if not external:
            return None
        logical = stage["attrs"].get("parameter")
        if logical is None or image_parameters[logical].get("scale_gpu_address") is None:
            return None
        return symbol("scale." + logical)

    def parameter_dtype(stage: dict) -> str:
        if not external:
            return "fp32"
        logical = stage["attrs"].get("parameter")
        return image_parameters[logical].get("storage_dtype", "fp32") if logical else "fp32"
    half = spec["head_dim"] // 2
    max_position = prefill + decode
    inv_freq = np.power(spec.get("rope_theta", 10000.0),
                        -np.arange(half, dtype=np.float32) / half)
    angles = np.arange(max_position, dtype=np.float32)[:, None] * inv_freq
    declarations.append(array_decl("v_rope_cos", np.cos(angles), max_position * half))
    declarations.append(array_decl("v_rope_sin", np.sin(angles), max_position * half))
    if model == "olmo3_7b":
        inverse = olmo_graph_reference.yarn_frequencies(
            spec["rope_scaling"], spec["rope_theta"], spec["head_dim"])
        olmo_angles = np.arange(max_position, dtype=np.float32)[:, None] * inverse
        factor = np.float32(spec["rope_scaling"]["attention_factor"])
        declarations.append(array_decl("v_olmo_cos", np.cos(olmo_angles) * factor,
                                       max_position * half))
        declarations.append(array_decl("v_olmo_sin", np.sin(olmo_angles) * factor,
                                       max_position * half))
    stage_src = []
    uses_int8 = external and image_manifest["dtype"] in ("int8_scaled", "int8_fp16_tied",
                                                        "int4_int8_tied",
                                                        "int5_group64_int6_tied")
    uses_int4 = external and image_manifest["dtype"] == "int4_int8_tied"
    uses_packed = external and image_manifest["dtype"] == "int5_group64_int6_tied"
    uses_tied = external and image_manifest["dtype"] in ("int8_fp16_tied",
                                                         "int4_int8_tied")
    uses_tied = uses_tied or (external and any(
        stage["attrs"].get("weight_transposed") and
        parameter_dtype(stage) == "fp16" for stage in graph.stages))
    uses_fp32_tied = any(stage["attrs"].get("weight_transposed") and
                         parameter_dtype(stage) == "fp32" for stage in graph.stages)
    uses_moe = any(stage["op"] in moe_graph_reference.moe_ops.OPS
                   for stage in graph.stages)
    uses_select_last = any(stage["op"] == "select_last_token"
                           for stage in graph.stages)
    uses_olmo = model == "olmo3_7b"
    uses_nemotron = model == "nemotron3_nano_4b"
    for i, stage in enumerate(graph.stages):
        stage_src.append(emit_stage(i, stage, graph, parameter_symbol(stage),
                                    parameter_dtype(stage), scale_symbol(stage),
                                    batch_clusters, execution_backend, mx_linear_policy, fallback_call_abi))
    stage_files = []
    if stages_per_object:
        for first in range(0, len(graph.stages), stages_per_object):
            last = min(first + stages_per_object, len(graph.stages))
            filename = f"stage_chunk_{first // stages_per_object:03d}.cpp"
            stage_files.append(filename)
            symbols = {}
            external_params = {}
            for stage in graph.stages[first:last]:
                for tensor_name in (*stage["reads"], stage["writes"]):
                    tensor = graph.tensors[tensor_name]
                    symbols[symbol(tensor_name)] = (
                        "int32_t" if tensor["dtype"] == "int32" else "float")
                param = parameter_symbol(stage)
                if param:
                    if external:
                        external_params[param] = (
                            addresses[stage["attrs"]["parameter"]],
                            parameter_dtype(stage))
                        scale = scale_symbol(stage)
                        if scale:
                            external_params[scale] = (
                                image_parameters[stage["attrs"]["parameter"]]
                                ["scale_gpu_address"], "fp32")
                    else:
                        symbols[param] = "float"
                for name in extra_param_symbols(stage, graph):
                    symbols[name] = "float"
                if stage["op"] == "rope":
                    symbols["v_rope_cos"] = "float"
                    symbols["v_rope_sin"] = "float"
                if stage["op"] == "rope_yarn":
                    symbols["v_olmo_cos"] = "float"
                    symbols["v_olmo_sin"] = "float"
            externs = "\n".join(f"extern __global {dtype} {name}[];"
                                for name, dtype in sorted(symbols.items()))
            pointers = "\n".join(
                f"#define {name} ((const __global "
                f"{'uint16_t' if dtype in ('fp16', 'fp16_tied') else 'uint8_t' if dtype in ('int4_scaled', 'int5_group64', 'int5_tied', 'int6_scaled') else 'int8_t' if dtype in ('int8_scaled', 'int8_tied') else 'float'}*)0x{address:08x}u)"
                for name, (address, dtype) in sorted(external_params.items()))
            source = ("#include <mu_intrinsics.h>\n#include <mu_schedule.h>\n"
                      "#include <stdint.h>\n#include \"pipeline_math.hpp\"\n"
                      + ("#include \"pipeline_moe.hpp\"\n" if uses_moe else "")
                      + ("#include \"pipeline_select_last.hpp\"\n"
                         if uses_select_last else "")
                      + ("#include \"pipeline_olmo.hpp\"\n" if uses_olmo else "")
                      + ("#include \"pipeline_nemotron.hpp\"\n" if uses_nemotron else "")
                      + ("#include \"pipeline_int8.hpp\"\n" if uses_int8 else "")
                      + ("#include \"pipeline_int4.hpp\"\n" if uses_int4 else "")
                      + ("#include \"pipeline_packed.hpp\"\n" if uses_packed else "")
                      + ("#include \"pipeline_tied.hpp\"\n" if uses_tied else "")
                      + ("#include \"pipeline_fp32_tied.hpp\"\n" if uses_fp32_tied else "")
                      + externs + "\n" + pointers + "\n"
                      + "\n".join(item.replace("static void stage_", "void stage_", 1)
                                  for item in stage_src[first:last]))
            write_if_changed(target / filename, backend_source(source))
    device_checks = []
    for name in graph.outputs:
        gold = values[name].astype(np.float32)
        gold_name = symbol("gold." + name)
        declarations.append(array_decl(gold_name, gold, gold.size))
        device_checks.append((symbol(name), gold_name, gold.size))
    for stage in graph.stages:
        if stage["op"] in ("causal_gqa", "gated_activation"):
            name = stage["writes"]
            gold = values[name].astype(np.float32)
            gold_name = symbol("gold." + name)
            declarations.append(array_decl(gold_name, gold, gold.size))
            device_checks.append((symbol(name), gold_name, gold.size))
    native_checks = []
    native_extra = []
    device_names = {out for out, _, _ in device_checks}
    for stage in graph.stages:
        name = stage["writes"]
        if graph.tensors[name]["dtype"] == "int32":
            continue
        gold_name = symbol("gold." + name)
        gold = values[name].astype(np.float32)
        if symbol(name) not in device_names:
            native_extra.append(array_decl(gold_name, gold, gold.size))
        native_checks.append((symbol(name), gold_name, gold.size))
    integer_checks = []
    for stage in graph.stages:
        name = stage["writes"]
        if graph.tensors[name]["dtype"] != "int32":
            continue
        gold = values[name].astype(np.int32)
        gold_name = symbol("gold." + name)
        native_extra.append(array_decl(gold_name, gold, gold.size, "int32_t"))
        integer_checks.append((symbol(name), gold_name, gold.size))
    if device_check_all_stages:
        declarations.extend(native_extra)
    device_source_files = list(stage_files)
    if stage_files:
        data_file = "model_data.cpp"
        write_if_changed(target / data_file,
                         "#include <mu_intrinsics.h>\n#include <stdint.h>\n"
                         + "\n".join(declarations))
        device_source_files.append(data_file)
        externs = []
        for declaration in declarations:
            match = re.match(r"alignas\(64\) __global (float|int32_t) (\w+)\[",
                             declaration)
            if not match:
                raise ValueError("generated data declaration cannot be externed")
            externs.append(f"extern __global {match.group(1)} {match.group(2)}[];")
        device_declarations = "\n".join(externs)
    else:
        device_declarations = "\n".join(declarations)
    schedule = "\n".join(f"  mu_schedule(stage_{i}, "
                         + ("(void*)(uintptr_t)replica" if batch_clusters == 2
                            else "nullptr") + ", 1);\n"
                         + "  mu_barrier(0, MU_NUM_CORES);\n  mu_fence();"
                         for i in range(len(graph.stages)))
    if device_check_all_stages:
        def actual(out, count):
            return (f"{out}[i + replica * {((count + 15) // 16 * 16)}u]"
                    if batch_clusters == 2 else f"{out}[i]")
        float_verify = "\n".join(
            f"  errors = 0; uint32_t first_{index} = {count};\n"
            f"  for (uint32_t i = 0; i < {count}; ++i) "
            f"if (!mu_close({actual(out, count)}, {gold}[i], 5e-3f, 5e-4f)) "
            f"{{ if (!errors) first_{index} = i; ++errors; }}\n"
            f"  if (errors) mu_tohost(({index + 1}u << 16) | (first_{index} << 1) | 1u);"
            for index, (out, gold, count) in enumerate(native_checks))
        int_verify = "\n".join(
            f"  errors = 0; uint32_t first_int_{index} = {count};\n"
            f"  for (uint32_t i = 0; i < {count}; ++i) "
            f"if ({actual(out, count)} != {gold}[i]) "
            f"{{ if (!errors) first_int_{index} = i; ++errors; }}\n"
            f"  if (errors) mu_tohost(({len(native_checks) + index + 1}u << 16) "
            f"| (first_int_{index} << 1) | 1u);"
            for index, (out, gold, count) in enumerate(integer_checks))
        verify = float_verify + ("\n" + int_verify if int_verify else "")
    else:
        verify = "\n".join(
            f"  for (uint32_t i = 0; i < {count}; ++i) "
            f"if (!mu_close({out}[i"
            + (f" + replica * {((count + 15) // 16 * 16)}u" if batch_clusters == 2
               else "") + f"], {gold}[i], 5e-3f, 5e-4f)) ++errors;"
            for out, gold, count in device_checks)
    if stage_files and device_check_all_stages:
        all_checks = [(out, gold, count, "float")
                      for out, gold, count in native_checks]
        all_checks += [(out, gold, count, "int32_t")
                       for out, gold, count in integer_checks]
        verifier_files = []
        for first in range(0, len(all_checks), stages_per_object):
            selected = all_checks[first:first + stages_per_object]
            filename = f"verify_chunk_{first // stages_per_object:03d}.cpp"
            verifier_files.append(filename)
            externs = {name: dtype for out, gold, _, dtype in selected
                       for name in (out, gold)}
            checks = []
            for offset, (out, gold, count, dtype) in enumerate(selected):
                index = first + offset
                comparison = (f"!mu_close({actual(out, count)}, {gold}[i], 5e-3f, 5e-4f)"
                              if dtype == "float" else f"{actual(out, count)} != {gold}[i]")
                checks.append(
                    f"  errors = 0; uint32_t first_{index} = {count};\n"
                    f"  for (uint32_t i = 0; i < {count}; ++i) "
                    f"if ({comparison}) "
                    f"{{ if (!errors) first_{index} = i; ++errors; }}\n"
                    f"  if (errors) {{ mu_tohost(({index + 1}u << 16) | "
                    f"(first_{index} << 1) | 1u); return false; }}")
            source = ("#include <mu_intrinsics.h>\n#include <stdint.h>\n"
                      "#include \"kernel_verify.h\"\n"
                      + "\n".join(f"extern __global {dtype} {name}[];"
                                  for name, dtype in sorted(externs.items()))
                      + f"\nbool verify_chunk_{first // stages_per_object:03d}({'uint32_t replica' if batch_clusters == 2 else ''}) {{\n"
                      "  uint32_t errors = 0;\n" + "\n".join(checks)
                      + "\n  return true;\n}\n")
            write_if_changed(target / filename, source)
        device_source_files.extend(verifier_files)
        device_declarations = ""
        verify = "\n".join(
            f"  if (!verify_chunk_{index:03d}({'replica' if batch_clusters == 2 else ''})) return 0;"
            for index in range(len(verifier_files)))
        verify_prototypes = "\n".join(
            f"bool verify_chunk_{index:03d}({'uint32_t' if batch_clusters == 2 else ''});"
            for index in range(len(verifier_files)))
    else:
        verify_prototypes = ""
    device_stages = ("\n".join(f"void stage_{i}(void*, uint32_t, uint32_t, uint32_t);"
                                for i in range(len(stage_src))) if stage_files
                     else "\n".join(stage_src))
    cpp = ("#include <mu_intrinsics.h>\n#include <mu_schedule.h>\n"
           "#include <stdint.h>\n#include \"kernel_verify.h\"\n"
           + ("#include \"model_cluster_id.h\"\n" if batch_clusters == 2 else "") +
           "#include \"pipeline_math.hpp\"\n"
           + ("#include \"pipeline_moe.hpp\"\n" if uses_moe else "")
           + ("#include \"pipeline_select_last.hpp\"\n"
              if uses_select_last else "")
           + ("#include \"pipeline_nemotron.hpp\"\n" if uses_nemotron else "")
           + ("#include \"pipeline_int8.hpp\"\n" if uses_int8 else "")
           + ("#include \"pipeline_int4.hpp\"\n" if uses_int4 else "")
           + ("#include \"pipeline_packed.hpp\"\n" if uses_packed else "")
           + ("#include \"pipeline_tied.hpp\"\n" if uses_tied else "")
           + ("#include \"pipeline_fp32_tied.hpp\"\n" if uses_fp32_tied else "")
           + "extern \"C\" uint32_t __mu_num_warps = 1;\n"
           + device_declarations + "\n" + device_stages + "\n" + verify_prototypes +
           "\nint main() {\n"
           + ("  const uint32_t replica = model_chain_cluster_id();\n"
              if batch_clusters == 2 else "") + schedule +
           ("\n  if (vx_core_id() != 0) { for (;;) {} }\n" if batch_clusters == 2
            else "\n  if (mu_hart_id() != 0) { for (;;) {} }\n") +
           "  uint32_t errors = 0;\n" + verify +
           "\n  mu_tohost(errors ? ((errors << 1) | 1u) : 0u);\n  return 0;\n}\n")
    write_if_changed(target / "kernel.cpp", backend_source(cpp))
    native_check_source = "\n".join(
        f"  for (uint32_t i = 0; i < {count}; ++i) {{\n"
        f"    const float delta = __builtin_fabsf({out}[i"
        + (f" + {replica * ((count + 15) // 16 * 16)}u" if batch_clusters == 2
           else "") + f"] - {gold}[i]);\n"
        f"    if (delta > maximum) maximum = delta;\n"
        f"    if (delta > {literal(native_rtol)} * __builtin_fabsf({gold}[i]) + {literal(native_atol)}) ++errors;\n"
        f"  }}"
        for replica in range(batch_clusters)
        for out, gold, count in native_checks)
    native_int_check_source = "\n".join(
        f"  for (uint32_t i = 0; i < {count}; ++i) "
        f"if ({out}[i"
        + (f" + {replica * ((count + 15) // 16 * 16)}u" if batch_clusters == 2
           else "") + f"] != {gold}[i]) ++errors;"
        for replica in range(batch_clusters)
        for out, gold, count in integer_checks)
    native = ("#include <stdint.h>\n#include <stdio.h>\n"
              "#include \"pipeline_math.hpp\"\n"
              + ("#include \"pipeline_moe.hpp\"\n" if uses_moe else "")
              + ("#include \"pipeline_select_last.hpp\"\n"
                 if uses_select_last else "")
              + ("#include \"pipeline_olmo.hpp\"\n" if uses_olmo else "")
              + ("#include \"pipeline_nemotron.hpp\"\n" if uses_nemotron else "")
              + ("#include \"pipeline_int8.hpp\"\n" if uses_int8 else "")
              + ("#include \"pipeline_int4.hpp\"\n" if uses_int4 else "")
              + ("#include \"pipeline_packed.hpp\"\n" if uses_packed else "")
              + ("#include \"pipeline_tied.hpp\"\n" if uses_tied else "")
              + ("#include \"pipeline_fp32_tied.hpp\"\n" if uses_fp32_tied else "")
              + "\n".join(part.replace("__global ", "")
                          for part in declarations + ([] if device_check_all_stages else native_extra))
              + "\n" + "\n".join(stage_src) + "\nint main() {\n"
              + "\n".join(f"  stage_{i}((void*)(uintptr_t){replica}, 0, 1, 0);"
                          for replica in range(batch_clusters)
                          for i in range(len(graph.stages)))
              + "\n  uint32_t errors = 0; float maximum = 0.0f;\n"
              + native_check_source + "\n" + native_int_check_source +
              "\n  printf(\"errors=%u max_abs_error=%.9g\\n\", errors, maximum);\n"
              "  return errors ? 1 : 0;\n}\n")
    if not external:
        write_if_changed(target / "native.cpp", backend_source(native))
    write_if_changed(target / "host.cpp", (HERE / "host.cpp").read_text())
    if use_mx_vpu:
        write_if_changed(target / "mx_vpu_runtime.cpp", (HERE / "mx_vpu/runtime.cpp").read_text())
        device_source_files.append("mx_vpu_runtime.cpp")
    if fallback_call_abi:
        write_if_changed(target / "fallback_runtime.cpp", (HERE / "fallback_runtime.cpp").read_text())
        device_source_files.append("fallback_runtime.cpp")
    write_if_changed(target / "Makefile",
        "PROJECT = model_chain\nMU_SRCS = kernel.cpp\nHOST_SRCS = host.cpp\n"
        + ("MU_SRC_DEPS = " + " ".join(device_source_files) + "\n"
           if device_source_files else "")
        + (f"EXTRA_MU_CFLAGS += -{device_opt}\n" if device_opt != "O3" else "")
        + ("EXTRA_MU_CFLAGS += -ffp-contract=off\n" if use_mx_vpu else "")
        + f"RADIANCE_LIB_PATH := {ROOT / 'lib'}\n"
        "RADIANCE_INCLUDE_PATH := $(RADIANCE_LIB_PATH)/include\n"
        "GEMMINI_SW_PATH := $(RADIANCE_LIB_PATH)/mxgemmini\n"
        f"SOC_DIR := {ROOT / 'soc'}\n"
        f"LLVM_MUON ?= {ROOT / 'llvm/llvm-muon'}\n"
        f"EXTRA_MU_CFLAGS += -I{HERE}\n"
        f"include {ROOT / 'kernels/common.mk'}\n"
        f"kernel.mu.o: {HERE / 'pipeline_math.hpp'}"
        + (f" {HERE / 'pipeline_moe.hpp'}" if uses_moe else "")
        + (f" {HERE / 'pipeline_select_last.hpp'}" if uses_select_last else "")
        + (f" {HERE / 'pipeline_olmo.hpp'}" if uses_olmo else "")
        + (f" {HERE / 'pipeline_nemotron.hpp'}" if uses_nemotron else "")
        + (f" {HERE / 'pipeline_int8.hpp'}" if uses_int8 else "")
        + (f" {HERE / 'pipeline_int4.hpp'}" if uses_int4 else "")
        + (f" {HERE / 'pipeline_packed.hpp'}" if uses_packed else "")
        + (f" {HERE / 'pipeline_tied.hpp'}" if uses_tied else "")
        + (f" {HERE / 'pipeline_fp32_tied.hpp'}" if uses_fp32_tied else "")
        + (f" {HERE / 'mx_vpu/backend.hpp'}" if use_mx_vpu else "")
        + "\n"
        + (" ".join(filename.replace(".cpp", ".mu.o") for filename in stage_files)
           + f": {HERE / 'pipeline_math.hpp'}"
           + (f" {HERE / 'pipeline_moe.hpp'}" if uses_moe else "")
           + (f" {HERE / 'pipeline_select_last.hpp'}" if uses_select_last else "")
           + (f" {HERE / 'pipeline_olmo.hpp'}" if uses_olmo else "")
           + (f" {HERE / 'pipeline_nemotron.hpp'}" if uses_nemotron else "")
           + (f" {HERE / 'pipeline_int8.hpp'}" if uses_int8 else "")
           + (f" {HERE / 'pipeline_int4.hpp'}" if uses_int4 else "")
           + (f" {HERE / 'pipeline_packed.hpp'}" if uses_packed else "")
           + (f" {HERE / 'pipeline_tied.hpp'}" if uses_tied else "")
           + (f" {HERE / 'pipeline_fp32_tied.hpp'}" if uses_fp32_tied else "")
           + (f" {HERE / 'mx_vpu/backend.hpp'}" if use_mx_vpu else "")
           + "\n" if stage_files else ""))
    manifest = {
        "model": model,
        "source_config_sha256": specs[model]["source_sha256"],
        "tied_lm_head": bool(spec.get("tied_lm_head")),
        "scope": ("full_dimension_checkpoint_stage_probe" if stage_limit is not None
                  else "full_dimension_checkpoint_decoder") if external
                 else "reduced_synthetic_decoder",
        "device_optimization": device_opt,
        "stages_per_device_object": stages_per_object,
        "device_source_files": ["kernel.cpp", *device_source_files],
        "device_source_generated": True, "device_elf_built": False,
        "device_execution": False, "measured_cycles": None,
        "checkpoint_weights": external,
        "full_model_dimensions": external, "layers": layers,
        "stage_limit": stage_limit,
        "weight_image_manifest": str(weight_image.resolve()) if external else None,
        "weight_image_sha256": image_manifest["image_sha256"] if external else None,
        "weight_image_segments": len(image_manifest.get("segments", [None])) if external else None,
        "weight_segment_layout": [
            {key: item[key] for key in ("gpu_base_address", "image_size_bytes", "image_sha256")}
            for item in image_segments(image_manifest)
        ] if external else None,
        "weight_image_base_address": image_manifest["gpu_base_address"] if external else None,
        "checkpoint_sha256": image_manifest["checkpoint_weight_sha256"] if external else None,
        "reference_output_sha256": hashlib.sha256(
            values[graph.outputs[-1]].tobytes()).hexdigest() if external else None,
        "checkpoint_weight_format": image_manifest["dtype"] if external else None,
        "quantization_comparison": quantization_comparison,
        "upstream_execution_equivalent": False,
        "buffers_cache_line_padded": True,
        "prefill_tokens": prefill, "decode_steps": decode,
        "prefill_logits_policy": prefill_logits,
        "control_sliding_window": control_sliding_window,
        "input_token_ids": {name: tensor.ravel().tolist()
                            for name, tensor in sorted(inputs.items())},
        "token_sequence_explicit": token_ids is not None,
        "reference_generated_token_ids": (
            [int(values[f"decode{step}.token_ids"][0, 0])
             for step in range(decode)
             if f"decode{step}.token_ids" in values]
            if generation == "greedy" else None),
        "generation": generation, "stages": len(graph.stages),
        "device_check_all_stages": device_check_all_stages,
        "native_check_tolerance": {"rtol": native_rtol, "atol": native_atol}
        if not external else None,
        "batch_clusters": batch_clusters,
        "verified_tensors": (
            [stage["id"] for stage in graph.stages
             if graph.tensors[stage["writes"]]["dtype"] != "int32"]
            + [stage["id"] for stage in graph.stages
               if graph.tensors[stage["writes"]]["dtype"] == "int32"]
            if device_check_all_stages else
            [stage["id"] for stage in graph.stages
             if stage["op"] in ("causal_gqa", "gated_activation")]
            + graph.outputs),
        "native_verified_float_stages": len(native_checks),
        "native_verified_integer_stages": len(integer_checks),
        "source_spec_sha256": hashlib.sha256(stitch.SPECS.read_bytes()).hexdigest(),
    }
    if fallback_call_abi:
        manifest.update(fallback_call_abi=True,
                        fallback_call_source_sha256={name: sha256_file(HERE/name)
                            for name in ("fallback_abi.hpp", "fallback_runtime.cpp", "fallback_codegen.py")})
    if use_mx_vpu:
        manifest.update(execution_backend=execution_backend,
                        mx_linear_policy=mx_linear_policy,
                        mx_linear_macs=sum(int(np.prod(s["shape"])) * graph.tensors[s["reads"][0]]["shape"][-1]
                                           for s in graph.stages if mx_selected(s)),
                        total_linear_macs=sum(int(np.prod(s["shape"])) * graph.tensors[s["reads"][0]]["shape"][-1]
                                              for s in graph.stages if s["op"] == "linear"),
                        format_comparison=format_comparison,
                        backend_source_sha256={name: sha256_file(HERE / "mx_vpu" / name)
                                               for name in ("backend.hpp", "runtime.cpp", "reference.py", "fp32_reference.py")},
                        simulator_preset="radiance", simulator_overrides="",
                        hardware_compatible=False,
                        stage_backends=[dict(stage=s["id"],
                            backend="fp8_mx" if mx_selected(s) else
                                    "bf16_vpu" if s["op"] in ("add", "bias_add") else "muon",
                            reason=None if mx_selected(s) or s["op"] in ("add", "bias_add") else
                            "Preserve FP32 arithmetic under the selected projection/norm/attention fallback policy") for s in graph.stages])
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def verify_native(target: Path) -> str:
    manifest = json.loads((target / "manifest.json").read_text())
    verify_backend_sources(manifest)
    if manifest.get("execution_backend") == "rakanic_mx_vpu":
        archive = os.environ["RADIANCE_LIBGEMMINI_ARCHIVE"]
        root = os.environ["RADIANCE_LIBGEMMINI_ROOT"]
        command = ["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-no-pie",
                   "-I", str(HERE), "-I", root, "native.cpp", "mx_vpu_runtime.cpp",
                   archive, "-lm", "-o", "native"]
    else:
        command = ["clang++", "-O2", "-nostdlib++", "-I", str(HERE), "native.cpp", "-lm", "-o", "native"]
    if manifest.get("fallback_call_abi"):
        command[1:1] = ["-D__global=", "fallback_runtime.cpp"]
    subprocess.run(command, cwd=target, check=True)
    with (target / "native.log").open("w") as log:
        subprocess.run([str(target / "native")], cwd=target,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if manifest.get("execution_backend") == "rakanic_mx_vpu":
        text = (target / "native.log").read_text()
        counts = re.search(r"MX_VPU_MODEL commands=(\d+) mx=(\d+) vpu=(\d+) cycles=(\d+)", text)
        if counts is None or int(counts[2]) == 0 or int(counts[3]) == 0:
            raise ValueError("MX+VPU graph did not execute both command families")
        manifest["native_model_check"] = dict(
            format_reference="passed", original_greedy_exact=manifest["format_comparison"]["original_greedy_exact"],
            commands=int(counts[1]), mx_commands=int(counts[2]), vpu_commands=int(counts[3]), model_cycles=int(counts[4]),
            binary_sha256=sha256_file(target / "native"), log_sha256=sha256_file(target / "native.log"),
            libgemmini_archive_sha256=sha256_file(Path(os.environ["RADIANCE_LIBGEMMINI_ARCHIVE"])),
            libgemmini_commit=subprocess.check_output(["git", "-C", os.environ["RADIANCE_LIBGEMMINI_ROOT"],
                                                       "rev-parse", "HEAD"], text=True).strip(),
            full_checkpoint_execution=False, device_execution=False)
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return (target / "native.log").read_text().strip()


def build_device(target: Path) -> None:
    verify_backend_sources(json.loads((target / "manifest.json").read_text()))
    jobs = int(os.environ.get("RADIANCE_BUILD_JOBS", "2"))
    if not 1 <= jobs <= 16:
        raise ValueError("RADIANCE_BUILD_JOBS must be between 1 and 16")
    toolchain = Path(os.environ.get(
        "RISCV_TOOLCHAIN_PATH",
        "/scratch/agustin/projects/chipyard/.conda-env/riscv-tools"))
    if not toolchain.is_dir():
        raise FileNotFoundError("set RISCV_TOOLCHAIN_PATH to a RISC-V toolchain")
    host_toolchain = Path(os.environ.get("RISCV64_TOOLCHAIN_PATH", str(toolchain)))
    llvm_muon = Path(os.environ.get("LLVM_MUON", str(ROOT / "llvm/llvm-muon")))
    env = os.environ.copy()
    env.setdefault("RISCV_TOOLCHAIN_PATH", str(toolchain))
    env.setdefault("RISCV64_TOOLCHAIN_PATH", str(host_toolchain))
    env.setdefault("RISCV_SYSROOT", str(toolchain / "riscv64-unknown-elf"))
    env.setdefault("MU_LIBC_INCLUDE", str(toolchain / "riscv64-unknown-elf/include"))
    env.setdefault("LLVM_MUON", str(llvm_muon))
    extra = "-nostdinc++ -isystem " + str(llvm_muon / "include/c++/v1")
    env["EXTRA_MU_CFLAGS"] = (env.get("EXTRA_MU_CFLAGS", "") + " " + extra).strip()
    env.setdefault("MU_USE_LIBC", "1")
    command = ["make", f"-j{jobs}", "kernel.soc.elf"]
    with subprocess.Popen(command, cwd=target, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT) as process:
        tail = bytearray()
        total = 0
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.read(65536), b""):
            total += len(chunk)
            tail.extend(chunk)
            if len(tail) > 2_000_000:
                del tail[:len(tail) - 2_000_000]
        result = process.wait()
    (target / "build.log").write_text(
        f"make exit={result}; retained final {len(tail)} of {total} output bytes\n"
        + tail.decode(errors="replace"))
    if result:
        raise subprocess.CalledProcessError(result, command)
    subprocess.run([sys.executable, str(HERE / "check_layout.py"),
                    str(target / "kernel.radiance.elf"), "--generated", "--nm",
                    str(toolchain / "bin/riscv64-unknown-elf-nm")], check=True)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    images = [verify_image(Path(manifest[key]))[0]
              for key in ("weight_image_manifest", "input_image_manifest", "stage_golden_image_manifest")
              if manifest.get(key)]
    if manifest.get('replica_buffers'):
        from replica_buffers import reserved_image
        images.append(reserved_image(manifest['replica_buffers']))
    if images:
        verify_image_placement(target / "kernel.radiance.elf", images,
                               manifest.get('num_clusters', manifest.get('batch_clusters', 1)))
        if manifest.get("execution_backend") == "rakanic_mx_vpu":
            if any(s["gpu_base_address"] + s["image_size_bytes"] > 1 << 31
                   for image in images for s in image_segments(image)):
                raise ValueError("MX+VPU target images exceed the 2 GiB GPU region")
        manifest["image_placement_verified"] = True
    manifest["device_elf_built"] = True
    manifest["radiance_elf_sha256"] = hashlib.sha256(
        (target / "kernel.radiance.elf").read_bytes()).hexdigest()
    manifest["soc_elf_sha256"] = hashlib.sha256(
        (target / "kernel.soc.elf").read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--prefill", type=int, default=3)
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--generation", choices=("teacher_forced", "greedy"),
                        default="teacher_forced")
    parser.add_argument("--prefill-logits", choices=("all", "last"), default="all",
                        help="project only the final prefill token in greedy Qwen3")
    parser.add_argument("--control-sliding-window", type=int,
                        help="short OLMo 3 local window for a reduced mask control")
    parser.add_argument("--native-rtol", type=float, default=1e-4)
    parser.add_argument("--native-atol", type=float, default=1e-5)
    parser.add_argument("--execution-backend", choices=("muon", "rakanic_mx_vpu"), default="muon")
    parser.add_argument("--mx-linear-policy", choices=("all", "q_proj"), default="all")
    parser.add_argument("--fallback-call-abi", action="store_true")
    parser.add_argument("--out-root", type=Path, default=HERE / "generated")
    parser.add_argument("--device-check-all-stages", action="store_true")
    parser.add_argument("--device-opt", choices=("O1", "O2", "O3"), default="O3")
    parser.add_argument("--stages-per-object", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--weight-image", type=Path,
                        help="weights-image.json from export_decoder_weights.py")
    parser.add_argument("--stage-limit", type=int,
                        help="compile only the first N stages as a checkpoint probe")
    parser.add_argument("--batch-clusters", type=int, choices=(1, 2), default=1,
                        help="independent identical requests on one or two clusters")
    parser.add_argument("--token-ids", type=int, nargs="+",
                        help="prefill IDs; append decode IDs for teacher-forced mode")
    parser.add_argument("--verify-native", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.layers <= 0 or args.prefill <= 0 or args.decode_steps < 0:
        parser.error("layers/prefill must be positive and decode steps nonnegative")
    target = generate(args.model, args.layers, args.prefill,
                      args.decode_steps, args.generation, args.out_root,
                      args.device_check_all_stages, args.device_opt,
                      args.stages_per_object, args.checkpoint_dir,
                      args.weight_image, args.stage_limit, args.token_ids,
                      args.batch_clusters, args.prefill_logits,
                      args.control_sliding_window, args.native_rtol,
                      args.native_atol, args.execution_backend, args.mx_linear_policy, args.fallback_call_abi)
    if args.verify_native and args.weight_image:
        parser.error("native C++ checking is not supported for external weight images")
    if args.verify_native or (args.build and args.weight_image is None):
        print(verify_native(target))
    if args.build:
        build_device(target)
    print(target)


if __name__ == "__main__":
    main()
