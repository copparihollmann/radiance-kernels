#!/usr/bin/env python3
"""Compile a connected decoder graph into a Radiance SoC ELF.

The default uses reduced dimensions and deterministic weights to check stage
handoffs. A checkpoint build uses the pinned full dimensions and an external
FP32 weight image; it requires a separate Cyclotron run for device validation.
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
from checkpoint import SafeTensorWeights  # noqa: E402
from export_decoder_weights import WARP_STACK_BOTTOM, WARP_STACK_TOP  # noqa: E402
from split_decoder_weights import image_segments, verify_image  # noqa: E402


MODELS = ("tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma_2_2b_it")


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


def param_data(stage: dict) -> tuple[str, np.ndarray] | None:
    op, attrs = stage["op"], stage["attrs"]
    if op == "embedding":
        width, vocab = stage["shape"][-1], attrs["vocab_size"]
        data = np.stack([reference.parameter(f"{attrs['parameter']}.{i}", (width,))
                         for i in range(vocab)])
    elif op == "rmsnorm":
        width = stage["shape"][-1]
        data = reference.parameter(attrs["parameter"], (width,)) + 1.0
    elif op == "linear":
        shape = tuple(attrs["weight_shape"])
        data = reference.parameter(attrs["parameter"], shape,
                                   1.0 / np.sqrt(shape[0]))
    elif op == "bias_add":
        data = reference.parameter(attrs["parameter"], tuple(stage["shape"][2:]))
    else:
        return None
    return symbol("parameter." + attrs["parameter"]), data.astype(np.float32)


def emit_stage(index: int, stage: dict, graph: stitch.Graph,
               parameter: str | None) -> str:
    op, attrs, shape = stage["op"], stage["attrs"], stage["shape"]
    ins = [symbol(name) for name in stage["reads"]]
    out = symbol(stage["writes"])
    batch = shape[0]
    count = int(np.prod(shape))
    if op == "embedding":
        body = f"embedding({ins[0]}, {parameter}, {out}, {count // shape[-1]}, {shape[-1]}, {literal(attrs['scale'])}, tid, tpb)"
    elif op == "rmsnorm":
        body = f"rmsnorm({ins[0]}, {parameter}, {out}, {count // shape[-1]}, {shape[-1]}, {literal(attrs['epsilon'])}, tid, tpb)"
    elif op == "linear":
        k, n = attrs["weight_shape"]
        body = f"linear({ins[0]}, {parameter}, nullptr, {out}, {count // n}, {k}, {n}, tid, tpb)"
    elif op == "bias_add":
        width = int(np.prod(shape[2:]))
        body = f"bias_add({ins[0]}, {parameter}, {out}, {count}, {width}, tid, tpb)"
    elif op == "rope":
        tokens, heads, width = shape[1:]
        body = f"rope({ins[0]}, v_rope_cos, v_rope_sin, {out}, {batch}, {tokens}, {heads}, {width}, {attrs['position_start']}, tid, tpb)"
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
    else:
        raise ValueError(f"unsupported operation {op} at {stage['id']}")
    return (f"static void stage_{index}(void*, uint32_t tid, uint32_t tpb, uint32_t) {{\n"
            "#ifdef RADIANCE_DEVICE\n"
            "  // Each stage consumes the previous stage's GMEM output. Muon L0d is\n"
            "  // private per core, so keep this correctness build on core 0.\n"
            "  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return;\n"
            "  tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS\n"
            "      + tid % MU_NUM_THREADS;\n"
            "  tpb /= MU_NUM_CORES;\n"
            "#endif\n"
            f"  model_chain::{body};\n}}\n")


def generate(model: str, layers: int, prefill: int, decode: int,
             generation: str, out_root: Path,
             device_check_all_stages: bool = False,
             device_opt: str = "O3", stages_per_object: int = 0,
             checkpoint_dir: Path | None = None,
             weight_image: Path | None = None,
             stage_limit: int | None = None) -> Path:
    if device_opt not in ("O1", "O2", "O3"):
        raise ValueError(f"unsupported device optimization level: {device_opt}")
    if stages_per_object < 0:
        raise ValueError("stages per object cannot be negative")
    external = checkpoint_dir is not None or weight_image is not None
    if external and (checkpoint_dir is None or weight_image is None):
        raise ValueError("checkpoint directory and weight image are both required")
    if stage_limit is not None and not external:
        raise ValueError("stage limit is only for checkpoint device probes")
    if external and (model not in ("tinyllama", "deepseek_r1_distill_qwen_1_5b")
                     or stages_per_object <= 0 or not device_check_all_stages
                     or generation != "teacher_forced"):
        raise ValueError("checkpoint builds require a pinned model, sharded objects, "
                         "all-stage checks, and teacher-forced tokens")
    specs = stitch.model_specs()
    spec = dict(specs[model]) if external else reference.reduced_spec(specs[model])
    if not 1 <= layers <= specs[model]["num_hidden_layers"]:
        raise ValueError("layer count exceeds the pinned model")
    spec["num_hidden_layers"] = layers
    graph = stitch.build(model, prefill=prefill, decode_steps=decode,
                         specs={model: spec}, generation=generation)
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
                or image_manifest["dtype"] != "fp32"
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
        if sha256_file(checkpoint_dir / "model.safetensors") != image_manifest[
                "checkpoint_weight_sha256"]:
            raise ValueError("checkpoint weights differ from image source")
        addresses = {item["logical_name"]: item["gpu_address"]
                     for item in image_manifest["parameters"]}
        provider = SafeTensorWeights(checkpoint_dir / "model.safetensors",
                                     spec["family"])
    inputs = {name: np.arange(1, shape["shape"][1] + 1, dtype=np.int32)[None]
              for name, shape in graph.tensors.items()
              if name.endswith(".token_ids") and name.startswith(("prefill", "decode"))
              and name not in {stage["writes"] for stage in graph.stages}}
    values = reference.execute(graph, inputs, weights=provider)
    target = (out_root / model).resolve()
    target.mkdir(parents=True, exist_ok=True)
    declarations = []
    for name, tensor in graph.tensors.items():
        data = inputs.get(name)
        dtype = "int32_t" if tensor["dtype"] == "int32" else "float"
        declarations.append(array_decl(symbol(name), data,
                                       int(np.prod(tensor["shape"])), dtype))
    if not external:
        params = {}
        for stage in graph.stages:
            item = param_data(stage)
            if item is not None:
                name, data = item
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
    half = spec["head_dim"] // 2
    max_position = prefill + decode
    inv_freq = np.power(10000.0, -np.arange(half, dtype=np.float32) / half)
    angles = np.arange(max_position, dtype=np.float32)[:, None] * inv_freq
    declarations.append(array_decl("v_rope_cos", np.cos(angles), max_position * half))
    declarations.append(array_decl("v_rope_sin", np.sin(angles), max_position * half))
    stage_src = []
    for i, stage in enumerate(graph.stages):
        stage_src.append(emit_stage(i, stage, graph, parameter_symbol(stage)))
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
                        external_params[param] = addresses[stage["attrs"]["parameter"]]
                    else:
                        symbols[param] = "float"
                if stage["op"] == "rope":
                    symbols["v_rope_cos"] = "float"
                    symbols["v_rope_sin"] = "float"
            externs = "\n".join(f"extern __global {dtype} {name}[];"
                                for name, dtype in sorted(symbols.items()))
            pointers = "\n".join(
                f"#define {name} ((const __global float*)0x{address:08x}u)"
                for name, address in sorted(external_params.items()))
            source = ("#include <mu_intrinsics.h>\n#include <mu_schedule.h>\n"
                      "#include <stdint.h>\n#include \"pipeline_math.hpp\"\n"
                      + externs + "\n" + pointers + "\n"
                      + "\n".join(item.replace("static void stage_", "void stage_", 1)
                                  for item in stage_src[first:last]))
            write_if_changed(target / filename, source)
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
    schedule = "\n".join(f"  mu_schedule(stage_{i}, nullptr, 1);\n"
                         "  mu_barrier(0, MU_NUM_CORES);\n  mu_fence();"
                         for i in range(len(graph.stages)))
    if device_check_all_stages:
        float_verify = "\n".join(
            f"  errors = 0; uint32_t first_{index} = {count};\n"
            f"  for (uint32_t i = 0; i < {count}; ++i) "
            f"if (!mu_close({out}[i], {gold}[i], 5e-3f, 5e-4f)) "
            f"{{ if (!errors) first_{index} = i; ++errors; }}\n"
            f"  if (errors) mu_tohost(({index + 1}u << 16) | (first_{index} << 1) | 1u);"
            for index, (out, gold, count) in enumerate(native_checks))
        int_verify = "\n".join(
            f"  errors = 0; uint32_t first_int_{index} = {count};\n"
            f"  for (uint32_t i = 0; i < {count}; ++i) "
            f"if ({out}[i] != {gold}[i]) "
            f"{{ if (!errors) first_int_{index} = i; ++errors; }}\n"
            f"  if (errors) mu_tohost(({len(native_checks) + index + 1}u << 16) "
            f"| (first_int_{index} << 1) | 1u);"
            for index, (out, gold, count) in enumerate(integer_checks))
        verify = float_verify + ("\n" + int_verify if int_verify else "")
    else:
        verify = "\n".join(
            f"  for (uint32_t i = 0; i < {count}; ++i) "
            f"if (!mu_close({out}[i], {gold}[i], 5e-3f, 5e-4f)) ++errors;"
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
                comparison = (f"!mu_close({out}[i], {gold}[i], 5e-3f, 5e-4f)"
                              if dtype == "float" else f"{out}[i] != {gold}[i]")
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
                      + f"\nbool verify_chunk_{first // stages_per_object:03d}() {{\n"
                      "  uint32_t errors = 0;\n" + "\n".join(checks)
                      + "\n  return true;\n}\n")
            write_if_changed(target / filename, source)
        device_source_files.extend(verifier_files)
        device_declarations = ""
        verify = "\n".join(
            f"  if (!verify_chunk_{index:03d}()) return 0;"
            for index in range(len(verifier_files)))
        verify_prototypes = "\n".join(
            f"bool verify_chunk_{index:03d}();"
            for index in range(len(verifier_files)))
    else:
        verify_prototypes = ""
    device_stages = ("\n".join(f"void stage_{i}(void*, uint32_t, uint32_t, uint32_t);"
                                for i in range(len(stage_src))) if stage_files
                     else "\n".join(stage_src))
    cpp = ("#include <mu_intrinsics.h>\n#include <mu_schedule.h>\n"
           "#include <stdint.h>\n#include \"kernel_verify.h\"\n"
           "#include \"pipeline_math.hpp\"\n"
           "extern \"C\" uint32_t __mu_num_warps = 1;\n"
           + device_declarations + "\n" + device_stages + "\n" + verify_prototypes +
           "\nint main() {\n" + schedule +
           "\n  if (mu_hart_id() != 0) { for (;;) {} }\n"
           "  uint32_t errors = 0;\n" + verify +
           "\n  mu_tohost(errors ? ((errors << 1) | 1u) : 0u);\n  return 0;\n}\n")
    write_if_changed(target / "kernel.cpp", cpp)
    native_check_source = "\n".join(
        f"  for (uint32_t i = 0; i < {count}; ++i) {{\n"
        f"    const float delta = __builtin_fabsf({out}[i] - {gold}[i]);\n"
        f"    if (delta > maximum) maximum = delta;\n"
        f"    if (delta > 1e-4f * __builtin_fabsf({gold}[i]) + 1e-5f) ++errors;\n"
        f"  }}"
        for out, gold, count in native_checks)
    native_int_check_source = "\n".join(
        f"  for (uint32_t i = 0; i < {count}; ++i) "
        f"if ({out}[i] != {gold}[i]) ++errors;"
        for out, gold, count in integer_checks)
    native = ("#include <stdint.h>\n#include <stdio.h>\n"
              "#include \"pipeline_math.hpp\"\n"
              + "\n".join(part.replace("__global ", "")
                          for part in declarations + ([] if device_check_all_stages else native_extra))
              + "\n" + "\n".join(stage_src) + "\nint main() {\n"
              + "\n".join(f"  stage_{i}(nullptr, 0, 1, 0);"
                          for i in range(len(graph.stages)))
              + "\n  uint32_t errors = 0; float maximum = 0.0f;\n"
              + native_check_source + "\n" + native_int_check_source +
              "\n  printf(\"errors=%u max_abs_error=%.9g\\n\", errors, maximum);\n"
              "  return errors ? 1 : 0;\n}\n")
    if not external:
        write_if_changed(target / "native.cpp", native)
    write_if_changed(target / "host.cpp", (HERE / "host.cpp").read_text())
    write_if_changed(target / "Makefile",
        "PROJECT = model_chain\nMU_SRCS = kernel.cpp\nHOST_SRCS = host.cpp\n"
        + ("MU_SRC_DEPS = " + " ".join(device_source_files) + "\n"
           if device_source_files else "")
        + (f"EXTRA_MU_CFLAGS += -{device_opt}\n" if device_opt != "O3" else "")
        + f"RADIANCE_LIB_PATH := {ROOT / 'lib'}\n"
        "RADIANCE_INCLUDE_PATH := $(RADIANCE_LIB_PATH)/include\n"
        "GEMMINI_SW_PATH := $(RADIANCE_LIB_PATH)/mxgemmini\n"
        f"SOC_DIR := {ROOT / 'soc'}\n"
        f"LLVM_MUON ?= {ROOT / 'llvm/llvm-muon'}\n"
        f"EXTRA_MU_CFLAGS += -I{HERE}\n"
        f"include {ROOT / 'kernels/common.mk'}\n"
        f"kernel.mu.o: {HERE / 'pipeline_math.hpp'}\n"
        + (" ".join(filename.replace(".cpp", ".mu.o") for filename in stage_files)
           + f": {HERE / 'pipeline_math.hpp'}\n" if stage_files else ""))
    manifest = {
        "model": model,
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
        "upstream_execution_equivalent": False,
        "buffers_cache_line_padded": True,
        "prefill_tokens": prefill, "decode_steps": decode,
        "generation": generation, "stages": len(graph.stages),
        "device_check_all_stages": device_check_all_stages,
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
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def verify_native(target: Path) -> str:
    subprocess.run(["clang++", "-O2", "-nostdlib++", "-I", str(HERE), "native.cpp",
                    "-lm", "-o", "native"], cwd=target, check=True)
    with (target / "native.log").open("w") as log:
        subprocess.run([str(target / "native")], cwd=target,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return (target / "native.log").read_text().strip()


def build_device(target: Path) -> None:
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
    command = ["make", "-j2", "kernel.soc.elf"]
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
    parser.add_argument("--out-root", type=Path, default=HERE / "generated")
    parser.add_argument("--device-check-all-stages", action="store_true")
    parser.add_argument("--device-opt", choices=("O1", "O2", "O3"), default="O3")
    parser.add_argument("--stages-per-object", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--weight-image", type=Path,
                        help="weights-image.json from export_decoder_weights.py")
    parser.add_argument("--stage-limit", type=int,
                        help="compile only the first N stages as a checkpoint probe")
    parser.add_argument("--verify-native", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.layers <= 0 or args.prefill <= 0 or args.decode_steps < 0:
        parser.error("layers/prefill must be positive and decode steps nonnegative")
    target = generate(args.model, args.layers, args.prefill,
                      args.decode_steps, args.generation, args.out_root,
                      args.device_check_all_stages, args.device_opt,
                      args.stages_per_object, args.checkpoint_dir,
                      args.weight_image, args.stage_limit)
    if args.verify_native and args.weight_image:
        parser.error("native C++ checking is not supported for external weight images")
    if args.verify_native or (args.build and args.weight_image is None):
        print(verify_native(target))
    if args.build:
        build_device(target)
    print(target)


if __name__ == "__main__":
    main()
