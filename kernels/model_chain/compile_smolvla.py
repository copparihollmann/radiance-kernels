#!/usr/bin/env python3
"""Lower the pinned SmolVLA action-chunk schedule to one Radiance ELF.

The graph keeps its three vision branches, cached VLM prefix, ten action
denoising iterations, and action carry. FP32 math is used for a first device
mapping; the output must still be compared with the upstream policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

from compile_decoder import build_device, literal, symbol, write_if_changed
from plan_buffers import plan as plan_buffers
from split_decoder_weights import verify_image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernels/evaluation/llm"))
from stitch import build, model_specs  # noqa: E402


def product(values) -> int:
    return math.prod(values)


def emit_stage(stage: dict, graph, allocation: dict, addresses: dict) -> str:
    op, attrs, shape = stage["op"], stage["attrs"], stage["shape"]
    reads = stage["reads"]

    def buffer(name: str, dtype: str = "float") -> str:
        return f"((__global {dtype}*)(v_arena + {allocation[name]['offset_bytes']}u))"

    def param(name: str) -> str:
        if name not in addresses:
            raise KeyError(f"{stage['id']}: weight image lacks {name}")
        return f"((const __global float*)0x{addresses[name]:08x}u)"

    def f(index: int) -> str:
        return buffer(reads[index])

    def i(index: int) -> str:
        return buffer(reads[index], "int32_t")

    def b(index: int) -> str:
        return buffer(reads[index], "uint32_t")

    out = buffer(stage["writes"])
    count = product(shape)
    body = None
    if op == "patch_embed":
        names = attrs["parameters"]
        weight = next(name for name in names if name.endswith("patch_embedding.weight"))
        bias = next(name for name in names if name.endswith("patch_embedding.bias"))
        position = next(name for name in names if name.endswith("position_embedding.weight"))
        body = (f"patch_embed({f(0)}, {param(weight)}, {param(bias)}, "
                f"{param(position)}, {out}, 512, 16, 3, {shape[-1]}, tid, tpb)")
    elif op == "layernorm":
        gamma = next(name for name in attrs["parameters"] if name.endswith(".weight"))
        bias = next(name for name in attrs["parameters"] if name.endswith(".bias"))
        body = (f"layernorm({f(0)}, {param(gamma)}, {param(bias)}, {out}, "
                f"{count // shape[-1]}, {shape[-1]}, {literal(attrs['epsilon'])}, tid, tpb)")
    elif op == "linear":
        k, n = attrs["weight_shape"]
        body = (f"linear({f(0)}, {param(attrs['parameter'])}, nullptr, {out}, "
                f"{count // n}, {k}, {n}, tid, tpb)")
    elif op == "bias_add":
        width = product(shape[2:]) if len(shape) > 2 else shape[-1]
        body = (f"bias_add({f(0)}, {param(attrs['parameter'])}, {out}, "
                f"{count}, {width}, tid, tpb)")
    elif op == "bidirectional_mha":
        body = (f"bidirectional_mha({f(0)}, {f(1)}, {f(2)}, {out}, "
                f"v_attention_scratch, {shape[1]}, {attrs['heads']}, "
                f"{attrs['head_dim']}, tid, tpb)")
    elif op == "add":
        body = f"residual({f(0)}, {f(1)}, {out}, {count}, tid, tpb)"
    elif op == "gelu_tanh":
        body = f"gelu_tanh_values({f(0)}, {out}, {count}, tid, tpb)"
    elif op == "pixel_shuffle":
        source = graph.tensors[reads[0]]["shape"]
        axis = math.isqrt(source[1])
        if axis * axis != source[1]:
            raise ValueError("vision patch grid is not square")
        body = (f"pixel_shuffle({f(0)}, {out}, {axis}, {source[-1]}, "
                f"{attrs['scale_factor']}, tid, tpb)")
    elif op == "embedding_scale":
        body = (f"scale_values({f(0)}, {out}, {count}, "
                f"{literal(attrs['scale'])}, tid, tpb)")
    elif op == "embedding":
        body = (f"embedding({i(0)}, {param(attrs['parameter'])}, {out}, "
                f"{count // shape[-1]}, {shape[-1]}, 1.0f, tid, tpb)")
    elif op == "multimodal_merge":
        parts = reads[:5]
        if sum(product(graph.tensors[name]["shape"]) for name in parts) != count:
            raise ValueError("multimodal merge dimensions disagree")
        offset = 0
        calls = []
        for name in parts:
            size = product(graph.tensors[name]["shape"])
            calls.append(f"model_chain::copy_values({buffer(name)}, {out} + {offset}, "
                         f"{size}, tid, tpb);")
            offset += size
        body = "\n  ".join(calls)
    elif op == "prefix_pad_mask":
        camera_tokens = attrs["image_tokens_per_camera"]
        language_tokens = attrs["language_tokens"]
        output = buffer(stage["writes"], "uint32_t")
        body = (f"for (uint32_t j = tid; j < {count}; j += tpb) {{\n"
                f"    {output}[j] = j < {camera_tokens} ? {b(0)}[0] : "
                f"j < {2 * camera_tokens} ? {b(1)}[0] : "
                f"j < {3 * camera_tokens} ? {b(2)}[0] : "
                f"j < {3 * camera_tokens + language_tokens} ? "
                f"{b(3)}[j - {3 * camera_tokens}] : 1u;\n  }}")
    elif op == "prefix_attention_groups":
        output = buffer(stage["writes"], "uint32_t")
        body = (f"for (uint32_t j = tid; j < {count}; j += tpb) "
                f"{output}[j] = j == {count - 1} ? 1u : 0u")
    elif op == "prefix_attention_mask":
        tokens = shape[1]
        output = buffer(stage["writes"], "uint32_t")
        body = (f"for (uint32_t j = tid; j < {count}; j += tpb) {{\n"
                f"    uint32_t q = j / {tokens}, k = j % {tokens};\n"
                f"    {output}[j] = {b(0)}[q] && {b(0)}[k] && "
                f"({b(1)}[k] <= {b(1)}[q]);\n  }}")
    elif op == "prefix_position_ids":
        output = buffer(stage["writes"], "int32_t")
        body = (f"for (uint32_t j = tid; j < {count}; j += tpb) {{\n"
                f"    int32_t valid = -1; for (uint32_t k = 0; k <= j; ++k) "
                f"valid += {b(0)}[k] ? 1 : 0;\n"
                f"    {output}[j] = valid;\n  }}")
    elif op == "rmsnorm":
        body = (f"rmsnorm({f(0)}, {param(attrs['parameter'])}, {out}, "
                f"{count // shape[-1]}, {shape[-1]}, "
                f"{literal(attrs['epsilon'])}, tid, tpb)")
    elif op == "rope_with_positions":
        body = (f"rope_positions({f(0)}, {i(1)}, v_rope_cos, v_rope_sin, "
                f"{out}, {shape[1]}, {shape[2]}, {shape[3]}, tid, tpb)")
    elif op == "masked_gqa":
        keys = graph.tensors[reads[1]]["shape"][1]
        body = (f"masked_gqa({f(0)}, {f(1)}, {f(2)}, {b(3)}, {out}, "
                f"v_attention_scratch, {shape[1]}, {keys}, "
                f"{attrs['q_heads']}, {attrs['kv_heads']}, "
                f"{attrs['head_dim']}, tid, tpb)")
    elif op == "gated_activation":
        gelu = "true" if attrs["function"] == "gelu_tanh" else "false"
        body = f"gated_activation({f(0)}, {f(1)}, {out}, {count}, {gelu}, tid, tpb)"
    elif op == "timestep_constant":
        body = f"if (tid == 0) {out}[0] = {literal(attrs['value'])}"
    elif op == "sinusoidal_time_embedding":
        step = int(stage["id"].split(".")[0].removeprefix("denoise"))
        body = (f"copy_values(v_time_embedding + {step * count}, {out}, "
                f"{count}, tid, tpb)")
    elif op == "broadcast_tokens":
        body = f"repeat_rows({f(0)}, {out}, {shape[1]}, {shape[-1]}, tid, tpb)"
    elif op == "concat_features":
        first = graph.tensors[reads[0]]["shape"][-1]
        second = graph.tensors[reads[1]]["shape"][-1]
        body = (f"concat_features({f(0)}, {f(1)}, {out}, {count // shape[-1]}, "
                f"{first}, {second}, tid, tpb)")
    elif op == "silu":
        body = f"silu_values({f(0)}, {out}, {count}, tid, tpb)"
    elif op == "suffix_pad_mask" or op == "suffix_attention_groups":
        output = buffer(stage["writes"], "uint32_t")
        body = (f"for (uint32_t j = tid; j < {count}; j += tpb) "
                f"{output}[j] = 1u")
    elif op == "suffix_attention_mask":
        prefix = graph.tensors[reads[0]]["shape"][1]
        suffix = shape[1]
        output = buffer(stage["writes"], "uint32_t")
        body = (f"for (uint32_t j = tid; j < {count}; j += tpb) {{\n"
                f"    uint32_t q = j / {prefix + suffix}, k = j % {prefix + suffix};\n"
                f"    {output}[j] = {b(1)}[q] && (k < {prefix} ? {b(0)}[k] : "
                f"({b(1)}[k - {prefix}] && k - {prefix} <= q));\n  }}")
    elif op == "suffix_position_ids":
        prefix = graph.tensors[reads[0]]["shape"][1]
        output = buffer(stage["writes"], "int32_t")
        body = (f"for (uint32_t j = tid; j < {count}; j += tpb) {{\n"
                f"    int32_t valid = -1; for (uint32_t k = 0; k < {prefix}; ++k) "
                f"valid += {b(0)}[k] ? 1 : 0;\n"
                f"    for (uint32_t k = 0; k <= j; ++k) valid += {b(1)}[k] ? 1 : 0;\n"
                f"    {output}[j] = valid;\n  }}")
    elif op == "suffix_local_position_ids":
        output = buffer(stage["writes"], "int32_t")
        body = (f"for (uint32_t j = tid; j < {count}; j += tpb) "
                f"{output}[j] = (int32_t)j")
    elif op == "suffix_kv_append":
        first_count = product(graph.tensors[reads[0]]["shape"])
        second_count = product(graph.tensors[reads[1]]["shape"])
        body = (f"append_tokens({f(0)}, {f(1)}, {out}, "
                f"{first_count}, {second_count}, tid, tpb)")
    elif op == "prefix_attention_slice":
        source_width = graph.tensors[reads[0]]["shape"][-1]
        target_width = shape[-1]
        output = buffer(stage["writes"], "uint32_t")
        body = (f"for (uint32_t j = tid; j < {count}; j += tpb) "
                f"{output}[j] = {b(0)}[(j / {target_width}) * {source_width} + "
                f"j % {target_width}]")
    elif op == "euler_step":
        body = (f"euler_step({f(0)}, {f(1)}, {out}, {count}, "
                f"{literal(attrs['step_size'])}, tid, tpb)")
    else:
        raise ValueError(f"{stage['id']}: no Radiance lowering for {op}")
    if not body.endswith(";") and not body.endswith("}"):
        body += ";"
    control_flow = {
        "prefix_pad_mask", "prefix_attention_groups", "prefix_attention_mask",
        "prefix_position_ids", "suffix_pad_mask", "suffix_attention_groups",
        "suffix_attention_mask", "suffix_position_ids",
        "suffix_local_position_ids", "prefix_attention_slice",
        "timestep_constant", "multimodal_merge",
    }
    code = body if op in control_flow else "model_chain::" + body
    return (
        f"void stage_{symbol(stage['id'])}(void*, uint32_t tid, uint32_t tpb, uint32_t) {{\n"
        "  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return;\n"
        "  tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS\n"
        "      + tid % MU_NUM_THREADS;\n"
        "  tpb /= MU_NUM_CORES;\n"
        f"  {code}\n"
        "}\n")


def generate(weight_image: Path, out_root: Path, stage_limit: int | None = None,
             stages_per_object: int = 50,
             input_image: Path | None = None,
             golden_output: Path | None = None) -> Path:
    if stages_per_object <= 0:
        raise ValueError("stages per object must be positive")
    graph = build("smolvla_base")
    storage = plan_buffers(graph)
    image, _ = verify_image(weight_image)
    spec = model_specs()["smolvla_base"]
    if (image["model"] != "smolvla_base" or image["dtype"] != "fp32" or
            image["checkpoint_weight_sha256"] != spec["checkpoint_weight_sha256"] or
            image["execution_schedule_sha256"] != storage[
                "execution_schedule_sha256"]):
        raise ValueError("SmolVLA image differs from pinned graph or checkpoint")
    addresses = {item["logical_name"]: item["gpu_address"]
                 for item in image["parameters"]}
    if len(addresses) != image["distinct_parameters"]:
        raise ValueError("SmolVLA image has duplicate checkpoint parameters")
    input_manifest = None
    input_addresses = {}
    if input_image is not None:
        input_manifest, _ = verify_image(input_image)
        if (input_manifest["model"] != "smolvla_base" or
                input_manifest["checkpoint_weight_sha256"] !=
                image["checkpoint_weight_sha256"]):
            raise ValueError("SmolVLA input image differs from pinned checkpoint")
        input_addresses = {item["logical_name"]: item["gpu_address"]
                           for item in input_manifest["parameters"]}
        required_inputs = set(graph.tensors) - {stage["writes"] for stage in graph.stages}
        if set(input_addresses) != required_inputs:
            raise ValueError("SmolVLA input image does not cover graph inputs")
    golden = None
    golden_values = None
    if golden_output is not None:
        if input_manifest is None:
            raise ValueError("upstream golden comparison requires exact input image")
        golden = json.loads(golden_output.read_text())
        golden_values = np.asarray(golden["output_values"], dtype=np.float32)
        if (not golden["passed"] or golden["model"] != "smolvla_base" or
                golden["checkpoint_weight_sha256"] != image[
                    "checkpoint_weight_sha256"] or
                input_manifest["source_input_sha256"] !=
                golden["input_sha256"] or
                hashlib.sha256(golden_values.tobytes()).hexdigest() !=
                golden["output_sha256"] or
                input_manifest["source_golden_output_sha256"] !=
                golden["output_sha256"] or
                tuple(golden_values.shape) != tuple(graph.tensors[graph.outputs[-1]]["shape"])):
            raise ValueError("upstream golden output differs from input or graph")
    stages = graph.stages if stage_limit is None else graph.stages[:stage_limit]
    if not stages:
        raise ValueError("at least one graph stage is required")
    target = (out_root / "smolvla_base").resolve()
    target.mkdir(parents=True, exist_ok=True)
    allocation = storage["allocation"]
    half = spec["vlm_head_dim"] // 2
    positions = 1 + graph.tensors["denoise0.attention_mask"]["shape"][-1]
    inv = np.power(10000.0, -np.arange(half, dtype=np.float32) / half)
    angles = np.arange(positions, dtype=np.float32)[:, None] * inv
    steps = spec["num_denoise_steps"]
    dimension = spec["expert_hidden_size"]
    fraction = np.linspace(0.0, 1.0, dimension // 2, dtype=np.float64)
    periods = spec["min_period"] * (spec["max_period"] / spec["min_period"]) ** fraction
    table = []
    for step in range(steps):
        time = 1.0 - step / steps
        phase = (2.0 * math.pi / periods) * time
        table.extend(np.concatenate((np.sin(phase), np.cos(phase))).astype(np.float32))

    def declaration(name: str, values) -> str:
        return (f"alignas(64) __global float {name}[{len(values)}] = {{" +
                ", ".join(literal(x) for x in values) + "};\n")

    data = ("#include <mu_intrinsics.h>\n#include <stdint.h>\n"
            f"alignas(64) __global unsigned char v_arena[{storage['arena_bytes']}] = {{0}};\n"
            "alignas(64) __global float v_attention_scratch[MU_NUM_THREADS * 1024] = {0};\n"
            + declaration("v_rope_cos", np.cos(angles).ravel())
            + declaration("v_rope_sin", np.sin(angles).ravel())
            + declaration("v_time_embedding", table)
            + (declaration("v_golden_output", golden_values.ravel())
               if golden_values is not None else ""))
    write_if_changed(target / "model_data.cpp", data)
    chunks = []
    for first in range(0, len(stages), stages_per_object):
        filename = f"stage_chunk_{first // stages_per_object:03d}.cpp"
        chunks.append(filename)
        source = ("#include <mu_intrinsics.h>\n#include <mu_schedule.h>\n"
                  "#include <stdint.h>\n#include \"pipeline_smolvla.hpp\"\n"
                  "extern __global unsigned char v_arena[];\n"
                  "extern __global float v_attention_scratch[];\n"
                  "extern __global float v_rope_cos[];\n"
                  "extern __global float v_rope_sin[];\n"
                  "extern __global float v_time_embedding[];\n"
                  + "\n".join(emit_stage(stage, graph, allocation, addresses)
                              for stage in stages[first:first + stages_per_object]))
        write_if_changed(target / filename, source)
    inputs = []
    if input_manifest is not None:
        for name, tensor in graph.tensors.items():
            if name not in input_addresses:
                continue
            offset = allocation[name]["offset_bytes"]
            address = input_addresses[name]
            count = product(tensor["shape"])
            inputs.append(
                f"  {{ __global uint32_t* target = (__global uint32_t*)(v_arena + {offset}u); "
                f"const __global uint32_t* source = "
                f"(const __global uint32_t*)0x{address:08x}u; "
                f"for (uint32_t j = tid; j < {count}u; j += tpb) "
                "target[j] = source[j]; }")
    else:
        for camera in range(3):
            offset = allocation[f"camera{camera}.image"]["offset_bytes"]
            inputs.append(
                f"  {{ __global float* image = (__global float*)(v_arena + {offset}u); "
                f"for (uint32_t j = tid; j < {3 * 512 * 512}u; j += tpb) {{ "
                f"uint32_t c = j / {512 * 512}u, r = (j / 512u) % 512u, "
                f"col = j % 512u; uint32_t shifted = "
                f"(col + 512u - {camera * 17}u) % 512u; "
                f"uint32_t source = c * {512 * 512}u + r * 512u + shifted; "
                f"image[j] = -1.0f + 2.0f * (float)source / "
                f"(float){3 * 512 * 512 - 1}u; }} }}")
            valid = allocation[f"camera{camera}.valid"]["offset_bytes"]
            inputs.append(f"  if (tid == 0) *(__global uint32_t*)(v_arena + {valid}u) = 1u;")
        for name, count, expression, dtype in (
            ("language.ids", 48, "(int32_t)(j + 1u)", "int32_t"),
            ("language.valid", 48, "1u", "uint32_t"),
            ("robot.state", 32, "-0.5f + (float)j / 31.0f", "float"),
            ("action.noise", 1600, "-0.25f + 0.5f * (float)j / 1599.0f", "float")):
            offset = allocation[name]["offset_bytes"]
            inputs.append(f"  {{ __global {dtype}* data = (__global {dtype}*)(v_arena + {offset}u); "
                          f"for (uint32_t j = tid; j < {count}u; j += tpb) "
                          f"data[j] = {expression}; }}")
    init = ("void init_inputs(void*, uint32_t tid, uint32_t tpb, uint32_t) {\n"
            "  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return;\n"
            "  tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS\n"
            "      + tid % MU_NUM_THREADS; tpb /= MU_NUM_CORES;\n"
            + "\n".join(inputs) + "\n}\n")
    final = stages[-1]["writes"]
    final_count = product(graph.tensors[final]["shape"])
    final_offset = allocation[final]["offset_bytes"]
    schedule = "  mu_schedule(init_inputs, nullptr, 1);\n  mu_barrier(0, MU_NUM_CORES);\n  mu_fence();\n"
    for stage in stages:
        schedule += (f"  mu_schedule(stage_{symbol(stage['id'])}, nullptr, 1);\n"
                     "  mu_barrier(0, MU_NUM_CORES);\n  mu_fence();\n")
    prototypes = "\n".join(
        f"void stage_{symbol(stage['id'])}(void*, uint32_t, uint32_t, uint32_t);"
        for stage in stages)
    kernel = ("#include <mu_intrinsics.h>\n#include <mu_schedule.h>\n"
              "#include <stdint.h>\n#include \"kernel_verify.h\"\n"
              "extern \"C\" uint32_t __mu_num_warps = 1;\n"
              "extern __global unsigned char v_arena[];\n"
              + ("extern __global float v_golden_output[];\n"
                 if golden_values is not None else "")
              + prototypes + "\n" + init +
              "\nint main() {\n" + schedule +
              "  if (mu_hart_id() != 0) { for (;;) {} }\n"
              f"  const __global float* output = (const __global float*)(v_arena + {final_offset}u);\n"
              f"  for (uint32_t j = 0; j < {final_count}u; ++j) {{\n"
              "    if (!(output[j] == output[j]) || "
              "output[j] > 1.0e20f || output[j] < -1.0e20f) { "
              "mu_tohost((j << 1) | 1u); return 0; }\n"
              + ("    if (!mu_close(output[j], v_golden_output[j], 1e-2f, 1e-2f)) "
                 "{ mu_tohost((j << 1) | 1u); return 0; }\n"
                 if golden_values is not None else "")
              + "  }\n"
              "  mu_tohost(0u); return 0;\n}\n")
    write_if_changed(target / "kernel.cpp", kernel)
    shutil.copyfile(HERE / "host.cpp", target / "host.cpp")
    dependencies = ["model_data.cpp", *chunks]
    makefile = ("PROJECT = model_chain\nMU_SRCS = kernel.cpp\nHOST_SRCS = host.cpp\n"
                "MU_SRC_DEPS = " + " ".join(dependencies) + "\n"
                f"RADIANCE_LIB_PATH := {ROOT / 'lib'}\n"
                "RADIANCE_INCLUDE_PATH := $(RADIANCE_LIB_PATH)/include\n"
                "GEMMINI_SW_PATH := $(RADIANCE_LIB_PATH)/mxgemmini\n"
                f"SOC_DIR := {ROOT / 'soc'}\n"
                f"LLVM_MUON ?= {ROOT / 'llvm/llvm-muon'}\n"
                f"EXTRA_MU_CFLAGS += -I{HERE}\n"
                f"include {ROOT / 'kernels/common.mk'}\n"
                f"kernel.mu.o: {HERE / 'pipeline_smolvla.hpp'}\n"
                + " ".join(name.replace(".cpp", ".mu.o") for name in chunks)
                + f": {HERE / 'pipeline_smolvla.hpp'}\n")
    write_if_changed(target / "Makefile", makefile)
    result = {
        "model": "smolvla_base", "scope": "full_checkpoint_action_chunk",
        "device_elf_built": False, "device_execution": False,
        "output_validation": ("all_elements_vs_upstream_policy"
                              if golden_values is not None else "finite_range_only"),
        "output_check_tolerance": ({"rtol": 1e-2, "atol": 1e-2}
                                   if golden_values is not None else None),
        "golden_output_sha256": golden["output_sha256"] if golden else None,
        "input_image_manifest": str(input_image.resolve()) if input_image else None,
        "input_image_sha256": input_manifest["image_sha256"] if input_manifest else None,
        "upstream_execution_equivalent": False,
        "stage_limit": stage_limit, "stages": len(stages),
        "full_graph_stages": len(graph.stages),
        "execution_schedule_sha256": storage["execution_schedule_sha256"],
        "checkpoint_weight_format": "fp32",
        "checkpoint_sha256": image["checkpoint_weight_sha256"],
        "weight_image_sha256": image["image_sha256"],
        "weight_image_manifest": str(weight_image.resolve()),
        "activation_arena_bytes": storage["arena_bytes"],
        "source_spec_sha256": hashlib.sha256(
            (ROOT / "kernels/evaluation/llm/pr1-models.json").read_bytes()).hexdigest(),
        "device_source_files": ["kernel.cpp", *dependencies],
        "device_source_generated": True,
        "precision": "fp32_from_checkpoint_bf16",
    }
    (target / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-image", type=Path, required=True)
    parser.add_argument("--out-root", type=Path,
                        default=HERE / "generated/smolvla-elf")
    parser.add_argument("--stage-limit", type=int)
    parser.add_argument("--stages-per-object", type=int, default=50)
    parser.add_argument("--input-image", type=Path)
    parser.add_argument("--golden-output", type=Path)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    target = generate(args.weight_image, args.out_root, args.stage_limit,
                      args.stages_per_object, args.input_image,
                      args.golden_output)
    if args.build:
        build_device(target)
    print(target)


if __name__ == "__main__":
    main()
