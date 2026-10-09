#!/usr/bin/env python3
"""Compile a connected, reduced decoder graph into a Radiance SoC ELF.

The input shapes and weights are deliberately small and generated. This is an
operator handoff and compiler check, not a checkpoint or throughput result.
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
             device_check_all_stages: bool = False) -> Path:
    specs = stitch.model_specs()
    spec = reference.reduced_spec(specs[model])
    spec["num_hidden_layers"] = layers
    graph = stitch.build(model, prefill=prefill, decode_steps=decode,
                         specs={model: spec}, generation=generation)
    inputs = {name: np.arange(1, shape["shape"][1] + 1, dtype=np.int32)[None]
              for name, shape in graph.tensors.items()
              if name.endswith(".token_ids") and name.startswith(("prefill", "decode"))
              and name not in {stage["writes"] for stage in graph.stages}}
    values = reference.execute(graph, inputs)
    target = (out_root / model).resolve()
    target.mkdir(parents=True, exist_ok=True)
    declarations = []
    for name, tensor in graph.tensors.items():
        data = inputs.get(name)
        dtype = "int32_t" if tensor["dtype"] == "int32" else "float"
        declarations.append(array_decl(symbol(name), data,
                                       int(np.prod(tensor["shape"])), dtype))
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
    half = spec["head_dim"] // 2
    max_position = prefill + decode
    inv_freq = np.power(10000.0, -np.arange(half, dtype=np.float32) / half)
    angles = np.arange(max_position, dtype=np.float32)[:, None] * inv_freq
    declarations.append(array_decl("v_rope_cos", np.cos(angles), max_position * half))
    declarations.append(array_decl("v_rope_sin", np.sin(angles), max_position * half))
    stage_src = []
    for i, stage in enumerate(graph.stages):
        param = param_data(stage)
        stage_src.append(emit_stage(i, stage, graph, param[0] if param else None))
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
    cpp = ("#include <mu_intrinsics.h>\n#include <mu_schedule.h>\n"
           "#include <stdint.h>\n#include \"kernel_verify.h\"\n"
           "#include \"pipeline_math.hpp\"\n"
           "extern \"C\" uint32_t __mu_num_warps = 1;\n"
           + "\n".join(declarations) + "\n" + "\n".join(stage_src) +
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
    write_if_changed(target / "native.cpp", native)
    write_if_changed(target / "host.cpp", (HERE / "host.cpp").read_text())
    write_if_changed(target / "Makefile",
        "PROJECT = model_chain\nMU_SRCS = kernel.cpp\nHOST_SRCS = host.cpp\n"
        f"RADIANCE_LIB_PATH := {ROOT / 'lib'}\n"
        "RADIANCE_INCLUDE_PATH := $(RADIANCE_LIB_PATH)/include\n"
        "GEMMINI_SW_PATH := $(RADIANCE_LIB_PATH)/mxgemmini\n"
        f"SOC_DIR := {ROOT / 'soc'}\n"
        f"LLVM_MUON ?= {ROOT / 'llvm/llvm-muon'}\n"
        f"EXTRA_MU_CFLAGS += -I{HERE}\n"
        f"include {ROOT / 'kernels/common.mk'}\n"
        f"kernel.mu.o: {HERE / 'pipeline_math.hpp'}\n")
    manifest = {
        "model": model, "scope": "reduced_synthetic_decoder",
        "device_source_generated": True, "device_elf_built": False,
        "device_execution": False, "measured_cycles": None,
        "checkpoint_weights": False,
        "full_model_dimensions": False, "layers": layers,
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
    parser.add_argument("--verify-native", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.layers <= 0 or args.prefill <= 0 or args.decode_steps < 0:
        parser.error("layers/prefill must be positive and decode steps nonnegative")
    target = generate(args.model, args.layers, args.prefill,
                      args.decode_steps, args.generation, args.out_root,
                      args.device_check_all_stages)
    if args.verify_native or args.build:
        print(verify_native(target))
    if args.build:
        build_device(target)
    print(target)


if __name__ == "__main__":
    main()
