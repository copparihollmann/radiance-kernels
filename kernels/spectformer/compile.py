#!/usr/bin/env python3
"""Compile the full upstream vanilla SpectFormer schedule into one Radiance ELF."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernels/model_chain"))
from compile_decoder import build_device, literal, write_if_changed  # noqa: E402
from prepare import sha256  # noqa: E402

PATCH, NORM, ROW, COLUMN, FILTER, ICOLUMN, IROW = range(1, 8)
LINEAR, GELU, ADD, ATTENTION, MEAN = range(8, 13)


def align64(value: int) -> int:
    return (value + 63) // 64 * 64


def generate(image_manifest: Path, out: Path,
             logit_rtol: float = 5e-3, logit_atol: float = 1e-3,
             runtime_input: bool = False) -> Path:
    manifest = json.loads(image_manifest.read_text())
    if not (math.isfinite(logit_rtol) and math.isfinite(logit_atol) and
            logit_rtol >= 0 and logit_atol >= 0):
        raise ValueError("logit tolerances must be finite and nonnegative")
    if manifest["format"] != "spectformer.vanilla.fp32.v1":
        raise ValueError("unsupported image format")
    image_path = image_manifest.parent / "image.bin"
    if sha256(image_path) != manifest["image_sha256"]:
        raise ValueError("image hash differs from manifest")
    if runtime_input:
        for name, key in (("weights.bin", "weights_sha256"),
                          ("input.bin", "input_sha256"),
                          ("golden.bin", "golden_sha256")):
            if sha256(image_manifest.parent / name) != manifest[key]:
                raise ValueError(f"{name} differs from manifest")
    e = manifest["entries"]
    h = manifest["image_size"] // 16
    d = manifest["embed_dim"]
    n = h * h
    hidden = 4 * d
    bins = h // 2 + 1
    if manifest["heads"] != 6 or d % 6 or manifest["spectral_blocks"] != 4:
        raise ValueError("source schedule differs from the checked vanilla graph")
    if manifest["depth"] < 5 or h * 16 != manifest["image_size"]:
        raise ValueError("invalid SpectFormer dimensions")

    # These named buffers are reused across blocks. Each stage has its own
    # read/write contract; the residual lives in tokens throughout the graph.
    sizes = {
        "tokens": n * d * 4, "norm": n * d * 4,
        "tmp": n * d * 4, "hidden": n * hidden * 4,
        "qkv": n * 3 * d * 4,
        "freq0": h * bins * d * 2 * 4,
        "freq1": h * bins * d * 2 * 4,
        "pool": d * 4, "logits": manifest["num_classes"] * 4,
        "attention_scratch": 32 * n * 4,
    }
    offsets = {}
    arena_bytes = 0
    for name, size in sizes.items():
        offsets[name] = arena_bytes
        arena_bytes += align64(size)

    stages: list[dict] = []

    def addr(name: str) -> int:
        if name not in e:
            raise ValueError(f"weight image lacks {name}")
        return e[name]["address"]

    def add(name: str, kind: int, src0: str | int, dst: str,
            src1: str | int = 0, weight: int = 0, bias: int = 0,
            extra: int = 0, p0: int = 0, p1: int = 0, p2: int = 0) -> None:
        stages.append({"name": name, "kind": kind,
                       "src0": offsets[src0] if isinstance(src0, str) else src0,
                       "src1": offsets[src1] if isinstance(src1, str) else src1,
                       "dst": offsets[dst], "weight": weight, "bias": bias,
                       "extra": extra, "p0": p0, "p1": p1, "p2": p2})

    def norm(name: str, src: str, dst: str, parameter: str) -> None:
        add(name, NORM, src, dst, weight=addr(parameter + ".weight"),
            bias=addr(parameter + ".bias"), p0=n, p1=d)

    def linear(name: str, src: str, dst: str, parameter: str,
               rows: int, width_in: int, width_out: int) -> None:
        weight = e[parameter + ".weight"]
        if weight["shape"] != [width_in, width_out]:
            raise ValueError(f"{parameter}: unexpected linear weight shape")
        add(name, LINEAR, src, dst, weight=addr(parameter + ".weight"),
            bias=addr(parameter + ".bias"),
            p0=rows, p1=width_in, p2=width_out)

    add("patch_embed", PATCH, addr("input.image"), "tokens",
        weight=addr("patch_embed.proj.weight"),
        bias=addr("patch_embed.proj.bias"), extra=addr("pos_embed"),
        p0=manifest["image_size"], p1=16, p2=3)
    for layer in range(manifest["depth"]):
        prefix = f"blocks.{layer}"
        norm(f"{prefix}.norm1", "tokens", "norm", prefix + ".norm1")
        if layer < 4:
            add(f"{prefix}.fft_rows", ROW, "norm", "freq0", p0=h, p1=h, p2=d)
            add(f"{prefix}.fft_columns", COLUMN, "freq0", "freq1", p0=h,
                p1=bins, p2=d)
            add(f"{prefix}.complex_filter", FILTER, "freq1", "freq0",
                weight=addr(prefix + ".filter.complex_weight"),
                p0=h * bins * d)
            add(f"{prefix}.ifft_columns", ICOLUMN, "freq0", "freq1",
                p0=h, p1=bins, p2=d)
            add(f"{prefix}.ifft_rows", IROW, "freq1", "tmp", p0=h, p1=h, p2=d)
            norm(f"{prefix}.norm2", "tmp", "norm", prefix + ".norm2")
        else:
            linear(f"{prefix}.qkv", "norm", "qkv", prefix + ".attn.qkv",
                   n, d, 3 * d)
            add(f"{prefix}.attention", ATTENTION, "qkv", "tmp",
                p0=n, p1=d, p2=6)
            linear(f"{prefix}.attn.proj", "tmp", "norm", prefix + ".attn.proj",
                   n, d, d)
            add(f"{prefix}.attention_residual", ADD, "tokens", "tokens",
                src1="norm", p0=n * d)
            norm(f"{prefix}.norm2", "tokens", "norm", prefix + ".norm2")
        linear(f"{prefix}.mlp.fc1", "norm", "hidden", prefix + ".mlp.fc1",
               n, d, hidden)
        add(f"{prefix}.mlp.gelu", GELU, "hidden", "hidden", p0=n * hidden)
        linear(f"{prefix}.mlp.fc2", "hidden", "tmp", prefix + ".mlp.fc2",
               n, hidden, d)
        add(f"{prefix}.mlp_residual", ADD, "tokens", "tokens",
            src1="tmp", p0=n * d)
    norm("final_norm", "tokens", "norm", "norm")
    add("token_mean", MEAN, "norm", "pool", p0=n, p1=d)
    linear("head", "pool", "logits", "head", 1, d,
           manifest["num_classes"])
    if len(stages) != 4 * 11 + (manifest["depth"] - 4) * 10 + 4:
        raise AssertionError("SpectFormer schedule has an unexpected stage count")
    trace_points = [{"name": stage["name"].removesuffix(".mlp_residual"),
                     "stage_index": index, "offset": offsets["tokens"],
                     "count": n * d}
                    for index, stage in enumerate(stages)
                    if stage["name"].startswith("blocks.") and
                    stage["name"].endswith(".mlp_residual")]
    out.mkdir(parents=True, exist_ok=True)
    twiddle = np.arange(h, dtype=np.float32)
    phase = 2 * np.pi * np.outer(twiddle, twiddle) / h
    def float_array(name: str, values: np.ndarray) -> str:
        data = ", ".join(literal(x) for x in values.ravel())
        return f"alignas(64) __global float {name}[{align64(values.size * 4)//4}] = {{{data}}};\n"
    data = ("#include <mu_intrinsics.h>\n"
            f"alignas(64) __global unsigned char v_arena[{arena_bytes}] = {{0}};\n"
            + float_array("v_cosine", np.cos(phase).astype(np.float32))
            + float_array("v_sine", np.sin(phase).astype(np.float32)))
    write_if_changed(out / "model_data.cpp", data)
    stage_records = ",\n".join(
        "  {" + ", ".join(f"{stage[key]}u" for key in
                         ("kind", "src0", "src1", "dst", "weight", "bias",
                          "extra", "p0", "p1", "p2")) + "}"
        for stage in stages)
    padded_stages = (len(stages) + 7) // 8 * 8
    kernel = '''#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
#include "pipeline_smolvla.hpp"
#include "spectral.hpp"
extern "C" uint32_t __mu_num_warps = 1;
extern __global unsigned char v_arena[];
extern __global float v_cosine[];
extern __global float v_sine[];
#ifdef SPECTFORMER_STAGE_TRACE
extern "C" void sf_trace(uint32_t, const float*, uint32_t);
#endif
struct Stage { uint32_t kind, src0, src1, dst, weight, bias, extra, p0, p1, p2; };
alignas(64) static Stage v_stages[STAGE_CAPACITY] = {
STAGE_RECORDS
};
static void run_stage(void* raw, uint32_t tid, uint32_t tpb, uint32_t) {
  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return;
  tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS
      + tid % MU_NUM_THREADS;
  tpb /= MU_NUM_CORES;
  Stage* s = (Stage*)raw;
  __global float* a = (__global float*)(v_arena + s->src0);
  __global float* b = (__global float*)(v_arena + s->src1);
  __global float* z = (__global float*)(v_arena + s->dst);
  const __global float* w = (const __global float*)s->weight;
  const __global float* bias = (const __global float*)s->bias;
  switch (s->kind) {
  case 1: model_chain::patch_embed((const __global float*)s->src0, w, bias,
             (const __global float*)s->extra, z, s->p0, s->p1, s->p2,
             SF_DIM, tid, tpb); break;
  case 2: model_chain::layernorm(a, w, bias, z, s->p0, s->p1,
             1.0e-6f, tid, tpb); break;
  case 3: spectformer::real_dft_rows(a, z, v_cosine, v_sine,
             s->p0, s->p1, s->p2, tid, tpb); break;
  case 4: spectformer::complex_dft_columns(a, z, v_cosine, v_sine,
             s->p0, s->p1, s->p2, false, tid, tpb); break;
  case 5: spectformer::complex_filter(a, w, z, s->p0, tid, tpb); break;
  case 6: spectformer::complex_dft_columns(a, z, v_cosine, v_sine,
             s->p0, s->p1, s->p2, true, tid, tpb); break;
  case 7: spectformer::inverse_real_dft_rows(a, z, v_cosine, v_sine,
             s->p0, s->p1, s->p2, tid, tpb); break;
  case 8: model_chain::linear(a, w, bias, z, s->p0, s->p1, s->p2,
             tid, tpb); break;
  case 9: spectformer::gelu_exact(a, z, s->p0, tid, tpb); break;
  case 10: model_chain::residual(a, b, z, s->p0, tid, tpb); break;
  case 11: spectformer::attention_qkv(a, z,
             (__global float*)(v_arena + SCRATCH_OFFSET), s->p0, s->p1,
             s->p2, tid, tpb); break;
  case 12: spectformer::token_mean(a, z, s->p0, s->p1, tid, tpb); break;
  }
}
int main() {
  for (uint32_t stage = 0; stage < STAGE_COUNT; ++stage) {
    mu_schedule(run_stage, &v_stages[stage], 1);
    mu_barrier(0, MU_NUM_CORES);
    mu_fence();
TRACE_BLOCK
  }
  if (mu_hart_id() != 0) { for (;;) {} }
  const __global float* result =
      (const __global float*)(v_arena + LOGITS_OFFSET);
VERIFY_BLOCK
}
'''
    verify_block = ("  (void)result;\n  mu_tohost(0u); return 0;" if runtime_input else
                    "  const __global float* golden = (const __global float*)GOLDEN_ADDRESS;\n"
                    "  for (uint32_t i = 0; i < NUM_CLASSES; ++i)\n"
                    "    if (!mu_close(result[i], golden[i], VERIFY_RTOL, VERIFY_ATOL)) {\n"
                    "      mu_tohost((i << 1) | 1u); return 0;\n"
                    "    }\n"
                    "  mu_tohost(0u); return 0;")
    trace_block = ("#ifdef SPECTFORMER_STAGE_TRACE\n" +
                   "\n".join(f"    if (stage == {point['stage_index']}u) "
                             f"sf_trace({index}u, (const float*)(v_arena + "
                             f"{point['offset']}u), {point['count']}u);"
                             for index, point in enumerate(trace_points)) +
                   "\n#endif")
    for key, value in {
        "VERIFY_BLOCK": verify_block,
        "TRACE_BLOCK": trace_block,
        "STAGE_CAPACITY": padded_stages,
        "STAGE_RECORDS": stage_records,
        "STAGE_COUNT": len(stages),
        "SF_DIM": d,
        "SCRATCH_OFFSET": offsets["attention_scratch"],
        "LOGITS_OFFSET": offsets["logits"],
        "GOLDEN_ADDRESS": f"0x{addr('output.logits'):08x}u",
        "NUM_CLASSES": manifest["num_classes"],
        "VERIFY_RTOL": literal(logit_rtol),
        "VERIFY_ATOL": literal(logit_atol),
    }.items():
        kernel = kernel.replace(key, str(value))
    write_if_changed(out / "kernel.cpp", kernel)
    shutil.copyfile(ROOT / "kernels/model_chain/host.cpp", out / "host.cpp")
    makefile = ("PROJECT = spectformer\nMU_SRCS = kernel.cpp\nHOST_SRCS = host.cpp\n"
                "MU_SRC_DEPS = model_data.cpp\n"
                f"RADIANCE_LIB_PATH := {ROOT / 'lib'}\n"
                "RADIANCE_INCLUDE_PATH := $(RADIANCE_LIB_PATH)/include\n"
                "GEMMINI_SW_PATH := $(RADIANCE_LIB_PATH)/mxgemmini\n"
                f"SOC_DIR := {ROOT / 'soc'}\n"
                f"LLVM_MUON ?= {ROOT / 'llvm/llvm-muon'}\n"
                f"EXTRA_MU_CFLAGS += -I{HERE} -I{ROOT / 'kernels/model_chain'}\n"
                f"include {ROOT / 'kernels/common.mk'}\n"
                f"kernel.mu.o: {HERE / 'spectral.hpp'}\n")
    write_if_changed(out / "Makefile", makefile)
    result = {
        "model": "spectformer_vanilla", "source_image_manifest": str(image_manifest.resolve()),
        "image_sha256": manifest["image_sha256"], "image_size": manifest["image_size"],
        "embed_dim": d, "depth": manifest["depth"], "num_classes": manifest["num_classes"],
        "stages": len(stages), "stage_names": [stage["name"] for stage in stages],
        "trace_points": trace_points,
        "arena_bytes": arena_bytes, "logits_offset": offsets["logits"],
        "weight_provenance": manifest["weight_provenance"],
        "runtime_input": runtime_input,
        "weights_sha256": manifest.get("weights_sha256"),
        "input_address": addr("input.image"),
        "weights_base_address": manifest["base_address"],
        "upstream_source_sha256": manifest["upstream_source_sha256"],
        "device_elf_built": False, "device_execution": False,
        "performance_measurement": False,
        "logit_rtol": logit_rtol, "logit_atol": logit_atol,
        "validation": "full_logit_vector_against_upstream_fp32",
    }
    write_if_changed(out / "manifest.json", json.dumps(result, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--runtime-input", action="store_true",
                        help="read weights and input separately; compare logits in the runner")
    parser.add_argument("--logit-rtol", type=float, default=5e-3)
    parser.add_argument("--logit-atol", type=float, default=1e-3)
    args = parser.parse_args()
    target = generate(args.image_manifest, args.out,
                      args.logit_rtol, args.logit_atol,
                      args.runtime_input)
    if args.build:
        build_device(target)
    print(target)


if __name__ == "__main__":
    main()
