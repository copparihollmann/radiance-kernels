#!/usr/bin/env python3
"""Generate a connected FP32 Radiance ELF for hierarchical SpectFormer S/B/L."""

from __future__ import annotations

import argparse
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


SPECS = {
    "h-s": ([64, 128, 320, 448], [2, 4, 10, 14], [8, 8, 4, 4], [3, 4, 6, 3], 32),
    "h-b": ([64, 128, 320, 512], [2, 4, 10, 16], [8, 8, 4, 4], [3, 4, 12, 3], 64),
    "h-l": ([96, 192, 384, 512], [3, 6, 12, 16], [8, 8, 4, 4], [3, 6, 18, 3], 64),
}
CONV_INPUT, CONV, BN_RELU, NORM, ROW, COLUMN, FILTER, ICOLUMN, IROW = range(1, 10)
LINEAR, GELU, ADD, DEPTHWISE, ATTENTION, CLS_INIT, AUX_MAX, COMBINE = range(10, 18)


def align64(value: int) -> int:
    return (value + 63) // 64 * 64


def generate(image_manifest: Path, out: Path, runtime_input: bool = True,
             logit_rtol: float = 5e-3, logit_atol: float = 1e-3,
             stop_after_stem: bool = False) -> Path:
    image = json.loads(image_manifest.read_text())
    if image["format"] != "spectformer.hierarchical.fp32.v1":
        raise ValueError("wrong image format")
    if stop_after_stem and not runtime_input:
        raise ValueError("a prefix build requires runtime input and an external reference")
    if not (math.isfinite(logit_rtol) and math.isfinite(logit_atol) and
            logit_rtol >= 0 and logit_atol >= 0):
        raise ValueError("invalid tolerance")
    variant = image["variant"]
    dims, heads, ratios, depths, stem = SPECS[variant]
    if image["image_size"] != 224 or image["num_classes"] != 1000:
        raise ValueError("hierarchical compiler supports canonical 224/1000 only")
    source_image = image_manifest.parent / "image.bin"
    if sha256(source_image) != image["image_sha256"]:
        raise ValueError("image differs from manifest")
    for name, key in (("weights.bin", "weights_sha256"),
                      ("input.bin", "input_sha256"),
                      ("golden.bin", "golden_sha256")):
        if sha256(image_manifest.parent / name) != image[key]:
            raise ValueError(f"{name} differs from manifest")
    entries = image["entries"]
    shapes = [(56, dims[0]), (28, dims[1]), (14, dims[2]), (7, dims[3])]
    max_tokens = max(h * h * d for h, d in shapes)
    max_hidden = max(h * h * d * ratio for (h, d), ratio in zip(shapes, ratios))
    max_attention = max(14 * 14, 7 * 7, 50)
    spectrum = max(h * (h // 2 + 1) * d * 2 for h, d in shapes[:2])
    sizes = {
        "tokens": max_tokens * 4, "norm": max_tokens * 4,
        "tmp": max_tokens * 4, "proj": max_tokens * 4,
        "hidden": max_hidden * 4, "hidden2": max_hidden * 4,
        "stem0": 112 * 112 * stem * 4, "stem1": 112 * 112 * stem * 4,
        "q": max(14 * 14 * dims[2], 7 * 7 * dims[3], 50 * dims[3]) * 4,
        "kv": 2 * max(14 * 14 * dims[2], 7 * 7 * dims[3], 50 * dims[3]) * 4,
        "freq0": spectrum * 4, "freq1": spectrum * 4,
        "class_tokens": 50 * dims[3] * 4,
        "aux": 49 * 1000 * 4, "head_logits": 1000 * 4,
        "pool": 1000 * 4, "logits": 1000 * 4,
        "attention_scratch": 32 * max_attention * 4,
    }
    offsets: dict[str, int] = {}
    arena_bytes = 0
    for name, size in sizes.items():
        offsets[name] = arena_bytes
        arena_bytes += align64(size)
    stages: list[dict] = []

    def addr(name: str) -> int:
        if name not in entries:
            raise ValueError(f"missing upstream tensor {name}")
        return entries[name]["address"]

    def offset(value: str | int) -> int:
        return offsets[value] if isinstance(value, str) else value

    def add(name: str, kind: int, src0: str | int, dst: str | int,
            src1: str | int = 0, weight: int = 0, bias: int = 0,
            extra: int = 0, dims6: tuple[int, ...] = ()) -> None:
        parameters = list(dims6) + [0] * (6 - len(dims6))
        if len(parameters) != 6:
            raise ValueError(f"{name}: too many parameters")
        stages.append({"name": name, "kind": kind, "src0": offset(src0),
                       "src1": offset(src1), "dst": offset(dst),
                       "weight": weight, "bias": bias, "extra": extra,
                       **{f"p{i}": parameters[i] for i in range(6)}})

    def norm(name: str, src: str | int, dst: str,
             parameter: str, rows: int, channels: int) -> None:
        add(name, NORM, src, dst, weight=addr(parameter + ".weight"),
            bias=addr(parameter + ".bias"), dims6=(rows, channels))

    def linear(name: str, src: str | int, dst: str,
               parameter: str, rows: int, width_in: int, width_out: int) -> None:
        if entries[parameter + ".weight"]["shape"] != [width_in, width_out]:
            raise ValueError(f"{parameter}: unexpected linear shape")
        add(name, LINEAR, src, dst, weight=addr(parameter + ".weight"),
            bias=addr(parameter + ".bias"),
            dims6=(rows, width_in, width_out))

    def conv(name: str, src: str | int, dst: str, parameter: str,
             height: int, width: int, channels_in: int, channels_out: int,
             kernel: int, stride: int, input_nchw: bool = False,
             bias: bool = True) -> None:
        if entries[parameter + ".weight"]["shape"] != [channels_out, channels_in,
                                                          kernel, kernel]:
            raise ValueError(f"{parameter}: unexpected convolution shape")
        add(name, CONV_INPUT if input_nchw else CONV, src, dst,
            weight=addr(parameter + ".weight"),
            bias=addr(parameter + ".bias") if bias else 0,
            dims6=(height, width, channels_in, channels_out, kernel, stride))

    def bn(name: str, buffer: str, parameter: str, count: int,
           channels: int) -> None:
        add(name, BN_RELU, buffer, buffer,
            src1=addr(parameter + ".running_var"),
            weight=addr(parameter + ".weight"),
            bias=addr(parameter + ".bias"),
            extra=addr(parameter + ".running_mean"),
            dims6=(count, channels))

    conv("stem.conv0", addr("input.image"), "stem0", "patch_embed1.conv.0",
         224, 224, 3, stem, 7, 2, input_nchw=True, bias=False)
    bn("stem.bn0_relu", "stem0", "patch_embed1.conv.1", 112 * 112 * stem, stem)
    conv("stem.conv1", "stem0", "stem1", "patch_embed1.conv.3",
         112, 112, stem, stem, 3, 1, bias=False)
    bn("stem.bn1_relu", "stem1", "patch_embed1.conv.4", 112 * 112 * stem, stem)
    conv("stem.conv2", "stem1", "stem0", "patch_embed1.conv.6",
         112, 112, stem, stem, 3, 1, bias=False)
    bn("stem.bn2_relu", "stem0", "patch_embed1.conv.7", 112 * 112 * stem, stem)
    conv("stem.proj", "stem0", "tmp", "patch_embed1.proj",
         112, 112, stem, dims[0], 3, 2)
    norm("stem.norm", "tmp", "tokens", "patch_embed1.norm", 56 * 56, dims[0])

    for stage_index, ((h, d), count) in enumerate(zip(shapes, depths)):
        stage = stage_index + 1
        token_count = h * h
        if stage_index:
            previous_h, previous_d = shapes[stage_index - 1]
            norm(f"stage{stage-1}.out_norm", "tokens", "norm",
                 f"norm{stage-1}", previous_h * previous_h, previous_d)
            conv(f"stage{stage}.downsample", "norm", "tmp",
                 f"patch_embed{stage}.proj", previous_h, previous_h,
                 previous_d, d, 3, 2)
            norm(f"stage{stage}.embed_norm", "tmp", "tokens",
                 f"patch_embed{stage}.norm", token_count, d)
        hidden = d * ratios[stage_index]
        for block_index in range(count):
            prefix = f"block{stage}.{block_index}"
            norm(prefix + ".norm1", "tokens", "norm", prefix + ".norm1",
                 token_count, d)
            if stage_index < 2:
                add(prefix + ".fft_rows", ROW, "norm", "freq0",
                    dims6=(h, h, d))
                add(prefix + ".fft_columns", COLUMN, "freq0", "freq1",
                    dims6=(h, h // 2 + 1, d))
                add(prefix + ".filter", FILTER, "freq1", "freq0",
                    weight=addr(prefix + ".attn.complex_weight"),
                    dims6=(h * (h // 2 + 1) * d,))
                add(prefix + ".ifft_columns", ICOLUMN, "freq0", "freq1",
                    dims6=(h, h // 2 + 1, d))
                add(prefix + ".ifft_rows", IROW, "freq1", "tmp",
                    dims6=(h, h, d))
            else:
                linear(prefix + ".q", "norm", "q", prefix + ".attn.q",
                       token_count, d, d)
                linear(prefix + ".kv", "norm", "kv", prefix + ".attn.kv",
                       token_count, d, 2 * d)
                add(prefix + ".attention", ATTENTION, "q", "tmp", src1="kv",
                    dims6=(token_count, d, heads[stage_index], token_count))
                linear(prefix + ".attn.proj", "tmp", "proj",
                       prefix + ".attn.proj", token_count, d, d)
            add(prefix + ".attn_residual", ADD, "tokens", "tokens",
                src1="tmp" if stage_index < 2 else "proj",
                dims6=(token_count * d,))
            norm(prefix + ".norm2", "tokens", "norm", prefix + ".norm2",
                 token_count, d)
            linear(prefix + ".mlp.fc1", "norm", "hidden", prefix + ".mlp.fc1",
                   token_count, d, hidden)
            parameter = prefix + ".mlp.dwconv.dwconv"
            if entries[parameter + ".weight"]["shape"] != [hidden, 1, 3, 3]:
                raise ValueError(f"{parameter}: unexpected depthwise shape")
            add(prefix + ".mlp.dwconv", DEPTHWISE, "hidden", "hidden2",
                weight=addr(parameter + ".weight"),
                bias=addr(parameter + ".bias"), dims6=(h, h, hidden))
            add(prefix + ".mlp.gelu", GELU, "hidden2", "hidden2",
                dims6=(token_count * hidden,))
            linear(prefix + ".mlp.fc2", "hidden2", "tmp", prefix + ".mlp.fc2",
                   token_count, hidden, d)
            add(prefix + ".mlp_residual", ADD, "tokens", "tokens",
                src1="tmp", dims6=(token_count * d,))

    final_d = dims[3]
    add("class_token_mean_concat", CLS_INIT, "tokens", "class_tokens",
        dims6=(49, final_d))
    norm("class.norm1", "class_tokens", "norm", "post_network.0.norm1",
         50, final_d)
    linear("class.q", "norm", "q", "post_network.0.attn.q", 1,
           final_d, final_d)
    linear("class.kv", "norm", "kv", "post_network.0.attn.kv", 50,
           final_d, 2 * final_d)
    add("class.attention", ATTENTION, "q", "tmp", src1="kv",
        dims6=(50, final_d, heads[3], 1))
    linear("class.proj", "tmp", "proj", "post_network.0.attn.proj",
           1, final_d, final_d)
    add("class.attention_residual", ADD, "class_tokens", "class_tokens",
        src1="proj", dims6=(final_d,))
    norm("class.norm2", "class_tokens", "norm", "post_network.0.norm2",
         1, final_d)
    linear("class.mlp.fc1", "norm", "hidden", "post_network.0.mlp.fc1",
           1, final_d, final_d * ratios[3])
    add("class.mlp.gelu", GELU, "hidden", "hidden",
        dims6=(final_d * ratios[3],))
    linear("class.mlp.fc2", "hidden", "tmp", "post_network.0.mlp.fc2",
           1, final_d * ratios[3], final_d)
    add("class.mlp_residual", ADD, "class_tokens", "class_tokens",
        src1="tmp", dims6=(final_d,))
    norm("final_norm", "class_tokens", "norm", "norm4", 50, final_d)
    linear("head", "norm", "head_logits", "head", 1, final_d, 1000)
    linear("aux_head", offsets["norm"] + final_d * 4, "aux",
           "aux_head", 49, final_d, 1000)
    add("aux_max", AUX_MAX, "aux", "pool", dims6=(49, 1000))
    add("head_plus_aux", COMBINE, "head_logits", "logits", src1="pool",
        dims6=(1000,))

    trace_points = []
    for index, stage in enumerate(stages):
        if stage["name"].startswith("block") and stage["name"].endswith(".mlp_residual"):
            prefix = stage["name"].removesuffix(".mlp_residual")
            stage_number = int(prefix[5]) - 1
            h, d = shapes[stage_number]
            trace_points.append({"name": prefix, "stage_index": index,
                                 "offset": offsets["tokens"], "count": h * h * d})

    if stop_after_stem:
        if stages[7]["name"] != "stem.norm":
            raise AssertionError("hierarchical stem no longer ends at stage eight")
        stages = stages[:8]
        trace_points = []
    output_offset = offsets["tokens"] if stop_after_stem else offsets["logits"]
    output_count = 56 * 56 * dims[0] if stop_after_stem else 1000
    output_kind = "hierarchical_stem" if stop_after_stem else "logits"

    out.mkdir(parents=True, exist_ok=True)
    def float_array(name: str, values: np.ndarray) -> str:
        data = ", ".join(literal(x) for x in values.ravel())
        return f"alignas(64) __global float {name}[{align64(values.size * 4)//4}] = {{{data}}};\n"
    arrays = []
    for h in (56, 28):
        phase = 2 * np.pi * np.outer(np.arange(h, dtype=np.float32),
                                     np.arange(h, dtype=np.float32)) / h
        arrays.append(float_array(f"v_cos{h}", np.cos(phase).astype(np.float32)))
        arrays.append(float_array(f"v_sin{h}", np.sin(phase).astype(np.float32)))
    write_if_changed(out / "model_data.cpp",
                     "#include <mu_intrinsics.h>\n"
                     f"alignas(64) __global unsigned char v_arena[{arena_bytes}] = {{0}};\n"
                     + "".join(arrays))
    keys = ("kind", "src0", "src1", "dst", "weight", "bias", "extra",
            "p0", "p1", "p2", "p3", "p4", "p5")
    records = ",\n".join("  {" + ", ".join(f"{stage[key]}u" for key in keys) + "}"
                         for stage in stages)
    verify = ("  (void)result; mu_tohost(0u); return 0;" if runtime_input else
              "  const __global float* golden = (const __global float*)GOLDEN_ADDRESS;\n"
              "  for (uint32_t i = 0; i < 1000; ++i)\n"
              "    if (!mu_close(result[i], golden[i], VERIFY_RTOL, VERIFY_ATOL)) {\n"
              "      mu_tohost((i << 1) | 1u); return 0;\n"
              "    }\n  mu_tohost(0u); return 0;")
    kernel = '''#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
#include "pipeline_smolvla.hpp"
#include "hierarchical_ops.hpp"
extern "C" uint32_t __mu_num_warps = 1;
extern __global unsigned char v_arena[];
extern __global float v_cos56[], v_sin56[], v_cos28[], v_sin28[];
#ifdef SPECTFORMER_STAGE_TRACE
extern "C" void sf_trace(uint32_t, const float*, uint32_t);
#endif
struct Stage { uint32_t kind, src0, src1, dst, weight, bias, extra;
               uint32_t p0, p1, p2, p3, p4, p5; };
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
  const __global float* cosine = s->p0 == 56 ? v_cos56 : v_cos28;
  const __global float* sine = s->p0 == 56 ? v_sin56 : v_sin28;
  switch (s->kind) {
  case 1: spectformer::conv2d((const __global float*)s->src0, w, bias, z,
            s->p0,s->p1,s->p2,s->p3,s->p4,s->p5,true,tid,tpb); break;
  case 2: spectformer::conv2d(a,w,bias,z,s->p0,s->p1,s->p2,s->p3,s->p4,s->p5,
            false,tid,tpb); break;
  case 3: spectformer::batchnorm_relu(a,w,bias,(const __global float*)s->extra,
            (const __global float*)s->src1,s->p0,s->p1,tid,tpb); break;
  case 4: model_chain::layernorm(a,w,bias,z,s->p0,s->p1,1.0e-6f,tid,tpb); break;
  case 5: spectformer::real_dft_rows(a,z,cosine,sine,s->p0,s->p1,s->p2,tid,tpb); break;
  case 6: spectformer::complex_dft_columns(a,z,cosine,sine,s->p0,s->p1,s->p2,
            false,tid,tpb); break;
  case 7: spectformer::complex_filter(a,w,z,s->p0,tid,tpb); break;
  case 8: spectformer::complex_dft_columns(a,z,cosine,sine,s->p0,s->p1,s->p2,
            true,tid,tpb); break;
  case 9: spectformer::inverse_real_dft_rows(a,z,cosine,sine,s->p0,s->p1,s->p2,
            tid,tpb); break;
  case 10: model_chain::linear(a,w,bias,z,s->p0,s->p1,s->p2,tid,tpb); break;
  case 11: spectformer::gelu_exact(a,z,s->p0,tid,tpb); break;
  case 12: model_chain::residual(a,b,z,s->p0,tid,tpb); break;
  case 13: spectformer::depthwise3(a,w,bias,z,s->p0,s->p1,s->p2,tid,tpb); break;
  case 14: spectformer::split_attention(a,b,z,
             (__global float*)(v_arena + SCRATCH_OFFSET),s->p0,s->p1,s->p2,
             s->p3,tid,tpb); break;
  case 15: spectformer::init_class_token(a,z,s->p0,s->p1,tid,tpb); break;
  case 16: spectformer::aux_max(a,z,s->p0,s->p1,tid,tpb); break;
  case 17: spectformer::class_combine(a,b,z,s->p0,tid,tpb); break;
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
  const __global float* result = (const __global float*)(v_arena + LOGITS_OFFSET);
VERIFY_BLOCK
}
'''
    trace_block = ("#ifdef SPECTFORMER_STAGE_TRACE\n" +
                   "\n".join(f"    if (stage == {point['stage_index']}u) "
                             f"sf_trace({index}u, (const float*)(v_arena + "
                             f"{point['offset']}u), {point['count']}u);"
                             for index, point in enumerate(trace_points)) +
                   "\n#endif")
    for key, value in {
        "STAGE_CAPACITY": (len(stages) + 7) // 8 * 8,
        "STAGE_RECORDS": records, "STAGE_COUNT": len(stages),
        "SCRATCH_OFFSET": offsets["attention_scratch"],
        "LOGITS_OFFSET": output_offset,
        "VERIFY_BLOCK": verify,
        "TRACE_BLOCK": trace_block,
        "GOLDEN_ADDRESS": f"0x{addr('output.logits'):08x}u",
        "VERIFY_RTOL": literal(logit_rtol),
        "VERIFY_ATOL": literal(logit_atol),
    }.items():
        kernel = kernel.replace(key, str(value))
    write_if_changed(out / "kernel.cpp", kernel)
    shutil.copyfile(ROOT / "kernels/model_chain/host.cpp", out / "host.cpp")
    makefile = ("PROJECT = spectformer_hierarchical\nMU_SRCS = kernel.cpp\nHOST_SRCS = host.cpp\n"
                "MU_SRC_DEPS = model_data.cpp\n"
                f"RADIANCE_LIB_PATH := {ROOT / 'lib'}\n"
                "RADIANCE_INCLUDE_PATH := $(RADIANCE_LIB_PATH)/include\n"
                "GEMMINI_SW_PATH := $(RADIANCE_LIB_PATH)/mxgemmini\n"
                f"SOC_DIR := {ROOT / 'soc'}\n"
                f"LLVM_MUON ?= {ROOT / 'llvm/llvm-muon'}\n"
                f"EXTRA_MU_CFLAGS += -I{HERE} -I{ROOT / 'kernels/model_chain'}\n"
                f"include {ROOT / 'kernels/common.mk'}\n"
                f"kernel.mu.o: {HERE / 'hierarchical_ops.hpp'} {HERE / 'spectral.hpp'}\n")
    write_if_changed(out / "Makefile", makefile)
    result = {
        "model": "spectformer_hierarchical", "variant": variant,
        "source_image_manifest": str(image_manifest.resolve()),
        "image_sha256": image["image_sha256"],
        "weights_sha256": image["weights_sha256"],
        "input_address": addr("input.image"),
        "weights_base_address": image["base_address"],
        "runtime_input": runtime_input,
        "stages": len(stages), "stage_names": [stage["name"] for stage in stages],
        "trace_points": trace_points,
        "arena_bytes": arena_bytes, "logits_offset": offsets["logits"],
        "output_offset": output_offset, "output_count": output_count,
        "output_kind": output_kind, "stop_after_stem": stop_after_stem,
        "weight_provenance": image["weight_provenance"],
        "device_elf_built": False, "device_execution": False,
        "performance_measurement": False,
        "logit_rtol": logit_rtol, "logit_atol": logit_atol,
    }
    write_if_changed(out / "manifest.json", json.dumps(result, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--fixture-check", action="store_true",
                        help="embed the exported golden in the device pass/fail check")
    parser.add_argument("--stop-after-stem", action="store_true",
                        help="device probe of the full-size convolutional stem output")
    args = parser.parse_args()
    target = generate(args.image_manifest, args.out,
                      runtime_input=not args.fixture_check,
                      stop_after_stem=args.stop_after_stem)
    if args.build:
        build_device(target)
    print(target)


if __name__ == "__main__":
    main()
