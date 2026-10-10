#!/usr/bin/env python3
"""Check a SmolVLA expert attention stage on Radiance with real graph inputs.

The four inputs are traced from the exact-input, checkpoint-weight native
schedule. An independent NumPy grouped-query attention calculation supplies
the device golden output. This is a targeted stage check, not a full run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np

from compile_decoder import build_device, literal, symbol, write_if_changed
from plan_buffers import plan
from probe_smolvla_euler import run, sha256


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernels/evaluation/llm"))
from stitch import build  # noqa: E402


STAGE_ID = "denoise0.expert00.attention"
RTOL = 5e-3
ATOL = 5e-4


def attention_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                        mask: np.ndarray) -> np.ndarray:
    """Compute grouped-query masked attention independently of the C++ stage."""
    output = np.zeros_like(q)
    q_heads, kv_heads, width = q.shape[1], k.shape[1], q.shape[2]
    if (q.shape != (50, 15, 64) or k.shape != (291, 5, 64) or
            v.shape != k.shape or mask.shape != (50, 291) or
            q_heads % kv_heads):
        raise ValueError("unexpected SmolVLA expert attention shape")
    allowed = mask.astype(bool)
    for head in range(q_heads):
        kv_head = head // (q_heads // kv_heads)
        scores = np.einsum("qd,kd->qk", q[:, head, :], k[:, kv_head, :],
                           dtype=np.float32) / np.float32(np.sqrt(width))
        scores = np.where(allowed, scores, np.float32(-1e30))
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.where(allowed, np.exp(scores), np.float32(0))
        denominator = weights.sum(axis=1, keepdims=True)
        output[:, head, :] = np.where(
            denominator > 0,
            (weights @ v[:, kv_head, :]) / np.maximum(denominator, 1),
            np.float32(0))
    return output.reshape(-1)


def generate(full_root: Path, fixture_root: Path,
             out_root: Path) -> tuple[Path, dict]:
    full = (full_root / "smolvla_base").resolve()
    manifest = json.loads((full / "manifest.json").read_text())
    graph = build("smolvla_base")
    buffers = plan(graph)
    if (manifest["stages"] != len(graph.stages) or
            manifest["radiance_elf_sha256"] != sha256(full / "kernel.radiance.elf") or
            manifest["execution_schedule_sha256"] !=
            buffers["execution_schedule_sha256"]):
        raise ValueError("full SmolVLA ELF differs from the pinned graph")
    stages = [(index + 1, stage) for index, stage in enumerate(graph.stages)]
    stage_index, stage = next((index, item) for index, item in stages
                              if item["id"] == STAGE_ID)
    if (stage_index != 938 or stage["op"] != "masked_gqa" or
            stage["attrs"] != {"q_heads": 15, "kv_heads": 5,
                               "head_dim": 64, "mode": "self",
                               "cache_mode": "temporary_suffix"} or
            stage["shape"] != [1, 50, 960]):
        raise ValueError("expert attention graph is not the expected full stage")
    fixture = json.loads((fixture_root / "fixture-result.json").read_text())
    native = json.loads((HERE / "evaluation/smolvla-full-fp32-native-results.json")
                        .read_text())
    if (fixture["status"] != "passed" or fixture["stage_count"] != stage_index or
            fixture["device_elf_sha256"] != manifest["radiance_elf_sha256"] or
            fixture["generated_source_sha256"] !=
            native["generated_source_sha256"] or
            native["status"] != "passed" or
            native["device_elf_sha256"] != manifest["radiance_elf_sha256"] or
            sha256(Path(fixture["log_path"])) != fixture["log_sha256"] or
            sha256(fixture_root / "smolvla_base/native/smolvla_native") !=
            fixture["native_binary_sha256"]):
        raise ValueError("attention fixtures do not come from verified native sources")
    trace_dir = fixture_root / "traces"
    producer_indices = {item["id"]: index for index, item in stages}
    inputs = []
    input_paths = []
    for name in stage["reads"]:
        if name not in producer_indices or producer_indices[name] >= stage_index:
            raise ValueError(f"unresolved attention input: {name}")
        producer = graph.stages[producer_indices[name] - 1]
        path = trace_dir / f"{producer_indices[name]:04d}.bin"
        dtype = "<u4" if name.endswith("attention_mask") else "<f4"
        values = np.fromfile(path, dtype=dtype)
        if values.size != int(np.prod(producer["shape"])):
            raise ValueError(f"bad attention fixture: {path}")
        if dtype == "<f4" and not np.all(np.isfinite(values)):
            raise ValueError(f"nonfinite attention fixture: {path}")
        inputs.append(values.reshape(producer["shape"]))
        input_paths.append(path)
    q, k, v, mask = inputs
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("attention mask contains values other than zero or one")
    expected = attention_reference(q.reshape(50, 15, 64),
                                   k.reshape(291, 5, 64),
                                   v.reshape(291, 5, 64),
                                   mask.reshape(50, 291))
    traced_output_path = trace_dir / f"{stage_index:04d}.bin"
    traced_output = np.fromfile(traced_output_path, dtype="<f4")
    if traced_output.size != expected.size or not np.all(np.isfinite(traced_output)):
        raise ValueError("bad native attention output fixture")
    maximum_error = float(np.max(np.abs(traced_output - expected)))
    if not np.allclose(traced_output, expected, rtol=RTOL, atol=ATOL):
        raise ValueError("independent attention calculation differs from native stage")
    zero_failures = int(np.count_nonzero(
        np.abs(expected) > ATOL + RTOL * np.abs(expected)))
    if zero_failures == 0:
        raise ValueError("attention comparison cannot distinguish zero output")

    target = (out_root / "smolvla_base").resolve()
    target.mkdir(parents=True, exist_ok=True)
    needle = f"void stage_{symbol(STAGE_ID)}("
    chunks = [full / name for name in manifest["device_source_files"]
              if name.startswith("stage_chunk_") and
              needle in (full / name).read_text()]
    if len(chunks) != 1:
        raise ValueError("expected one generated expert attention source chunk")
    chunk = chunks[0]
    shutil.copyfile(chunk, target / chunk.name)
    if sha256(chunk) != sha256(target / chunk.name):
        raise ValueError("copied attention stage source differs")
    shutil.copyfile(full / "model_data.cpp", target / "model_data.cpp")
    shutil.copyfile(full / "host.cpp", target / "host.cpp")
    names = ["q", "k", "v", "mask", "expected"]
    arrays = [*inputs, expected]
    declarations = []
    for name, values in zip(names, arrays):
        flat = values.ravel()
        padded = ((flat.size + 15) // 16) * 16
        if name == "mask":
            declarations.append(
                f"alignas(64) __global uint32_t v_probe_{name}[{padded}] = {{" +
                ", ".join(f"{int(value)}u" for value in flat) + "};")
        else:
            declarations.append(
                f"alignas(64) __global float v_probe_{name}[{padded}] = {{" +
                ", ".join(literal(value) for value in flat) + "};")
    write_if_changed(target / "probe_data.cpp", "#include <mu_intrinsics.h>\n"
                     "#include <stdint.h>\n" + "\n".join(declarations) + "\n")
    offsets = [buffers["allocation"][name]["offset_bytes"]
               for name in [*stage["reads"], stage["writes"]]]
    copies = []
    for name, values, offset in zip(names[:-1], inputs, offsets[:-1]):
        scalar = "uint32_t" if name == "mask" else "float"
        copies.append(f"""
  __global {scalar}* dest_{name} = (__global {scalar}*)(v_arena + {offset}u);
  for (uint32_t j = tid; j < {values.size}u; j += tpb)
    dest_{name}[j] = v_probe_{name}[j];""")
    kernel = """#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
extern "C" uint32_t __mu_num_warps = 1;
extern __global unsigned char v_arena[];
""" + "\n".join(
        f"extern __global {'uint32_t' if name == 'mask' else 'float'} v_probe_{name}[];"
        for name in names) + "\n" + f"""
void stage_{symbol(STAGE_ID)}(void*, uint32_t, uint32_t, uint32_t);
static void load_inputs(void*, uint32_t tid, uint32_t tpb, uint32_t) {{
  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return;
  tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS
      + tid % MU_NUM_THREADS;
  tpb /= MU_NUM_CORES;
""" + "\n".join(copies) + f"""
}}
int main() {{
  mu_schedule(load_inputs, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES); mu_fence();
  mu_schedule(stage_{symbol(STAGE_ID)}, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES); mu_fence();
  if (mu_hart_id() != 0) {{ for (;;) {{}} }}
  const __global float* output = (const __global float*)(v_arena + {offsets[-1]}u);
  for (uint32_t j = 0; j < {expected.size}u; ++j) {{
    if (!(output[j] == output[j]) ||
        !mu_close(output[j], v_probe_expected[j], {RTOL}f, {ATOL}f)) {{
      mu_tohost((j << 1) | 1u); return 0;
    }}
  }}
  mu_tohost(0u); return 0;
}}
"""
    write_if_changed(target / "kernel.cpp", kernel)
    makefile = ("PROJECT = model_chain\nMU_SRCS = kernel.cpp\nHOST_SRCS = host.cpp\n"
                f"MU_SRC_DEPS = model_data.cpp probe_data.cpp {chunk.name}\n"
                f"RADIANCE_LIB_PATH := {ROOT / 'lib'}\n"
                f"RADIANCE_INCLUDE_PATH := {ROOT / 'lib/include'}\n"
                f"GEMMINI_SW_PATH := {ROOT / 'lib/mxgemmini'}\n"
                f"SOC_DIR := {ROOT / 'soc'}\n"
                f"LLVM_MUON ?= {ROOT / 'llvm/llvm-muon'}\n"
                f"EXTRA_MU_CFLAGS += -I{HERE}\n"
                f"include {ROOT / 'kernels/common.mk'}\n"
                f"{chunk.name.replace('.cpp', '.mu.o')}: "
                f"{HERE / 'pipeline_smolvla.hpp'}\n")
    write_if_changed(target / "Makefile", makefile)
    record = {
        "model": "smolvla_base", "scope": "expert_masked_gqa_real_graph_state",
        "stage_id": STAGE_ID, "stage_index_one_based": stage_index,
        "source_full_elf_sha256": manifest["radiance_elf_sha256"],
        "source_stage_chunk_sha256": sha256(chunk),
        "source_stage_chunk_name": chunk.name,
        "source_native_result_sha256": sha256(
            HERE / "evaluation/smolvla-full-fp32-native-results.json"),
        "fixture_native_binary_sha256": fixture["native_binary_sha256"],
        "fixture_native_log_sha256": fixture["log_sha256"],
        "fixture_input_sha256": {name: sha256(path) for name, path in
                                 zip(stage["reads"], input_paths)},
        "fixture_native_output_sha256": sha256(traced_output_path),
        "independent_numpy_output_sha256": hashlib.sha256(
            expected.astype("<f4").tobytes()).hexdigest(),
        "independent_numpy_max_abs_error_vs_native": maximum_error,
        "zero_output_rejected_elements": zero_failures,
        "input_offsets_bytes": offsets[:-1],
        "output_offset_bytes": offsets[-1],
        "output_elements": int(expected.size),
        "output_check_tolerance": {"rtol": RTOL, "atol": ATOL},
        "device_elf_built": False, "device_execution": False,
        "upstream_execution_equivalent": False,
    }
    (target / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return target, record


def run_negative_control(target: Path, record: dict, timeout: int,
                         result_path: Path) -> dict:
    negative = target.parent.with_name(target.parent.name + "-negative") / target.name
    shutil.copytree(target, negative, dirs_exist_ok=True)
    data_path = negative / "probe_data.cpp"
    modified, replacements = re.subn(
        r"(v_probe_expected\[48000\] = \{)[^,]+",
        r"\g<1>1000000.0f", data_path.read_text(), count=1)
    if replacements != 1:
        raise ValueError("could not mutate the attention output golden")
    write_if_changed(data_path, modified)
    build_device(negative)
    negative_record = dict(record)
    negative_record.update(
        mutation="first_expected_attention_value_replaced_by_1000000.0f",
        mutated_probe_data_sha256=sha256(data_path),
        positive_probe_elf_sha256=record["device_elf_sha256"])
    result = run(negative.resolve(), negative_record, timeout)
    if (result["status"] != "failed" or result["tohost"] != 1 or
            result["failure_reason"] != "device_output_mismatch" or
            result["device_elf_sha256"] == record["device_elf_sha256"]):
        raise ValueError("attention device check did not reject injected error")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-generated-root", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--negative-out", type=Path)
    args = parser.parse_args()
    target, record = generate(args.full_generated_root, args.fixture_root,
                              args.out_root)
    build_device(target)
    record.update(device_elf_built=True,
                  device_elf_sha256=sha256(target / "kernel.radiance.elf"))
    result = run(target, record, args.timeout)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "passed":
        raise ValueError(f"expert attention device probe failed: {result['failure_reason']}")
    if args.negative_out:
        negative = run_negative_control(target, record, args.timeout,
                                        args.negative_out)
        print(f"SmolVLA expert attention negative control: "
              f"rejected at tohost={negative['tohost']}")
    print(f"SmolVLA expert attention: passed, "
          f"{result['cycles_functional']} functional cycles")


if __name__ == "__main__":
    main()
