#!/usr/bin/env python3
"""Run a checkpoint-weight SmolVLA expert projection on Radiance.

The input is traced from the complete native action schedule. The unchanged
generated stage reads its packed weights from the full model's device image.
An independent NumPy matrix product checks every output element.
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
from run_smolvla_functional import run as run_functional, sha256


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernels/evaluation/llm"))
from stitch import build  # noqa: E402


STAGE_ID = "denoise0.expert00.q_proj"
RTOL = 5e-3
ATOL = 5e-4


def generate(full_root: Path, fixture_root: Path,
             out_root: Path) -> tuple[Path, dict]:
    full = (full_root / "smolvla_base").resolve()
    manifest = json.loads((full / "manifest.json").read_text())
    graph = build("smolvla_base")
    buffers = plan(graph)
    stage_index = next(index for index, stage in enumerate(graph.stages, 1)
                       if stage["id"] == STAGE_ID)
    stage = graph.stages[stage_index - 1]
    if (stage_index != 931 or stage["op"] != "linear" or
            stage["shape"] != [1, 50, 15, 64] or
            stage["reads"] != ["denoise0.expert00.input_norm"] or
            stage["attrs"]["weight_shape"] != [720, 960] or
            manifest["stages"] != len(graph.stages) or
            manifest["radiance_elf_sha256"] != sha256(full / "kernel.radiance.elf") or
            manifest["execution_schedule_sha256"] !=
            buffers["execution_schedule_sha256"]):
        raise ValueError("full ELF or expert projection graph differs from fixture")
    fixture = json.loads((fixture_root / "fixture-result.json").read_text())
    native = json.loads((HERE / "evaluation/smolvla-full-fp32-native-results.json")
                        .read_text())
    if (fixture["status"] != "passed" or fixture["stage_count"] != stage_index or
            fixture["device_elf_sha256"] != manifest["radiance_elf_sha256"] or
            fixture["generated_source_sha256"] != native["generated_source_sha256"] or
            sha256(Path(fixture["log_path"])) != fixture["log_sha256"] or
            sha256(fixture_root / "smolvla_base/native/smolvla_native") !=
            fixture["native_binary_sha256"] or native["status"] != "passed"):
        raise ValueError("projection inputs lack verified full-schedule provenance")

    image_path = Path(manifest["weight_image_manifest"])
    image = json.loads(image_path.read_text())
    if (image["image_sha256"] != manifest["weight_image_sha256"] or
            image["checkpoint_weight_sha256"] != manifest["checkpoint_sha256"]):
        raise ValueError("packed weights differ from the pinned checkpoint")
    parameter = next(item for item in image["parameters"]
                     if item["logical_name"] == stage["attrs"]["parameter"])
    if (parameter["storage_dtype"] != "fp32" or
            parameter["shape"] != [720, 960] or
            parameter["checkpoint_shape"] != [960, 720] or
            parameter["size_bytes"] != 720 * 960 * 4 or
            parameter["gpu_address"] !=
            image["segments"][parameter["segment_index"]]["gpu_base_address"]
            + parameter["offset_bytes"]):
        raise ValueError("unexpected expert projection weight layout")
    segment = image["segments"][parameter["segment_index"]]
    with (image_path.parent / segment["image_file"]).open("rb") as stream:
        stream.seek(parameter["offset_bytes"])
        packed = stream.read(parameter["size_bytes"])
    if hashlib.sha256(packed).hexdigest() != parameter["packed_sha256"]:
        raise ValueError("expert projection weight differs from image manifest")
    weights = np.frombuffer(packed, dtype="<f4").reshape(720, 960)

    input_path = fixture_root / "traces" / "0930.bin"
    output_path = fixture_root / "traces" / "0931.bin"
    inputs = np.fromfile(input_path, dtype="<f4")
    native_output = np.fromfile(output_path, dtype="<f4")
    if (inputs.size != 50 * 720 or native_output.size != 50 * 960 or
            not np.all(np.isfinite(inputs)) or
            not np.all(np.isfinite(native_output))):
        raise ValueError("expert projection trace is incomplete or nonfinite")
    expected = (inputs.reshape(50, 720) @ weights).astype("<f4").ravel()
    error = float(np.max(np.abs(native_output - expected)))
    if not np.allclose(native_output, expected, rtol=RTOL, atol=ATOL):
        raise ValueError("NumPy projection disagrees with full native schedule")
    zero_failures = int(np.count_nonzero(
        np.abs(expected) > ATOL + RTOL * np.abs(expected)))
    if zero_failures == 0:
        raise ValueError("projection output check cannot reject all-zero output")

    needle = f"void stage_{symbol(STAGE_ID)}("
    chunks = [full / name for name in manifest["device_source_files"]
              if name.startswith("stage_chunk_") and
              needle in (full / name).read_text()]
    if len(chunks) != 1:
        raise ValueError("expected exactly one generated projection source chunk")
    chunk = chunks[0]
    if f"(const __global float*)0x{parameter['gpu_address']:x}u" not in chunk.read_text():
        raise ValueError("device source does not address packed projection weights")
    target = (out_root / "smolvla_base").resolve()
    target.mkdir(parents=True, exist_ok=True)
    for source in (chunk, full / "model_data.cpp", full / "host.cpp"):
        shutil.copyfile(source, target / source.name)
    input_offset = buffers["allocation"][stage["reads"][0]]["offset_bytes"]
    output_offset = buffers["allocation"][stage["writes"]]["offset_bytes"]
    declarations = []
    for name, values in (("input", inputs), ("expected", expected)):
        declarations.append(
            f"alignas(64) __global float v_probe_{name}[{values.size}] = {{" +
            ", ".join(literal(value) for value in values) + "};")
    write_if_changed(target / "probe_data.cpp", "#include <mu_intrinsics.h>\n" +
                     "\n".join(declarations) + "\n")
    kernel = """#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
extern "C" uint32_t __mu_num_warps = 1;
extern __global unsigned char v_arena[];
extern __global float v_probe_input[];
extern __global float v_probe_expected[];
""" + f"""
void stage_{symbol(STAGE_ID)}(void*, uint32_t, uint32_t, uint32_t);
static void load_input(void*, uint32_t tid, uint32_t tpb, uint32_t) {{
  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return;
  tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS
      + tid % MU_NUM_THREADS;
  tpb /= MU_NUM_CORES;
  __global float* dest = (__global float*)(v_arena + {input_offset}u);
  for (uint32_t j = tid; j < {inputs.size}u; j += tpb)
    dest[j] = v_probe_input[j];
}}
int main() {{
  mu_schedule(load_input, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES); mu_fence();
  mu_schedule(stage_{symbol(STAGE_ID)}, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES); mu_fence();
  if (mu_hart_id() != 0) {{ for (;;) {{}} }}
  const __global float* output = (const __global float*)(v_arena + {output_offset}u);
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
        "model": "smolvla_base", "scope": "expert_q_projection_real_graph_state",
        "stages": 1, "full_graph_stages": len(graph.stages),
        "stage_id": STAGE_ID, "stage_index_one_based": stage_index,
        "source_full_elf_sha256": manifest["radiance_elf_sha256"],
        "source_stage_chunk_sha256": sha256(chunk),
        "source_stage_chunk_name": chunk.name,
        "source_native_result_sha256": sha256(
            HERE / "evaluation/smolvla-full-fp32-native-results.json"),
        "fixture_native_binary_sha256": fixture["native_binary_sha256"],
        "fixture_native_log_sha256": fixture["log_sha256"],
        "fixture_input_sha256": sha256(input_path),
        "fixture_native_output_sha256": sha256(output_path),
        "independent_numpy_output_sha256": hashlib.sha256(
            expected.tobytes()).hexdigest(),
        "independent_numpy_max_abs_error_vs_native": error,
        "zero_output_rejected_elements": zero_failures,
        "weight_image_manifest": str(image_path.resolve()),
        "weight_image_sha256": image["image_sha256"],
        "checkpoint_parameter": parameter["checkpoint_key"],
        "packed_parameter_sha256": parameter["packed_sha256"],
        "packed_parameter_gpu_address": parameter["gpu_address"],
        "input_offset_bytes": input_offset,
        "output_offset_bytes": output_offset,
        "output_elements": int(expected.size),
        "output_validation": "all_elements_vs_numpy_reference",
        "output_check_tolerance": {"rtol": RTOL, "atol": ATOL},
        "device_elf_built": False,
        "upstream_execution_equivalent": False,
    }
    (target / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return target, record


def run_probe(target: Path, record: dict, timeout: int,
              sim_cycles: int) -> dict:
    result = run_functional(target.parent, ROOT.parent / "generators/radiance/cyclotron",
                            timeout, sim_cycles)
    result.update(
        stage_id=STAGE_ID,
        source_full_elf_sha256=record["source_full_elf_sha256"],
        source_stage_chunk_sha256=record["source_stage_chunk_sha256"],
        fixture_input_sha256=record["fixture_input_sha256"],
        independent_numpy_output_sha256=record["independent_numpy_output_sha256"],
        independent_numpy_max_abs_error_vs_native=
        record["independent_numpy_max_abs_error_vs_native"],
        packed_parameter_sha256=record["packed_parameter_sha256"],
        packed_parameter_gpu_address=record["packed_parameter_gpu_address"],
        output_elements=record["output_elements"],
        stage_output_comparison=(result["status"] == "passed"),
    )
    (target / "functional-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def negative_control(target: Path, record: dict, timeout: int,
                     sim_cycles: int) -> dict:
    negative = target.parent.with_name(target.parent.name + "-negative") / target.name
    shutil.copytree(target, negative, dirs_exist_ok=True)
    data = negative / "probe_data.cpp"
    modified, count = re.subn(r"(v_probe_expected\[48000\] = \{)[^,]+",
                              r"\g<1>1000000.0f", data.read_text(), count=1)
    if count != 1:
        raise ValueError("could not corrupt projection golden for negative control")
    write_if_changed(data, modified)
    build_device(negative)
    negative_manifest = dict(record)
    negative_manifest.update(device_elf_built=True,
                             radiance_elf_sha256=sha256(negative / "kernel.radiance.elf"))
    (negative / "manifest.json").write_text(json.dumps(negative_manifest, indent=2) + "\n")
    result = run_probe(negative, negative_manifest, timeout, sim_cycles)
    if (result["status"] != "failed" or result["tohost"] != 1 or
            result["failure_reason"] != "device_output_mismatch_or_nonfinite" or
            result["device_elf_sha256"] == record["radiance_elf_sha256"]):
        raise ValueError("projection check did not reject a corrupted golden")
    result.update(mutation="first_expected_projection_value_replaced_by_1000000.0f",
                  positive_probe_elf_sha256=record["radiance_elf_sha256"],
                  mutated_probe_data_sha256=sha256(data))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-generated-root", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--sim-cycles", type=int, default=5_000_000_000)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--negative-out", type=Path)
    args = parser.parse_args()
    target, record = generate(args.full_generated_root, args.fixture_root,
                              args.out_root)
    build_device(target)
    record.update(device_elf_built=True,
                  radiance_elf_sha256=sha256(target / "kernel.radiance.elf"))
    (target / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    result = run_probe(target, record, args.timeout, args.sim_cycles)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "passed":
        raise ValueError(f"projection device probe failed: {result['failure_reason']}")
    if args.negative_out:
        negative = negative_control(target, record, args.timeout, args.sim_cycles)
        args.negative_out.parent.mkdir(parents=True, exist_ok=True)
        args.negative_out.write_text(json.dumps(negative, indent=2) + "\n")
        print(f"SmolVLA expert projection negative control: "
              f"rejected at tohost={negative['tohost']}")
    print(f"SmolVLA expert projection: passed, "
          f"{result['cycles_functional']} functional cycles")


if __name__ == "__main__":
    main()
