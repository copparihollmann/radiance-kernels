#!/usr/bin/env python3
"""Run all ten SmolVLA action updates on Radiance with real native velocities.

The velocities come from the complete, checkpoint-weight generated-C++ run.
Only the Euler stages execute in this probe; the expert networks remain covered
by the full native run and the separate full-ELF simulator attempt.
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


def read_input_noise(input_manifest_path: Path) -> tuple[np.ndarray, dict]:
    manifest = json.loads(input_manifest_path.read_text())
    entries = [item for item in manifest["parameters"]
               if item["logical_name"] == "action.noise"]
    if len(entries) != 1 or entries[0]["size_bytes"] != 1600 * 4:
        raise ValueError("input image has no 1,600-element action noise")
    image_path = input_manifest_path.parent / manifest["image_file"]
    if sha256(image_path) != manifest["image_sha256"]:
        raise ValueError("input image hash mismatch")
    with image_path.open("rb") as stream:
        stream.seek(entries[0]["offset_bytes"])
        raw = stream.read(entries[0]["size_bytes"])
    if hashlib.sha256(raw).hexdigest() != entries[0]["packed_sha256"]:
        raise ValueError("action noise hash mismatch")
    return np.frombuffer(raw, dtype="<f4").copy(), manifest


def generate(full_root: Path, trace_dir: Path, input_manifest_path: Path,
             golden_path: Path, out_root: Path) -> tuple[Path, dict]:
    full = (full_root / "smolvla_base").resolve()
    manifest = json.loads((full / "manifest.json").read_text())
    graph = build("smolvla_base")
    buffers = plan(graph)
    if (manifest["stages"] != len(graph.stages) or
            manifest["radiance_elf_sha256"] != sha256(full / "kernel.radiance.elf") or
            manifest["execution_schedule_sha256"] !=
            buffers["execution_schedule_sha256"]):
        raise ValueError("full SmolVLA ELF differs from the pinned graph")
    native_binary = full / "native/smolvla_native"
    native_result = json.loads((HERE / "evaluation/smolvla-full-fp32-native-results.json")
                               .read_text())
    if (native_result["status"] != "passed" or
            native_result["device_elf_sha256"] != manifest["radiance_elf_sha256"] or
            native_result["native_binary_sha256"] != sha256(native_binary)):
        raise ValueError("native trace source is not the verified full executable")
    trace_log = trace_dir / "full-trace.log"
    trace_text = trace_log.read_text()
    if ("completed 3673/3673: denoise9.euler" not in trace_text or
            "actions=1600 failures=0" not in trace_text):
        raise ValueError("native trace did not finish the complete action chunk")
    noise, input_manifest = read_input_noise(input_manifest_path)
    if input_manifest["image_sha256"] != manifest["input_image_sha256"]:
        raise ValueError("input image differs from full ELF")
    golden = json.loads(golden_path.read_text())
    expected_final = np.asarray(golden["output_values"], dtype="<f4").ravel()
    if (not golden["passed"] or not golden["weights_promoted_to_fp32"] or
            golden["checkpoint_weight_sha256"] != manifest["checkpoint_sha256"] or
            golden["output_sha256"] != manifest["golden_output_sha256"] or
            expected_final.size != 1600 or
            hashlib.sha256(expected_final.tobytes()).hexdigest() !=
            golden["output_sha256"]):
        raise ValueError("upstream FP32 action golden differs from full ELF")

    stages = []
    velocities = []
    expected = []
    velocity_paths = []
    action = noise.copy()
    copied_chunks = {}
    target = (out_root / "smolvla_base").resolve()
    target.mkdir(parents=True, exist_ok=True)
    source_files = [full / name for name in manifest["device_source_files"]
                    if name.startswith("stage_chunk_")]
    for iteration in range(10):
        name = f"denoise{iteration}.euler"
        matches = [(index, stage) for index, stage in enumerate(graph.stages)
                   if stage["id"] == name]
        if len(matches) != 1:
            raise ValueError(f"missing {name}")
        stage_index, stage = matches[0]
        if (stage["op"] != "euler_step" or
                stage["reads"] != ["action.noise" if iteration == 0
                                    else f"denoise{iteration - 1}.euler",
                                    f"denoise{iteration}.action_out"] or
                stage["shape"] != [1, 50, 32]):
            raise ValueError(f"unexpected action carry for {name}")
        velocity_path = trace_dir / f"{stage_index:04d}.bin"
        velocity = np.fromfile(velocity_path, dtype="<f4")
        if velocity.size != 1600 or not np.all(np.isfinite(velocity)):
            raise ValueError(f"bad native velocity fixture: {velocity_path}")
        action = action + np.float32(stage["attrs"]["step_size"]) * velocity
        velocities.append(velocity)
        expected.append(action.copy())
        velocity_paths.append(velocity_path)
        needle = f"void stage_{symbol(name)}("
        chunks = [source for source in source_files if needle in source.read_text()]
        if len(chunks) != 1:
            raise ValueError(f"expected exactly one source chunk for {name}")
        chunk = chunks[0]
        if chunk.name not in copied_chunks:
            shutil.copyfile(chunk, target / chunk.name)
            copied_chunks[chunk.name] = sha256(chunk)
            if sha256(target / chunk.name) != copied_chunks[chunk.name]:
                raise ValueError(f"copied device stage source differs: {chunk.name}")
        stages.append((stage, buffers["allocation"][stage["reads"][0]]["offset_bytes"],
                       buffers["allocation"][stage["reads"][1]]["offset_bytes"],
                       buffers["allocation"][stage["writes"]]["offset_bytes"]))
    if not np.allclose(action, expected_final, rtol=1e-3, atol=1e-3):
        raise ValueError("ten native velocities do not recover upstream FP32 actions")
    zero_failures = [int(np.count_nonzero(
        np.abs(values) > 1e-3 + 1e-3 * np.abs(values))) for values in expected]
    if any(count == 0 for count in zero_failures):
        raise ValueError("an action check cannot distinguish a zero device output")
    penultimate_path = trace_dir / "denoise8-euler.bin"
    if penultimate_path.exists():
        penultimate = np.fromfile(penultimate_path, dtype="<f4")
        if penultimate.size != 1600 or not np.allclose(
                expected[8], penultimate, rtol=1e-3, atol=1e-3):
            raise ValueError("eighth action carry differs from native stage output")

    shutil.copyfile(full / "model_data.cpp", target / "model_data.cpp")
    shutil.copyfile(full / "host.cpp", target / "host.cpp")
    arrays = [f"alignas(64) __global float v_probe_noise[1600] = {{" +
              ", ".join(literal(value) for value in noise) + "};"]
    for iteration in range(10):
        for kind, values in (("velocity", velocities[iteration]),
                             ("expected", expected[iteration])):
            arrays.append(f"alignas(64) __global float v_probe_{kind}_{iteration}[1600] = {{" +
                          ", ".join(literal(value) for value in values) + "};")
    write_if_changed(target / "probe_data.cpp", "#include <mu_intrinsics.h>\n" +
                     "\n".join(arrays) + "\n")
    prototypes = "\n".join(
        f"void stage_{symbol(stage['id'])}(void*, uint32_t, uint32_t, uint32_t);"
        for stage, _, _, _ in stages)
    steps = []
    for iteration, (stage, prior_offset, velocity_offset, output_offset) in enumerate(stages):
        steps.append(f"""
  mu_schedule(load_velocity_{iteration}, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES); mu_fence();
  mu_schedule(stage_{symbol(stage['id'])}, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES); mu_fence();
  if (mu_hart_id() == 0 && failure == 0) {{
    const __global float* output = (const __global float*)(v_arena + {output_offset}u);
    for (uint32_t j = 0; j < 1600u; ++j) {{
      if (!(output[j] == output[j]) ||
          !mu_close(output[j], v_probe_expected_{iteration}[j], 1e-3f, 1e-3f)) {{
        failure = (({iteration + 1}u) << 16) | (j << 1) | 1u;
        break;
      }}
    }}
  }}
  mu_barrier(0, MU_NUM_CORES); mu_fence();""")
        if iteration and prior_offset != stages[iteration - 1][3]:
            raise ValueError(f"{stage['id']} does not read the prior action buffer")
    loaders = []
    for iteration, (_, _, velocity_offset, _) in enumerate(stages):
        loaders.append(f"""
static void load_velocity_{iteration}(void*, uint32_t tid, uint32_t tpb, uint32_t) {{
  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return;
  tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS
      + tid % MU_NUM_THREADS;
  tpb /= MU_NUM_CORES;
  __global float* destination = (__global float*)(v_arena + {velocity_offset}u);
  for (uint32_t j = tid; j < 1600u; j += tpb)
    destination[j] = v_probe_velocity_{iteration}[j];
}}""")
    source = """#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
extern "C" uint32_t __mu_num_warps = 1;
extern __global unsigned char v_arena[];
extern __global float v_probe_noise[];
""" + "\n".join(
        f"extern __global float v_probe_{kind}_{iteration}[];"
        for iteration in range(10) for kind in ("velocity", "expected")) + "\n" + prototypes + "\n" + f"""
static void load_noise(void*, uint32_t tid, uint32_t tpb, uint32_t) {{
  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return;
  tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS
      + tid % MU_NUM_THREADS;
  tpb /= MU_NUM_CORES;
  __global float* destination = (__global float*)(v_arena + {stages[0][1]}u);
  for (uint32_t j = tid; j < 1600u; j += tpb)
    destination[j] = v_probe_noise[j];
}}
""" + "\n".join(loaders) + "\nint main() {\n  uint32_t failure = 0;\n" + """
  mu_schedule(load_noise, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES); mu_fence();
""" + "\n".join(steps) + """
  if (mu_hart_id() != 0) { for (;;) {} }
  mu_tohost(failure); return 0;
}
"""
    write_if_changed(target / "kernel.cpp", source)
    dependencies = ["model_data.cpp", "probe_data.cpp", *sorted(copied_chunks)]
    makefile = ("PROJECT = model_chain\nMU_SRCS = kernel.cpp\nHOST_SRCS = host.cpp\n"
                "MU_SRC_DEPS = " + " ".join(dependencies) + "\n"
                f"RADIANCE_LIB_PATH := {ROOT / 'lib'}\n"
                f"RADIANCE_INCLUDE_PATH := {ROOT / 'lib/include'}\n"
                f"GEMMINI_SW_PATH := {ROOT / 'lib/mxgemmini'}\n"
                f"SOC_DIR := {ROOT / 'soc'}\n"
                f"LLVM_MUON ?= {ROOT / 'llvm/llvm-muon'}\n"
                f"EXTRA_MU_CFLAGS += -I{HERE}\n"
                f"include {ROOT / 'kernels/common.mk'}\n" +
                "".join(f"{name.replace('.cpp', '.mu.o')}: "
                        f"{HERE / 'pipeline_smolvla.hpp'}\n"
                        for name in copied_chunks))
    write_if_changed(target / "Makefile", makefile)
    record = {
        "model": "smolvla_base", "scope": "ten_euler_stages_real_graph_state",
        "stage_ids": [stage["id"] for stage, _, _, _ in stages],
        "stage_indices_one_based": [next(index + 1 for index, candidate
                                         in enumerate(graph.stages)
                                         if candidate["id"] == stage["id"])
                                    for stage, _, _, _ in stages],
        "source_full_elf_sha256": manifest["radiance_elf_sha256"],
        "source_stage_chunks_sha256": copied_chunks,
        "source_native_binary_sha256": native_result["native_binary_sha256"],
        "native_trace_log_sha256": sha256(trace_log),
        "native_velocity_sha256": {path.name: sha256(path) for path in velocity_paths},
        "input_image_sha256": input_manifest["image_sha256"],
        "input_noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
        "upstream_output_sha256": golden["output_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "action_buffer_offsets_bytes": [stages[0][1],
                                         *[entry[3] for entry in stages]],
        "velocity_buffer_offsets_bytes": [entry[2] for entry in stages],
        "output_elements_per_iteration": 1600,
        "fixture_python_final_max_abs_error": float(np.max(
            np.abs(action - expected_final))),
        "zero_output_rejected_elements_per_iteration": zero_failures,
        "output_check_tolerance": {"rtol": 1e-3, "atol": 1e-3},
        "device_elf_built": False, "device_execution": False,
        "upstream_execution_equivalent": False,
    }
    (target / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return target, record


def run_negative_control(target: Path, record: dict, timeout: int,
                         result_path: Path) -> dict:
    negative_target = target.parent.with_name(target.parent.name + "-negative") / target.name
    shutil.copytree(target, negative_target, dirs_exist_ok=True)
    data_path = negative_target / "probe_data.cpp"
    modified, replacements = re.subn(
        r"(v_probe_expected_0\[1600\] = \{)[^,]+",
        r"\g<1>1000000.0f", data_path.read_text(), count=1)
    if replacements != 1:
        raise ValueError("could not mutate the first expected action")
    write_if_changed(data_path, modified)
    build_device(negative_target)
    negative_record = dict(record)
    negative_record.update(
        mutation="first_expected_action_replaced_by_1000000.0f",
        mutated_probe_data_sha256=sha256(data_path),
        positive_probe_elf_sha256=record["device_elf_sha256"])
    result = run(negative_target.resolve(), negative_record, timeout)
    if (result["status"] != "failed" or result["tohost"] != 65537 or
            result["failure_reason"] != "device_output_mismatch" or
            result["device_elf_sha256"] == record["device_elf_sha256"]):
        raise ValueError("device comparison did not reject the injected error")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-generated-root", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--golden-output", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--negative-out", type=Path)
    args = parser.parse_args()
    target, record = generate(args.full_generated_root, args.trace_dir,
                              args.input_image, args.golden_output, args.out_root)
    build_device(target)
    record.update(device_elf_built=True,
                  device_elf_sha256=sha256(target / "kernel.radiance.elf"))
    result = run(target, record, args.timeout)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.negative_out:
        negative = run_negative_control(target, record, args.timeout,
                                        args.negative_out)
        print(f"SmolVLA Euler chain negative control: rejected at "
              f"tohost={negative['tohost']}")
    print(f"SmolVLA Euler chain: {result['status']}, "
          f"{result['cycles_functional']} functional cycles")


if __name__ == "__main__":
    main()
