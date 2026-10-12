#!/usr/bin/env python3
"""Check the final SmolVLA Euler stage on Radiance with real graph state.

The two input tensors come from a complete generated-C++ action run with the
pinned checkpoint and exact policy input. This probe executes the unchanged
last stage from that ELF and checks every action against the FP32 upstream
policy. It supplements, but does not replace, the full device run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from compile_decoder import build_device, literal, symbol, write_if_changed
from plan_buffers import plan


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernels/evaluation/llm"))
from stitch import build  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_values(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size != 1600 or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: expected 1,600 finite FP32 action values")
    return values


def generate(full_root: Path, previous: Path, velocity: Path,
             previous_log: Path, trace_log: Path,
             golden_path: Path, out_root: Path) -> tuple[Path, dict]:
    full = (full_root / "smolvla_base").resolve()
    manifest = json.loads((full / "manifest.json").read_text())
    graph = build("smolvla_base")
    buffers = plan(graph)
    final = graph.stages[-1]
    if (manifest["stages"] != len(graph.stages) or
            manifest["radiance_elf_sha256"] != sha256(full / "kernel.radiance.elf") or
            manifest["execution_schedule_sha256"] !=
            buffers["execution_schedule_sha256"] or
            final["id"] != "denoise9.euler" or
            final["op"] != "euler_step" or
            final["reads"] != ["denoise8.euler", "denoise9.action_out"]):
        raise ValueError("full SmolVLA ELF no longer has the pinned final action stage")
    previous_values = load_values(previous)
    velocity_values = load_values(velocity)
    previous_text = previous_log.read_text()
    trace_text = trace_log.read_text()
    if ("completed 3397/3673: denoise8.euler" not in previous_text or
            "completed 3673/3673: denoise9.euler" not in trace_text or
            "actions=1600 failures=0" not in trace_text or
            velocity.name != "3672.bin"):
        raise ValueError("probe tensors lack complete native-run provenance")
    native_binary = full / "native/smolvla_native"
    native_result = json.loads((HERE / "evaluation/smolvla-full-fp32-native-results.json")
                               .read_text())
    if (native_result["status"] != "passed" or
            native_result["device_elf_sha256"] != manifest["radiance_elf_sha256"] or
            native_result["native_binary_sha256"] != sha256(native_binary)):
        raise ValueError("probe state comes from an unverified native executable")
    golden = json.loads(golden_path.read_text())
    expected = np.asarray(golden["output_values"], dtype="<f4").ravel()
    if (not golden["passed"] or not golden["weights_promoted_to_fp32"] or
            golden["checkpoint_weight_sha256"] != manifest["checkpoint_sha256"] or
            golden["output_sha256"] != manifest["golden_output_sha256"] or
            hashlib.sha256(expected.tobytes()).hexdigest() != golden["output_sha256"] or
            expected.size != 1600):
        raise ValueError("upstream action golden differs from the full ELF")
    fixture_output = previous_values + np.float32(final["attrs"]["step_size"]) * velocity_values
    fixture_error = float(np.max(np.abs(fixture_output - expected)))
    if not np.allclose(fixture_output, expected, rtol=1e-3, atol=1e-3):
        raise ValueError("native final-stage inputs do not produce the upstream action")
    zero_failures = int(np.count_nonzero(
        np.abs(expected) > 1e-3 + 1e-3 * np.abs(expected)))
    if zero_failures == 0:
        raise ValueError("action golden cannot distinguish a zero device output")
    target = (out_root / "smolvla_base").resolve()
    target.mkdir(parents=True, exist_ok=True)
    source_chunk = full / "stage_chunk_091.cpp"
    chunk = target / source_chunk.name
    shutil.copyfile(source_chunk, chunk)
    shutil.copyfile(full / "model_data.cpp", target / "model_data.cpp")
    shutil.copyfile(full / "host.cpp", target / "host.cpp")
    data = "#include <mu_intrinsics.h>\n#include <stdint.h>\n"
    for name, values in (("v_probe_previous", previous_values),
                         ("v_probe_velocity", velocity_values),
                         ("v_probe_golden", expected)):
        data += (f"alignas(64) __global float {name}[1600] = {{" +
                 ", ".join(literal(value) for value in values) + "};\n")
    write_if_changed(target / "probe_data.cpp", data)
    offsets = [buffers["allocation"][name]["offset_bytes"]
               for name in [*final["reads"], final["writes"]]]
    kernel = """#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
extern "C" uint32_t __mu_num_warps = 1;
extern __global unsigned char v_arena[];
extern __global float v_probe_previous[];
extern __global float v_probe_velocity[];
extern __global float v_probe_golden[];
void stage_""" + symbol(final["id"]) + "(void*, uint32_t, uint32_t, uint32_t);\n" + f"""
static void init_inputs(void*, uint32_t tid, uint32_t tpb, uint32_t) {{
  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return;
  tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS
      + tid % MU_NUM_THREADS;
  tpb /= MU_NUM_CORES;
  __global float* previous = (__global float*)(v_arena + {offsets[0]}u);
  __global float* velocity = (__global float*)(v_arena + {offsets[1]}u);
  for (uint32_t j = tid; j < 1600u; j += tpb) {{
    previous[j] = v_probe_previous[j];
    velocity[j] = v_probe_velocity[j];
  }}
}}
int main() {{
  mu_schedule(init_inputs, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES); mu_fence();
  mu_schedule(stage_{symbol(final['id'])}, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES); mu_fence();
  if (mu_hart_id() != 0) {{ for (;;) {{}} }}
  const __global float* output = (const __global float*)(v_arena + {offsets[2]}u);
  for (uint32_t j = 0; j < 1600u; ++j) {{
    if (!(output[j] == output[j]) ||
        !mu_close(output[j], v_probe_golden[j], 1e-3f, 1e-3f)) {{
      mu_tohost((j << 1) | 1u); return 0;
    }}
  }}
  mu_tohost(0u); return 0;
}}
"""
    write_if_changed(target / "kernel.cpp", kernel)
    makefile = ("PROJECT = model_chain\nMU_SRCS = kernel.cpp\nHOST_SRCS = host.cpp\n"
                "MU_SRC_DEPS = model_data.cpp probe_data.cpp stage_chunk_091.cpp\n"
                f"RADIANCE_LIB_PATH := {ROOT / 'lib'}\n"
                f"RADIANCE_INCLUDE_PATH := {ROOT / 'lib/include'}\n"
                f"GEMMINI_SW_PATH := {ROOT / 'lib/mxgemmini'}\n"
                f"SOC_DIR := {ROOT / 'soc'}\n"
                f"LLVM_MUON ?= {ROOT / 'llvm/llvm-muon'}\n"
                f"EXTRA_MU_CFLAGS += -I{HERE}\n"
                f"include {ROOT / 'kernels/common.mk'}\n"
                f"stage_chunk_091.mu.o: {HERE / 'pipeline_smolvla.hpp'}\n")
    write_if_changed(target / "Makefile", makefile)
    record = {
        "model": "smolvla_base", "scope": "final_euler_stage_real_graph_state",
        "stage_id": final["id"], "stage_index_zero_based": len(graph.stages) - 1,
        "source_full_elf_sha256": manifest["radiance_elf_sha256"],
        "source_stage_chunk_sha256": sha256(source_chunk),
        "probe_stage_chunk_sha256": sha256(chunk),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "upstream_output_sha256": golden["output_sha256"],
        "previous_action_sha256": sha256(previous),
        "velocity_sha256": sha256(velocity),
        "previous_run_log_sha256": sha256(previous_log),
        "full_trace_log_sha256": sha256(trace_log),
        "source_native_binary_sha256": native_result["native_binary_sha256"],
        "input_offsets_bytes": offsets[:2], "output_offset_bytes": offsets[2],
        "output_elements": 1600, "output_check_tolerance": {
            "rtol": 1e-3, "atol": 1e-3},
        "fixture_python_max_abs_error": fixture_error,
        "zero_output_rejected_elements": zero_failures,
        "device_elf_built": False, "device_execution": False,
        "upstream_execution_equivalent": False,
    }
    (target / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return target, record


def run(target: Path, record: dict, timeout: int) -> dict:
    simulator_root = (ROOT.parent / "generators/radiance/cyclotron").resolve()
    simulator = simulator_root / "target/release/cyclotron"
    config = target / "cyclotron-config.toml"
    source = (simulator_root / "config.toml").read_text()
    modified, matched = re.subn(r"(?m)^timeout\s*=\s*\d+", "timeout = 100000000", source, count=1)
    if matched != 1:
        raise ValueError("Cyclotron config has no unique cycle limit")
    modified = modified.replace('"config/timing/',
                                f'"{simulator_root}/config/timing/')
    config.write_text(modified)
    env = os.environ.copy()
    env["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
    try:
        process = subprocess.run(
            [str(simulator), str(config), "--binary-path",
             str(target / "kernel.radiance.elf"), "--gen-trace", "false"],
            cwd=simulator_root, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=timeout)
        output = process.stdout
        exit_code = process.returncode
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        exit_code = 124
    log = target / "functional.log"
    log.write_text(output)
    cycles = re.findall(r"simulation finished after (\d+) cycles", output)
    passed = (exit_code == 0 and len(cycles) == 1 and
              "isa-test passed with tohost=0" in output)
    failed = re.search(r"isa-test failed with tohost=(\d+)", output)
    result = dict(record)
    result.update(status="passed" if passed else "failed",
                  device_elf_built=True, device_execution=passed,
                  device_elf_sha256=sha256(target / "kernel.radiance.elf"),
                  cycles_functional=int(cycles[0]) if cycles else None,
                  tohost=0 if passed else int(failed.group(1)) if failed else None,
                  process_exit_code=exit_code,
                  failure_reason=(None if passed else "wall_clock_timeout"
                                  if exit_code == 124 else "device_output_mismatch"
                                  if failed else "device_or_simulator_failure"),
                  simulator_sha256=sha256(simulator),
                  config_sha256=sha256(config), log_sha256=sha256(log),
                  rtl_execution=False, performance_measurement=False)
    (target / "functional-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-generated-root", type=Path, required=True)
    parser.add_argument("--previous-action", type=Path, required=True)
    parser.add_argument("--velocity", type=Path, required=True)
    parser.add_argument("--previous-log", type=Path, required=True)
    parser.add_argument("--trace-log", type=Path, required=True)
    parser.add_argument("--golden-output", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    target, record = generate(args.full_generated_root, args.previous_action,
                              args.velocity, args.previous_log, args.trace_log,
                              args.golden_output, args.out_root)
    build_device(target)
    record.update(device_elf_built=True,
                  device_elf_sha256=sha256(target / "kernel.radiance.elf"))
    result = run(target, record, args.timeout)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"SmolVLA {record['stage_id']}: {result['status']}, "
          f"{result['cycles_functional']} functional cycles")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
