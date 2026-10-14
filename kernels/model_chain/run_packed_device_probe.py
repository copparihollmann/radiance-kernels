#!/usr/bin/env python3
"""Build and run a small Radiance ISA check for Gemma's packed operators.

This checks the device implementation independently of the multi-gigabyte
checkpoint images. It is a functional check, not a latency measurement.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def packed(values: list[int], bits: int) -> list[int]:
    data = [0] * ((len(values) * bits + 7) // 8)
    for index, signed in enumerate(values):
        value = signed & ((1 << bits) - 1)
        for bit in range(bits):
            position = index * bits + bit
            data[position // 8] |= ((value >> bit) & 1) << (position % 8)
    return data


def initializer(values: list[int]) -> str:
    return ", ".join(str(value) for value in values)


def source() -> str:
    five = packed([-15, 0, 15, 3, -4, 7, 1, -1], 5)
    six = packed([-31, 0, 31, 7], 6)
    table = packed([1] * 64 + [-1] * 64, 5)
    return f"""#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
#include "pipeline_packed.hpp"

extern "C" uint32_t __mu_num_warps = 1;
__global uint8_t weights_five[5] = {{{initializer(five)}}};
__global uint8_t weights_six[3] = {{{initializer(six)}}};
__global uint8_t table[80] = {{{initializer(table)}}};
__global int32_t ids[1] = {{1}};
__global float linear_input[1] = {{2.0f}};
__global float linear_scales[4] = {{1.0f, 0.5f, 2.0f, 3.0f}};
__global float table_scales[2] = {{2.0f, 3.0f}};
__global int32_t decoded_five[8];
__global float embedded[64];
__global float linear_output[4];
__global float tied_output[2];

static void first_stage(void*, uint32_t tid, uint32_t tpb, uint32_t) {{
  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return;
  tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS
      + tid % MU_NUM_THREADS;
  tpb /= MU_NUM_CORES;
  for (uint32_t i = tid; i < 8; i += tpb)
    decoded_five[i] = model_chain::signed_packed_weight(weights_five, i, 5);
  model_chain::embedding_i5_group64(ids, table, table_scales, embedded,
                                     1, 64, 1.0f, tid, tpb);
}}

static void second_stage(void*, uint32_t tid, uint32_t tpb, uint32_t) {{
  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return;
  tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS
      + tid % MU_NUM_THREADS;
  tpb /= MU_NUM_CORES;
  model_chain::linear_i6(linear_input, weights_six, linear_scales,
                          linear_output, 1, 1, 4, tid, tpb);
  model_chain::linear_i5_tied(embedded, table, table_scales, tied_output,
                               1, 64, 2, tid, tpb);
}}

int main() {{
  mu_schedule(first_stage, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES);
  mu_fence();
  mu_schedule(second_stage, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES);
  mu_fence();
  if (mu_hart_id() != 0) {{ for (;;) {{}} }}
  const int32_t expected_five[8] = {{-15, 0, 15, 3, -4, 7, 1, -1}};
  const float expected_six[4] = {{-62.0f, 0.0f, 124.0f, 42.0f}};
  uint32_t errors = 0;
  for (uint32_t i = 0; i < 8; ++i)
    if (decoded_five[i] != expected_five[i]) ++errors;
  for (uint32_t i = 0; i < 64; ++i)
    if (!mu_close(embedded[i], -3.0f, 0.0f, 0.0f)) ++errors;
  for (uint32_t i = 0; i < 4; ++i)
    if (!mu_close(linear_output[i], expected_six[i], 0.0f, 0.0f)) ++errors;
  if (!mu_close(tied_output[0], -384.0f, 0.0f, 0.0f)) ++errors;
  if (!mu_close(tied_output[1], 576.0f, 0.0f, 0.0f)) ++errors;
  mu_tohost(errors ? ((errors << 1) | 1u) : 0u);
  return 0;
}}
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cyclotron-root", type=Path, required=True)
    parser.add_argument("--llvm-muon", type=Path, required=True)
    parser.add_argument("--riscv-toolchain", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    target = args.out_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    simulator_root = args.cyclotron_root.resolve()
    simulator = simulator_root / "target/release/cyclotron"
    config = simulator_root / "config.toml"
    if not simulator.is_file() or not config.is_file():
        parser.error("Cyclotron binary or config is missing")
    llvm_muon = args.llvm_muon.resolve()
    riscv_toolchain = args.riscv_toolchain.resolve()
    if not (llvm_muon / "bin/clang++").is_file() or not (
            riscv_toolchain / "riscv64-unknown-elf/include").is_dir():
        parser.error("Muon LLVM or RISC-V toolchain is missing")
    kernel = target / "kernel.cpp"
    kernel.write_text(source())
    (target / "Makefile").write_text(
        "PROJECT = packed_device_probe\nMU_SRCS = kernel.cpp\n"
        f"RADIANCE_LIB_PATH := {ROOT / 'lib'}\n"
        f"RADIANCE_INCLUDE_PATH := {ROOT / 'lib/include'}\n"
        f"GEMMINI_SW_PATH := {ROOT / 'lib/mxgemmini'}\n"
        f"SOC_DIR := {ROOT / 'soc'}\n"
        f"LLVM_MUON ?= {ROOT / 'llvm/llvm-muon'}\n"
        f"EXTRA_MU_CFLAGS += -I{HERE}\n"
        f"include {ROOT / 'kernels/common.mk'}\n"
    )
    env = os.environ.copy()
    env["LLVM_MUON"] = str(llvm_muon)
    env["RISCV_TOOLCHAIN_PATH"] = str(riscv_toolchain)
    env["RISCV64_TOOLCHAIN_PATH"] = str(riscv_toolchain)
    env["RISCV_SYSROOT"] = str(riscv_toolchain / "riscv64-unknown-elf")
    env["MU_LIBC_INCLUDE"] = str(riscv_toolchain / "riscv64-unknown-elf/include")
    env["MU_USE_LIBC"] = "1"
    env["EXTRA_MU_CFLAGS"] = (
        "-nostdinc++ -isystem " + str(llvm_muon / "include/c++/v1"))
    with (target / "build.log").open("w") as log:
        subprocess.run(["make", "-j1", "kernel.radiance.elf"], cwd=target,
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=args.timeout)
    elf = target / "kernel.radiance.elf"
    process = subprocess.run(
        [str(simulator), str(config), "--binary-path", str(elf),
         "--gen-trace", "false"], cwd=simulator_root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=args.timeout,
    )
    log = target / "functional.log"
    log.write_text(process.stdout)
    cycles = re.findall(r"simulation finished after (\d+) cycles", process.stdout)
    passed = (process.returncode == 0 and len(cycles) == 1
              and "isa-test passed with tohost=0" in process.stdout)
    result = {
        "scope": "small fixed INT5/INT6 device arithmetic and stage dataflow probe",
        "status": "passed" if passed else "failed",
        "checked_operations": ["signed_packed_weight_int5",
                               "embedding_i5_group64", "linear_i6",
                               "linear_i5_tied"],
        "tohost": 0 if passed else None,
        "simulation": "Cyclotron functional ISA model",
        "rtl_execution": False,
        "performance_measurement": False,
        "cycles_functional": int(cycles[0]) if cycles else None,
        "process_exit_code": process.returncode,
        "device_source_sha256": sha256(kernel),
        "device_elf_sha256": sha256(elf),
        "packed_ops_sha256": sha256(HERE / "pipeline_packed.hpp"),
        "simulator_sha256": sha256(simulator),
        "config_sha256": sha256(config),
        "log_sha256": sha256(log),
    }
    (target / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
