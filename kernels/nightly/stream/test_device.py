#!/usr/bin/env python3
"""Run STREAM's shared device body in Cyclotron with one or two logical partitions.

The two-partition control executes the partitions sequentially on one simulated
cluster. It checks indexing and arithmetic, not concurrent two-SM timing.
"""

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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "kernels/stream"))
from run import result_value  # noqa: E402

CONTROL_N = 4102  # multiple full grid strides plus an uneven tail


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_case(kind, partitions, out, cyclotron_root, toolchain, llvm_muon,
             simulator=None, base_config=None, gmem_transport="environment"):
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(HERE / "gen_data.py"),
                    "--kind", kind, "--n", str(CONTROL_N)], cwd=out, check=True)
    header = (out / "stream_data.h").read_text()
    output = {"copy": "C", "scale": "B", "add": "C", "triad": "A"}[kind]
    match = re.search(rf"#define STREAM_{output}_ADDR 0x([0-9a-f]+)u", header)
    if not match:
        raise ValueError("generated header lacks output address")
    output_address = int(match.group(1), 16)
    shutil.copy2(HERE / "stream_ops.hpp", out / "stream_ops.hpp")
    (out / "kernel.cpp").write_text(f'''#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
#include "stream_ops.hpp"
extern "C" uint32_t __mu_num_warps = 4;
static void stage(void *, uint32_t tid, uint32_t tpb, uint32_t) {{
  stream_apply<{partitions}>(tid, tpb, 0);
  {'stream_apply<2>(tid, tpb, 1);' if partitions == 2 else ''}
}}
int main() {{
  mu_schedule(stage, nullptr, 4);
  mu_barrier(0, MU_NUM_CORES);
  mu_fence();
  if (mu_hart_id() != 0) {{ for (;;) {{}} }}
  mu_tohost(0u); return 0;
}}
''')
    (out / "Makefile").write_text(f'''PROJECT = stream
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp
RADIANCE_LIB_PATH := {ROOT / 'lib'}
RADIANCE_INCLUDE_PATH := $(RADIANCE_LIB_PATH)/include
GEMMINI_SW_PATH := $(RADIANCE_LIB_PATH)/mxgemmini
SOC_DIR := {ROOT / 'soc'}
LLVM_MUON := {llvm_muon}
EXTRA_MU_CFLAGS += -I{ROOT / 'kernels/model_chain'}
include {ROOT / 'kernels/common.mk'}
kernel.mu.o: stream_data.h stream_ops.hpp
''')
    env = os.environ.copy()
    env["RISCV_TOOLCHAIN_PATH"] = str(toolchain)
    env["RISCV64_TOOLCHAIN_PATH"] = str(toolchain)
    env["RISCV_SYSROOT"] = str(toolchain / "riscv64-unknown-elf")
    env["MU_LIBC_INCLUDE"] = str(toolchain / "riscv64-unknown-elf/include")
    env["EXTRA_MU_CFLAGS"] = (
        "-nostdinc++ -isystem " + str(llvm_muon / "include/c++/v1"))
    build = subprocess.run(["make", "-j1", "kernel.radiance.elf"], cwd=out,
                           env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    (out / "build.log").write_text(
        f"make exit={build.returncode}; retained final 65536 characters\n"
        + build.stdout[-65536:])
    if build.returncode:
        raise RuntimeError(f"device build failed: {out / 'build.log'}")

    config = (base_config or cyclotron_root / "config.toml").read_text()
    config, count = re.subn(r"(?m)^timeout\s*=\s*\d+", "timeout = 10000000",
                            config, count=1)
    if count != 1:
        raise ValueError("Cyclotron config lacks unique timeout")
    config = config.replace('"config/timing/', f'"{cyclotron_root}/config/timing/')
    config = config.replace('"timing/', f'"{cyclotron_root}/config/timing/')
    if gmem_transport == "config":
        loads = [dict(address=region * 0x20000000,
                      bytes=(out / f"data/stream.region{region}.bin").stat().st_size,
                      path=str(out / f"data/stream.region{region}.bin")) for region in (1, 2, 3)]
        dumps = [dict(address=output_address, bytes=CONTROL_N * 4, path=str(out / "output.bin"))]
        def render(items):
            return "[" + ", ".join("{address="+str(r["address"])+",bytes="+str(r["bytes"])+",path="+json.dumps(r["path"])+"}" for r in items) + "]"
        config = config.replace("[sim]", "[sim]\ngmem_load="+render(loads)+"\ngmem_dump="+render(dumps), 1)
    (out / "cyclotron-config.toml").write_text(config)
    env["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
    env["CYCLOTRON_WEIGHTS"] = ",".join(
        f"0x{region * 0x20000000:x}:{out / f'data/stream.region{region}.bin'}"
        for region in (1, 2, 3))
    env["CYCLOTRON_DUMP_GMEM"] = (
        f"0x{output_address:x}:{CONTROL_N * 4}:{out / 'output.bin'}")
    simulator = simulator or cyclotron_root / "target/release/cyclotron"
    run = subprocess.run([str(simulator), str(out / "cyclotron-config.toml"),
                          "--binary-path", str(out / "kernel.radiance.elf"),
                          "--gen-trace", "false"], cwd=cyclotron_root,
                         env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, timeout=120)
    (out / "cyclotron.log").write_text(run.stdout)
    if run.returncode or "isa-test passed with tohost=0" not in run.stdout:
        raise RuntimeError(f"device execution failed: {out / 'cyclotron.log'}")
    actual = np.fromfile(out / "output.bin", dtype="<f4")
    expected = np.fromiter((result_value(kind, i) for i in range(CONTROL_N)),
                           dtype="<f4", count=CONTROL_N)
    bad = (int(np.count_nonzero(actual != expected)) if actual.size == CONTROL_N
           else CONTROL_N)
    cycles = re.findall(r"simulation finished after (\d+) cycles", run.stdout)
    record = {
        "operation": kind, "logical_partitions": partitions,
        "elements": CONTROL_N, "status": "passed" if bad == 0 else "failed",
        "out_of_tolerance": bad, "exact_fp32": bad == 0,
        "cycles_functional": int(cycles[0]) if len(cycles) == 1 else None,
        "elf_sha256": sha256(out / "kernel.radiance.elf"),
        "stream_ops_sha256": sha256(HERE / "stream_ops.hpp"),
        "data_receipt_sha256": sha256(out / "data/stream.json"),
        "data_blob_sha256": {str(region): sha256(
            out / f"data/stream.region{region}.bin") for region in (1, 2, 3)},
        "config_sha256": sha256(out / "cyclotron-config.toml"),
        "log_sha256": sha256(out / "cyclotron.log"),
        "output_sha256": sha256(out / "output.bin"),
        "reference_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        "simulator_sha256": sha256(simulator),
        "rtl_execution": False, "performance_measurement": False,
        "two_sm_execution": False,
        "gmem_transport": gmem_transport,
    }
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    if bad:
        raise AssertionError(f"{kind}/{partitions}: {bad} FP32 mismatches")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cyclotron-root", type=Path, required=True)
    parser.add_argument("--toolchain", type=Path, required=True)
    parser.add_argument("--llvm-muon", type=Path, required=True)
    parser.add_argument("--simulator", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--gmem-transport", choices=("environment", "config"), default="environment")
    args = parser.parse_args()
    results = []
    for kind in ("copy", "scale", "add", "triad"):
        for partitions in (1, 2):
            result = one_case(kind, partitions, args.out.resolve() / kind /
                              str(partitions), args.cyclotron_root.resolve(),
                              args.toolchain.resolve(), args.llvm_muon.resolve(),
                              args.simulator.resolve() if args.simulator else None,
                              args.base_config.resolve() if args.base_config else None,
                              args.gmem_transport)
            results.append(result)
            print(kind, partitions, result["status"], result["cycles_functional"],
                  flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
