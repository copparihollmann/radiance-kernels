#!/usr/bin/env python3
"""Cyclotron check of the shared Spatter body, including two logical partitions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SMOKE = ROOT / "kernels/spatter/smoke.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_case(case_id, partitions, out, suite, cyclotron_root, toolchain, llvm_muon,
             timeout, sim_cycles, collision_policy, max_layout_mib,
             simulator=None, base_config=None, gmem_transport="environment"):
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(HERE / "gen_data.py"),
               "--suite", str(suite), "--case", str(case_id),
               "--max-layout-mib", str(max_layout_mib)]
    if collision_policy:
        command += ["--collision-policy", collision_policy]
    elif suite == SMOKE and case_id == 5:
        command += ["--collision-policy", "ordered"]
    subprocess.run(command, cwd=out, check=True)
    data = json.loads((out / "data/spatter.json").read_text())
    header = (out / "spatter_data.h").read_text()
    def address(name):
        match = re.search(rf"#define {name} 0x([0-9a-f]+)u", header)
        if not match:
            raise ValueError(f"generated header lacks {name}")
        return int(match.group(1), 16)
    output_address = address("SPATTER_OUT_ADDR")
    golden_address = address("SPATTER_GOLD_ADDR")
    output_bytes = data["output_elements"] * 8
    shutil.copy2(HERE / "spatter_body.hpp", out / "spatter_body.hpp")
    (out / "kernel.cpp").write_text(f'''#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
#include "spatter_body.hpp"
extern "C" uint32_t __mu_num_warps = 4;
static void stage(void *, uint32_t tid, uint32_t tpb, uint32_t) {{
  spatter_apply<{partitions}>(tid, tpb, 0);
  {'spatter_apply<2>(tid, tpb, 1);' if partitions == 2 else ''}
}}
int main() {{
  mu_schedule(stage, nullptr, 4);
  mu_barrier(0, MU_NUM_CORES);
  mu_fence();
  if (mu_hart_id() != 0) {{ for (;;) {{}} }}
  mu_tohost(0u); return 0;
}}
''')
    (out / "Makefile").write_text(f'''PROJECT = spatter
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp
RADIANCE_LIB_PATH := {ROOT / 'lib'}
RADIANCE_INCLUDE_PATH := $(RADIANCE_LIB_PATH)/include
GEMMINI_SW_PATH := $(RADIANCE_LIB_PATH)/mxgemmini
SOC_DIR := {ROOT / 'soc'}
LLVM_MUON := {llvm_muon}
EXTRA_MU_CFLAGS += -I{ROOT / 'kernels/model_chain'} -I{ROOT / 'kernels/spatter'}
include {ROOT / 'kernels/common.mk'}
kernel.mu.o: spatter_data.h spatter_body.hpp
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
    config, count = re.subn(r"(?m)^timeout\s*=\s*\d+", f"timeout = {sim_cycles}",
                            config, count=1)
    if count != 1:
        raise ValueError("Cyclotron config lacks unique timeout")
    config = config.replace('"config/timing/', f'"{cyclotron_root}/config/timing/')
    config = config.replace('"timing/', f'"{cyclotron_root}/config/timing/')
    if gmem_transport == "config":
        loads = [dict(address=region * 0x20000000,
                      bytes=(out / f"data/spatter.region{region}.bin").stat().st_size,
                      path=str(out / f"data/spatter.region{region}.bin")) for region in (1, 2, 3)]
        dumps = [dict(address=output_address, bytes=output_bytes, path=str(out / "output.bin"))]
        def render(items):
            return "[" + ", ".join("{address="+str(r["address"])+",bytes="+str(r["bytes"])+",path="+json.dumps(r["path"])+"}" for r in items) + "]"
        config = config.replace("[sim]", "[sim]\ngmem_load="+render(loads)+"\ngmem_dump="+render(dumps), 1)
    (out / "cyclotron-config.toml").write_text(config)
    env["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
    env["CYCLOTRON_WEIGHTS"] = ",".join(
        f"0x{region * 0x20000000:x}:{out / f'data/spatter.region{region}.bin'}"
        for region in (1, 2, 3))
    env["CYCLOTRON_DUMP_GMEM"] = (
        f"0x{output_address:x}:{output_bytes}:{out / 'output.bin'}")
    simulator = simulator or cyclotron_root / "target/release/cyclotron"
    run = subprocess.run([str(simulator), str(out / "cyclotron-config.toml"),
                          "--binary-path", str(out / "kernel.radiance.elf"),
                          "--gen-trace", "false"], cwd=cyclotron_root,
                         env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, timeout=timeout)
    (out / "cyclotron.log").write_text(run.stdout)
    if run.returncode or "isa-test passed with tohost=0" not in run.stdout:
        raise RuntimeError(f"device execution failed: {out / 'cyclotron.log'}")
    with (out / "data/spatter.region3.bin").open("rb") as stream:
        stream.seek(golden_address - 0x60000000)
        expected = stream.read(output_bytes)
    actual = (out / "output.bin").read_bytes()
    if len(actual) != output_bytes or actual != expected:
        raise AssertionError(f"{case_id}/{partitions}: device output differs from serial golden")
    if hashlib.sha256(expected).hexdigest() != data["golden_sha256"]:
        raise AssertionError("golden bytes differ from generated receipt")
    cycles = re.findall(r"simulation finished after (\d+) cycles", run.stdout)
    record = {
        "case": case_id, "kind": data["kind"],
        "collision_policy": data["collision_policy"],
        "logical_partitions": partitions,
        "output_elements": data["output_elements"],
        "status": "passed", "exact_output_bytes": output_bytes,
        "cycles_functional": int(cycles[0]) if len(cycles) == 1 else None,
        "elf_sha256": sha256(out / "kernel.radiance.elf"),
        "spatter_body_sha256": sha256(HERE / "spatter_body.hpp"),
        "data_receipt_sha256": sha256(out / "data/spatter.json"),
        "config_sha256": sha256(out / "cyclotron-config.toml"),
        "log_sha256": sha256(out / "cyclotron.log"),
        "output_sha256": sha256(out / "output.bin"),
        "reference_sha256": data["golden_sha256"],
        "simulator_sha256": sha256(simulator),
        "rtl_execution": False, "performance_measurement": False,
        "two_sm_execution": False,
        "gmem_transport": gmem_transport,
    }
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=SMOKE)
    parser.add_argument("--case", type=int, action="append")
    parser.add_argument("--partitions", type=int, nargs="+", choices=(1, 2),
                        default=(1, 2))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sim-cycles", type=int, default=10_000_000)
    parser.add_argument("--collision-policy", choices=("parallel", "ordered"))
    parser.add_argument("--max-layout-mib", type=int, default=128)
    parser.add_argument("--cyclotron-root", type=Path, required=True)
    parser.add_argument("--toolchain", type=Path, required=True)
    parser.add_argument("--llvm-muon", type=Path, required=True)
    parser.add_argument("--simulator", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--gmem-transport", choices=("environment", "config"), default="environment")
    args = parser.parse_args()
    suite = args.suite.resolve()
    case_ids = (args.case if args.case is not None else
                range(len(json.loads(suite.read_text()))))
    results = []
    for case_id in case_ids:
        for partitions in args.partitions:
            result = one_case(case_id, partitions, args.out.resolve() /
                              f"case-{case_id}" / str(partitions),
                              suite, args.cyclotron_root.resolve(),
                              args.toolchain.resolve(), args.llvm_muon.resolve(),
                              args.timeout, args.sim_cycles,
                              args.collision_policy, args.max_layout_mib,
                              args.simulator.resolve() if args.simulator else None,
                              args.base_config.resolve() if args.base_config else None,
                              args.gmem_transport)
            results.append(result)
            print(case_id, partitions, result["status"],
                  result["cycles_functional"], flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
