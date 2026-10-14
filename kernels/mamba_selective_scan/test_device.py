#!/usr/bin/env python3
"""Build and check the Mamba2 recurrence on Cyclotron in an external build dir."""

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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_TOOLCHAIN = Path("/scratch/agustin/projects/chipyard/.conda-env/riscv-tools")
DEFAULT_LLVM = Path("/scratch/agustin/projects/chipyard/radiance-kernels-firesim/llvm/llvm-muon")
DEFAULT_CYCLOTRON = Path("/scratch/agustin/projects/chipyard/generators/radiance/cyclotron")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(args: argparse.Namespace) -> dict:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("kernel.cpp", "gen_data.py"):
        shutil.copy2(HERE / name, out / name)
    receipt = out / "fixture.json"
    subprocess.run([sys.executable, str(HERE / "gen_data.py"),
                    "--out", str(out / "data"), "--receipt", str(receipt)],
                   check=True)
    (out / "Makefile").write_text(f"""PROJECT = kernel
MU_SRCS = kernel.cpp
RADIANCE_LIB_PATH := {ROOT / 'lib'}
RADIANCE_INCLUDE_PATH := $(RADIANCE_LIB_PATH)/include
GEMMINI_SW_PATH := $(RADIANCE_LIB_PATH)/mxgemmini
SOC_DIR := {ROOT / 'soc'}
LLVM_MUON := {args.llvm_muon}
include {ROOT / 'kernels/common.mk'}
kernel.mu.o: data
""")
    env = os.environ.copy()
    env.update({
        "RISCV_TOOLCHAIN_PATH": str(args.toolchain),
        "RISCV64_TOOLCHAIN_PATH": str(args.toolchain),
        "RISCV_SYSROOT": str(args.toolchain / "riscv64-unknown-elf"),
        "MU_LIBC_INCLUDE": str(args.toolchain / "riscv64-unknown-elf/include"),
        "LLVM_MUON": str(args.llvm_muon),
        "MU_STACK_WORD_STRIDE": "1",
        "MU_USE_LIBC": "1",
        "EXTRA_MU_CFLAGS": ("-nostdinc++ -isystem "
                            + str(args.llvm_muon / "include/c++/v1")),
        "RADIANCE_DISABLE_CYCLOTRON_TRACE": "1",
    })
    build_log = out / "build.log"
    with build_log.open("w") as stream:
        build = subprocess.run(["make", "-j1", "kernel.radiance.elf"],
                               cwd=out, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=False)
    if build.returncode:
        raise RuntimeError(f"build failed ({build.returncode}); see {build_log}")

    config = (args.cyclotron / "config.toml").read_text()
    config, count = re.subn(r"(?m)^timeout\s*=\s*\d+",
                            f"timeout = {args.max_cycles}", config, count=1)
    if count != 1:
        raise ValueError("Cyclotron config has no unique simulation timeout")
    config = config.replace('"config/timing/',
                            f'"{args.cyclotron}/config/timing/')
    config_path = out / "cyclotron-config.toml"
    config_path.write_text(config)
    simulator = args.cyclotron / "target/release/cyclotron"
    elf = out / "kernel.radiance.elf"
    nm = subprocess.check_output([str(args.llvm_muon / "bin/llvm-nm"),
                                  "-n", str(elf)], text=True)
    symbols = {name: int(address, 16) for address, name in re.findall(
        r"(?m)^([0-9a-f]+) [BD] (\w+)$", nm)}
    required = ("y_raw", "state_raw", "gold_y_raw", "gold_state_raw")
    if any(name not in symbols for name in required):
        raise ValueError("device ELF lacks output or golden symbols")
    fixture = json.loads(receipt.read_text())
    y_bytes = fixture["tokens"] * fixture["heads"] * fixture["head_dim"] * 4
    state_bytes = fixture["heads"] * fixture["head_dim"] * fixture["state_size"] * 4
    base = min(symbols[name] for name in required)
    end = max(symbols[name] + (y_bytes if "y_raw" in name else state_bytes)
              for name in required)
    memory_path = out / "memory.bin"
    env["CYCLOTRON_DUMP_GMEM"] = f"0x{base:x}:{end - base}:{memory_path}"
    log = out / "cyclotron.log"
    with log.open("w") as stream:
        result = subprocess.run([str(simulator), str(config_path),
                                 "--binary-path", str(elf),
                                 "--gen-trace", "false"],
                                cwd=args.cyclotron, env=env,
                                stdout=stream, stderr=subprocess.STDOUT,
                                timeout=args.wall_timeout, check=False)
    output = log.read_text(errors="replace")
    pass_markers = re.findall(r"isa-test passed with tohost=0", output)
    cycle_matches = re.findall(r"simulation finished after (\d+) cycles", output)
    if result.returncode or len(pass_markers) != 1 or len(cycle_matches) != 1:
        raise RuntimeError(f"Cyclotron check failed ({result.returncode}); see {log}")
    memory = memory_path.read_bytes()
    if len(memory) != end - base:
        raise AssertionError("Cyclotron did not dump the requested memory range")

    def field(name: str, length: int) -> bytes:
        offset = symbols[name] - base
        return memory[offset:offset + length]

    y, gold_y = field("y_raw", y_bytes), field("gold_y_raw", y_bytes)
    state, gold_state = (field("state_raw", state_bytes),
                         field("gold_state_raw", state_bytes))
    if hashlib.sha256(gold_y).hexdigest() != fixture["gold_y_sha256"]:
        raise AssertionError("device golden output differs from NumPy golden")
    if hashlib.sha256(gold_state).hexdigest() != fixture["gold_final_state_sha256"]:
        raise AssertionError("device golden state differs from NumPy golden")
    if not np.allclose(np.frombuffer(y, dtype="<f4"),
                       np.frombuffer(gold_y, dtype="<f4"),
                       rtol=3e-5, atol=3e-6):
        raise AssertionError("dumped output differs from NumPy golden")
    if not np.allclose(np.frombuffer(state, dtype="<f4"),
                       np.frombuffer(gold_state, dtype="<f4"),
                       rtol=3e-5, atol=3e-6):
        raise AssertionError("dumped final state differs from NumPy golden")
    report = {
        "schema": "radiance.mamba-selective-scan-device-check.v1",
        "status": "passed", "backend": "cyclotron", "rtl_execution": False,
        "performance_measurement": False,
        "functional_cycles": int(cycle_matches[0]),
        "checked_output_elements": y_bytes // 4,
        "checked_final_state_elements": state_bytes // 4,
        "fixture": fixture,
        "output_sha256": hashlib.sha256(y).hexdigest(),
        "final_state_sha256": hashlib.sha256(state).hexdigest(),
        "kernel_source_sha256": digest(HERE / "kernel.cpp"),
        "generator_source_sha256": digest(HERE / "gen_data.py"),
        "test_source_sha256": digest(HERE / "test_device.py"),
        "elf_sha256": digest(elf),
        "config_sha256": digest(config_path),
        "simulator_sha256": digest(simulator),
        "build_log_sha256": digest(build_log),
        "cyclotron_log_sha256": digest(log),
    }
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--toolchain", type=Path, default=DEFAULT_TOOLCHAIN)
    parser.add_argument("--llvm-muon", type=Path, default=DEFAULT_LLVM)
    parser.add_argument("--cyclotron", type=Path, default=DEFAULT_CYCLOTRON)
    parser.add_argument("--max-cycles", type=int, default=20_000_000)
    parser.add_argument("--wall-timeout", type=int, default=300)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"],
                      "functional_cycles": report["functional_cycles"],
                      "result": str(args.out.resolve() / "result.json")}, indent=2))


if __name__ == "__main__":
    main()
