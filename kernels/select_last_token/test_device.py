#!/usr/bin/env python3
"""Build and verify final-token selection in Cyclotron."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from gen_data import generate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOLCHAIN = Path("/scratch/agustin/projects/chipyard/.conda-env/riscv-tools")
LLVM = Path("/scratch/agustin/projects/chipyard/radiance-kernels-firesim/llvm/llvm-muon")
CYCLOTRON = Path("/scratch/agustin/projects/chipyard/generators/radiance/cyclotron")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(out: Path, source: str, build_dir: Path | None,
        max_cycles: int, wall_timeout: int) -> dict:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fixture = generate(out / "data", source, build_dir)
    (out / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    shutil.copy2(HERE / "kernel.cpp", out / "kernel.cpp")
    (out / "Makefile").write_text(f"""PROJECT = kernel
MU_SRCS = kernel.cpp
RADIANCE_LIB_PATH := {ROOT / 'lib'}
RADIANCE_INCLUDE_PATH := $(RADIANCE_LIB_PATH)/include
GEMMINI_SW_PATH := $(RADIANCE_LIB_PATH)/mxgemmini
SOC_DIR := {ROOT / 'soc'}
LLVM_MUON := {LLVM}
include {ROOT / 'kernels/common.mk'}
kernel.mu.o: data
""")
    env = os.environ.copy()
    env.update({
        "RISCV_TOOLCHAIN_PATH": str(TOOLCHAIN),
        "RISCV64_TOOLCHAIN_PATH": str(TOOLCHAIN),
        "RISCV_SYSROOT": str(TOOLCHAIN / "riscv64-unknown-elf"),
        "MU_LIBC_INCLUDE": str(TOOLCHAIN / "riscv64-unknown-elf/include"),
        "LLVM_MUON": str(LLVM), "MU_STACK_WORD_STRIDE": "1", "MU_USE_LIBC": "1",
        "EXTRA_MU_CFLAGS": "-nostdinc++ -isystem " + str(LLVM / "include/c++/v1"),
        "RADIANCE_DISABLE_CYCLOTRON_TRACE": "1",
    })
    build_log = out / "build.log"
    with build_log.open("w") as stream:
        built = subprocess.run(["make", "-j1", "kernel.radiance.elf"],
                               cwd=out, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=False)
    if built.returncode:
        raise RuntimeError(f"build failed ({built.returncode}): {build_log}")
    elf = out / "kernel.radiance.elf"
    config = (CYCLOTRON / "config.toml").read_text()
    config, count = re.subn(r"(?m)^timeout\s*=\s*\d+",
                            f"timeout = {max_cycles}", config, count=1)
    if count != 1:
        raise ValueError("Cyclotron config has no unique simulation timeout")
    config = config.replace('"config/timing/', f'"{CYCLOTRON}/config/timing/')
    config_path = out / "cyclotron-config.toml"
    config_path.write_text(config)
    nm = subprocess.check_output([str(LLVM / "bin/llvm-nm"), "-n", str(elf)],
                                 text=True)
    symbols = {name: int(address, 16) for address, name in re.findall(
        r"(?m)^([0-9a-f]+) [BD] (\w+)$", nm)}
    length = fixture["batch"] * fixture["width"] * 4
    if "selected_raw" not in symbols or "gold_raw" not in symbols:
        raise ValueError("device ELF lacks selected or golden buffer")
    base = min(symbols[name] for name in ("selected_raw", "gold_raw"))
    end = max(symbols[name] + length for name in ("selected_raw", "gold_raw"))
    memory_path = out / "memory.bin"
    env["CYCLOTRON_DUMP_GMEM"] = f"0x{base:x}:{end - base}:{memory_path}"
    log = out / "cyclotron.log"
    with log.open("w") as stream:
        sim = subprocess.run([str(CYCLOTRON / "target/release/cyclotron"),
                              str(config_path), "--binary-path", str(elf),
                              "--gen-trace", "false"], cwd=CYCLOTRON,
                             env=env, stdout=stream, stderr=subprocess.STDOUT,
                             timeout=wall_timeout, check=False)
    output = log.read_text(errors="replace")
    cycles = re.findall(r"simulation finished after (\d+) cycles", output)
    if (sim.returncode or len(cycles) != 1
            or output.count("isa-test passed with tohost=0") != 1):
        raise RuntimeError(f"Cyclotron check failed ({sim.returncode}): {log}")
    memory = memory_path.read_bytes()
    if len(memory) != end - base:
        raise AssertionError("Cyclotron memory dump has wrong length")
    def values(name: str) -> np.ndarray:
        offset = symbols[name] - base
        return np.frombuffer(memory[offset:offset + length], dtype="<f4")
    actual, expected = values("selected_raw"), values("gold_raw")
    if (hashlib.sha256(expected.tobytes()).hexdigest() != fixture["gold_sha256"]
            or not np.array_equal(actual, expected)):
        raise AssertionError("device selection differs from fixture")
    report = {
        "schema": "radiance.select-last-token-device-check.v1",
        "status": "passed", "backend": "cyclotron", "rtl_execution": False,
        "performance_measurement": False,
        "functional_cycles": int(cycles[0]),
        "checked_elements": actual.size,
        "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "fixture": fixture,
        "kernel_source_sha256": sha(HERE / "kernel.cpp"),
        "generator_source_sha256": sha(HERE / "gen_data.py"),
        "elf_sha256": sha(elf), "config_sha256": sha(config_path),
        "simulator_sha256": sha(CYCLOTRON / "target/release/cyclotron"),
        "build_log_sha256": sha(build_log),
        "cyclotron_log_sha256": sha(log),
    }
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", choices=("synthetic", "qwen3"),
                        default="synthetic")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--max-cycles", type=int, default=20_000_000)
    parser.add_argument("--wall-timeout", type=int, default=300)
    args = parser.parse_args()
    report = run(args.out, args.source, args.build_dir,
                 args.max_cycles, args.wall_timeout)
    print(json.dumps({"status": report["status"],
                      "functional_cycles": report["functional_cycles"]}))


if __name__ == "__main__":
    main()
