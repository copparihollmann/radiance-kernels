#!/usr/bin/env python3
"""Compile and check the connected reduced Mamba2 mixer ELF."""

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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOLCHAIN = Path("/scratch/agustin/projects/chipyard/.conda-env/riscv-tools")
LLVM = Path("/scratch/agustin/projects/chipyard/radiance-kernels-firesim/llvm/llvm-muon")
CYCLOTRON = Path("/scratch/agustin/projects/chipyard/generators/radiance/cyclotron")
FIELDS = {
    "conv_x_raw": (336, "gold_conv_x_raw"),
    "dt_raw": (28, "gold_dt_raw"),
    "gate_raw": (224, "gold_gate_raw"),
    "conv_y_raw": (336, "gold_conv_y_raw"),
    "conv_state_raw": (192, "gold_conv_state_raw"),
    "scan_y_raw": (224, "gold_scan_y_raw"),
    "scan_state_raw": (128, "gold_scan_state_raw"),
    "full_y_raw": (224, "gold_full_y_raw"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(out: Path, conv_dir: Path, scan_dir: Path, reference_python: Path,
        max_cycles: int, wall_timeout: int) -> dict:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(reference_python), str(HERE / "gen_upstream_data.py"),
                    "--out", str(out / "data"),
                    "--receipt", str(out / "fixture.json"),
                    "--conv-dir", str(conv_dir),
                    "--scan-dir", str(scan_dir)], check=True)
    fixture = json.loads((out / "fixture.json").read_text())
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
        "LLVM_MUON": str(LLVM),
        "MU_STACK_WORD_STRIDE": "1", "MU_USE_LIBC": "1",
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
    names = list(FIELDS) + [gold for _, gold in FIELDS.values()]
    if any(name not in symbols for name in names):
        raise ValueError("device ELF lacks an output or golden symbol")
    sizes = {name: count * 4 for name, (count, _) in FIELDS.items()}
    sizes.update({gold: count * 4 for count, gold in FIELDS.values()})
    base = min(symbols[name] for name in names)
    end = max(symbols[name] + sizes[name] for name in names)
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
    checks = {}
    for name, (count, golden) in FIELDS.items():
        def values(field: str) -> np.ndarray:
            offset = symbols[field] - base
            return np.frombuffer(memory[offset:offset + count * 4], dtype="<f4")
        actual, expected = values(name), values(golden)
        gold_hashes = (fixture["base_fixture"]["array_sha256"]
                       | fixture["array_sha256"])
        if hashlib.sha256(expected.tobytes()).hexdigest() != gold_hashes[golden]:
            raise AssertionError(f"device {golden} is not fixture golden")
        rtol, atol = (5e-5, 5e-6) if name == "full_y_raw" else (3e-5, 3e-6)
        if not np.allclose(actual, expected, rtol=rtol, atol=atol):
            raise AssertionError(f"device {name} differs from golden")
        checks[name] = {
            "elements": count,
            "max_abs_error": float(np.max(np.abs(actual - expected))),
            "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        }
    report = {
        "schema": "radiance.mamba-mixer-device-check.v1",
        "status": "passed", "backend": "cyclotron", "rtl_execution": False,
        "performance_measurement": False, "same_device_program": True,
        "host_mediated_handoff": False,
        "functional_cycles": int(cycles[0]), "checks": checks,
        "fixture": fixture,
        "kernel_source_sha256": sha(HERE / "kernel.cpp"),
        "generator_source_sha256": sha(HERE / "gen_upstream_data.py"),
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
    parser.add_argument("--conv-dir", type=Path, required=True)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, default=20_000_000)
    parser.add_argument("--wall-timeout", type=int, default=300)
    args = parser.parse_args()
    report = run(args.out, args.conv_dir, args.scan_dir, args.reference_python,
                 args.max_cycles, args.wall_timeout)
    print(json.dumps({"status": report["status"],
                      "functional_cycles": report["functional_cycles"],
                      "result": str(args.out.resolve() / "result.json")}, indent=2))


if __name__ == "__main__":
    main()
