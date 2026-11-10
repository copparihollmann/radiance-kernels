#!/usr/bin/env python3
"""Exhaustively run every Gaussian Fan1/Fan2 stage in functional Cyclotron."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parent
MUON_NM = ROOT.parents[1] / "llvm/llvm-muon/bin/llvm-nm"
FINISHED = re.compile(r"simulation finished after (\d+) cycles")
KERNEL_CYCLES = re.compile(r"nightly_kernel_cycles=(\d+)")
MODEL_STAGE_FLAG = 0x80000000
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1
GUARDS = {
    "before": 0x6A550000,
    "middle": 0x6A551000,
    "model": 0x6A552000,
    "after": 0x6A553000,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv64(words) -> int:
    value = FNV_OFFSET
    for word in words:
        value = ((value ^ (int(word) & 0xFFFFFFFF)) * FNV_PRIME) & MASK64
    return value


def symbol_addresses(device_elf: Path) -> dict[str, int]:
    wanted = {
        "gaussian_model_a", "gaussian_model_b", "gaussian_model_m",
        "gaussian_owners", "gaussian_guard_before", "gaussian_guard_middle",
        "gaussian_guard_model", "gaussian_guard_after",
    }
    result = {}
    output = subprocess.check_output(
        [str(MUON_NM), "-n", str(device_elf)], text=True)
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2] in wanted:
            result[fields[2]] = int(fields[0], 16)
    missing = wanted - result.keys()
    if missing:
        raise ValueError(f"missing symbols: {sorted(missing)}")
    return result


def make_config(source: dict, elf: Path, addresses: dict[str, int],
                timeout: int, stage: int) -> str:
    sms = int(source["sm_count"])
    size = int(source["size"])
    matrix_words = size * size
    owner_stride = int(source["owner_stride"])
    tags = ",\n  ".join(
        f"[[0x80038, 0x{0xC1D00000 | cluster:08x}], "
        f"[0x8003c, 0x{0xC1D00000 | cluster:08x}]]"
        for cluster in range(sms)
    )
    digest = source["stage_expected_digests"][stage]
    t = stage // 2
    owners = [((row // 16) % sms if row < size and row > t else 0xFFFFFFFF)
              for row in range(owner_stride)]
    checks = [
        ("gaussian-a", addresses["gaussian_model_a"] +
         stage * matrix_words * 4, matrix_words, digest["a"]),
        ("gaussian-b", addresses["gaussian_model_b"] +
         stage * size * 4, size, digest["b"]),
        ("gaussian-m", addresses["gaussian_model_m"] +
         stage * matrix_words * 4, matrix_words, digest["m"]),
        ("gaussian-owners", addresses["gaussian_owners"] +
         stage * owner_stride * 4, owner_stride, f"{fnv64(owners):016x}"),
    ]
    for name, pattern in GUARDS.items():
        checks.append((f"gaussian-guard-{name}",
                       addresses[f"gaussian_guard_{name}"], 16,
                       f"{fnv64(pattern ^ i for i in range(16)):016x}"))
    check_lines = [
        f'  {{ label = "{label}", address = 0x{address:08x}, '
        f'words = {words}, expected_fnv64 = "{expected}" }},'
        for label, address, words, expected in checks
    ]
    return "\n".join((
        "[muon]", "num_lanes = 16",
        f"num_warps = {int(source['muon_warps_per_core'])}",
        "num_cores = 2", "num_regs = 256", "start_pc = 0x10000000", "",
        "[mem]", "io_cout_addr = 0xFF080000", "io_cout_size = 64", "",
        "[sim]", f"elf = {json.dumps(str(elf))}", f"timeout = {timeout}",
        "log_level = 0", "trace = false", "timing = false",
        f"num_clusters = {sms}", "report_nightly_cycles = true",
        "smem_init = [", "  [0x80000, 0x6e1a0c11],",
        "  [0x80004, 0x6e1a0c11],", "  [0x800c0, 0x60c0ffee],",
        "  [0x800c4, 0x60c0ffee],",
        f"  [0x80008, 0x{MODEL_STAGE_FLAG | stage:08x}],",
        f"  [0x8000c, 0x{MODEL_STAGE_FLAG | stage:08x}],", "]",
        "smem_init_per_cluster = [", f"  {tags}", "]",
        "gmem_checks = [", *check_lines, "]", "smem_checks = [", "]", "",
    ))


def run_stage(cyclotron: Path, source: dict, source_result: Path,
              elf: Path, addresses: dict[str, int], out: Path,
              timeout: int, stage: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.toml").write_text(
        make_config(source, elf, addresses, timeout, stage))
    started = time.time()
    proc = subprocess.run(
        [str(cyclotron), str(out / "config.toml")], cwd=out,
        capture_output=True, text=True,
        env={**os.environ, "CYCLOTRON_PERF_LOG_DIR": str(out / "perf")},
    )
    (out / "stdout.log").write_text(proc.stdout)
    (out / "stderr.log").write_text(proc.stderr)
    finished = FINISHED.search(proc.stdout)
    kernel = KERNEL_CYCLES.search(proc.stdout)
    labels = ("gaussian-a", "gaussian-b", "gaussian-m", "gaussian-owners",
              "gaussian-guard-before", "gaussian-guard-middle",
              "gaussian-guard-model", "gaussian-guard-after")
    checks = {label: bool(re.search(
        rf"gmem check {re.escape(label)}: PASS\b", proc.stdout))
        for label in labels}
    passed = (proc.returncode == 0 and finished is not None and
              kernel is not None and all(checks.values()))
    result = {
        "schema": "radiance-gaussian-cyclotron-stage-v1",
        "status": "passed" if passed else "failed",
        "memory_model": "functional-no-timing", "timing": False,
        "profile": source["profile"], "size": int(source["size"]),
        "sm_count": int(source["sm_count"]), "stage": stage,
        "pivot": stage // 2, "phase": "Fan1" if stage % 2 == 0 else "Fan2",
        "kernel_cycles": int(kernel.group(1)) if kernel else None,
        "simulation_cycles": int(finished.group(1)) if finished else None,
        "correctness_checks": checks, "returncode": proc.returncode,
        "wall_seconds": time.time() - started, "elf": str(elf),
        "elf_sha256": source["elf_sha256"],
        "source_result": str(source_result.resolve()),
        "source_result_sha256": sha256(source_result),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_one(cyclotron: Path, result_path: Path, out: Path,
            timeout: int) -> dict:
    source = json.loads(result_path.read_text())
    if source.get("status") != "built" or source.get("loader") != "logical":
        raise ValueError(f"not a built logical artifact: {result_path}")
    elf = Path(source["elf"]).resolve()
    device = Path(source["device_elf"]).resolve()
    if sha256(elf) != source["elf_sha256"]:
        raise ValueError(f"SoC ELF hash changed: {elf}")
    if sha256(device) != source["device_elf_sha256"]:
        raise ValueError(f"device ELF hash changed: {device}")
    addresses = symbol_addresses(device)
    stages = []
    for stage in range(int(source["stages"])):
        record = run_stage(cyclotron, source, result_path, elf, addresses,
                           out / f"stage-{stage:03d}", timeout, stage)
        stages.append(record)
        print(f"{record['status']:6} {source['name']} "
              f"{record['phase']}({record['pivot']}) "
              f"cycles={record['kernel_cycles']}", flush=True)
        if record["status"] != "passed":
            break
    passed = len(stages) == int(source["stages"]) and all(
        stage["status"] == "passed" for stage in stages)
    result = {
        "schema": "radiance-gaussian-cyclotron-functional-result-v1",
        "status": "passed" if passed else "failed",
        "memory_model": "functional-no-timing", "timing": False,
        "profile": source["profile"], "size": int(source["size"]),
        "sm_count": int(source["sm_count"]),
        "expected_stages": int(source["stages"]),
        "tested_stages": len(stages),
        "passed_stages": sum(s["status"] == "passed" for s in stages),
        "total_kernel_cycles": sum(int(s["kernel_cycles"] or 0)
                                   for s in stages),
        "stages": stages, "elf": str(elf),
        "elf_sha256": source["elf_sha256"],
        "source_result": str(result_path.resolve()),
        "source_result_sha256": sha256(result_path),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--cyclotron", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=2_000_000_000)
    args = parser.parse_args()
    cyclotron = args.cyclotron.resolve()
    if not cyclotron.is_file():
        parser.error(f"missing Cyclotron binary: {cyclotron}")
    paths = []
    for path in sorted(args.build_root.resolve().rglob("result.json")):
        candidate = json.loads(path.read_text())
        if (candidate.get("family") == "gaussian" and
                candidate.get("loader") == "logical" and
                candidate.get("status") == "built"):
            paths.append(path)
    if not paths:
        parser.error("no built logical Gaussian results found")
    records = []
    for path in paths:
        source = json.loads(path.read_text())
        job = args.out.resolve() / source["profile"] / f"{source['sm_count']}sm"
        records.append(run_one(cyclotron, path, job, args.timeout))
    summary = {
        "schema": "radiance-gaussian-cyclotron-functional-summary-v1",
        "memory_model": "functional-no-timing", "is_hbm": False,
        "result_count": len(records),
        "passed": sum(r["status"] == "passed" for r in records),
        "failed": sum(r["status"] != "passed" for r in records),
        "total_stage_checks": sum(r["tested_stages"] for r in records),
        "passed_stage_checks": sum(r["passed_stages"] for r in records),
        "results": records,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
