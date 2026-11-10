#!/usr/bin/env python3
"""Run B+Tree logical artifacts in Cyclotron's functional/no-timing model."""

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
CLUSTER_CYCLES = re.compile(
    r"nightly_cluster_cycles cluster=(\d+) cycles=(\d+) start=(\d+) end=(\d+)"
)
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv64(words) -> int:
    value = FNV_OFFSET
    for word in words:
        value = ((value ^ (int(word) & 0xffffffff)) * FNV_PRIME) & MASK64
    return value


def symbol_address(device_elf: Path, name: str) -> int:
    output = subprocess.check_output([str(MUON_NM), "-n", str(device_elf)],
                                     text=True)
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2] == name:
            return int(fields[0], 16)
    raise ValueError(f"missing {name} in {device_elf}")


def make_config(result: dict, elf: Path, point_address: int,
                range_address: int, guard_before_address: int,
                guard_middle_address: int, guard_after_address: int,
                timeout: int) -> str:
    sms = int(result["sm_count"])
    tags = ",\n  ".join(
        f"[[0x80038, 0x{0xC1D00000 | cluster:08x}], "
        f"[0x8003c, 0x{0xC1D00000 | cluster:08x}]]"
        for cluster in range(sms)
    )
    point_words = int(result["point_output_records"]) * 4
    range_words = int(result["range_output_records"]) * 4
    guard_digests = {
        "before": fnv64(0xB7EE0000 ^ index for index in range(16)),
        "middle": fnv64(0xB7DD0000 ^ index for index in range(16)),
        "after": fnv64(0xB7AA0000 ^ index for index in range(16)),
    }
    return "\n".join((
        "[muon]", "num_lanes = 16",
        f"num_warps = {int(result['muon_warps_per_core'])}",
        "num_cores = 2", "num_regs = 256", "start_pc = 0x10000000", "",
        "[mem]", "io_cout_addr = 0xFF080000", "io_cout_size = 64", "",
        "[sim]", f"elf = {json.dumps(str(elf))}", f"timeout = {timeout}",
        "log_level = 0", "trace = false", "timing = false",
        f"num_clusters = {sms}", "report_nightly_cycles = true",
        "smem_init = [", "  [0x80000, 0x6e1a0c11],",
        "  [0x80004, 0x6e1a0c11],", "  [0x800c0, 0x60c0ffee],",
        "  [0x800c4, 0x60c0ffee],", "]",
        "smem_init_per_cluster = [", f"  {tags}", "]",
        "gmem_checks = [",
        "  { label = \"btree-point-records\", address = "
        f"0x{point_address:08x}, words = {point_words}, expected_fnv64 = "
        f"\"{result['point_output_digest']}\" }},",
        "  { label = \"btree-range-records\", address = "
        f"0x{range_address:08x}, words = {range_words}, expected_fnv64 = "
        f"\"{result['range_output_digest']}\" }},",
        "  { label = \"btree-guard-before\", address = "
        f"0x{guard_before_address:08x}, words = 16, expected_fnv64 = "
        f"\"{guard_digests['before']:016x}\" }},",
        "  { label = \"btree-guard-middle\", address = "
        f"0x{guard_middle_address:08x}, words = 16, expected_fnv64 = "
        f"\"{guard_digests['middle']:016x}\" }},",
        "  { label = \"btree-guard-after\", address = "
        f"0x{guard_after_address:08x}, words = 16, expected_fnv64 = "
        f"\"{guard_digests['after']:016x}\" }},",
        "]", "smem_checks = [", "]", "",
    ))


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
    point_address = symbol_address(device, "btree_point_outputs")
    range_address = symbol_address(device, "btree_range_outputs")
    guard_before_address = symbol_address(device, "btree_guard_before")
    guard_middle_address = symbol_address(device, "btree_guard_middle")
    guard_after_address = symbol_address(device, "btree_guard_after")
    out.mkdir(parents=True, exist_ok=True)
    config = make_config(
        source, elf, point_address, range_address,
        guard_before_address, guard_middle_address, guard_after_address,
        timeout,
    )
    (out / "config.toml").write_text(config)
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
    clusters = [
        {"cluster": int(cluster), "cycles": int(cycles),
         "start": int(start), "end": int(end)}
        for cluster, cycles, start, end in CLUSTER_CYCLES.findall(proc.stdout)
    ]
    labels = (
        "btree-point-records", "btree-range-records",
        "btree-guard-before", "btree-guard-middle", "btree-guard-after",
    )
    checks = {
        label: bool(re.search(rf"gmem check {label}: PASS\b", proc.stdout))
        for label in labels
    }
    sms = int(source["sm_count"])
    passed = (
        proc.returncode == 0 and finished is not None and kernel is not None
        and all(checks.values()) and len(clusters) == sms
        and {item["cluster"] for item in clusters} == set(range(sms))
    )
    result = {
        "schema": "radiance-btree-cyclotron-functional-result-v1",
        "status": "passed" if passed else "failed",
        "memory_model": "functional-no-timing", "timing": False,
        "sm_count": sms, "name": source["name"],
        "kernel_cycles": int(kernel.group(1)) if kernel else None,
        "simulation_cycles": int(finished.group(1)) if finished else None,
        "cluster_kernel_cycles": clusters,
        "correctness_checks": checks,
        "returncode": proc.returncode, "wall_seconds": time.time() - started,
        "elf": str(elf), "elf_sha256": source["elf_sha256"],
        "source_result": str(result_path.resolve()),
        "source_result_sha256": sha256(result_path),
    }
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
        if candidate.get("family") == "btree" and candidate.get("loader") == "logical":
            paths.append(path)
    if not paths:
        parser.error("no built logical B+Tree results found")
    records = []
    for path in paths:
        source = json.loads(path.read_text())
        job = args.out.resolve() / f"{int(source['sm_count'])}sm"
        result = run_one(cyclotron, path, job, args.timeout)
        records.append(result)
        print(f"{result['status']:6} {result['name']} "
              f"cycles={result['kernel_cycles']}", flush=True)
    summary = {
        "schema": "radiance-btree-cyclotron-functional-summary-v1",
        "memory_model": "functional-no-timing", "is_hbm": False,
        "result_count": len(records),
        "passed": sum(record["status"] == "passed" for record in records),
        "failed": sum(record["status"] != "passed" for record in records),
        "results": records,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
