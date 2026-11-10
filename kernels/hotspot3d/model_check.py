#!/usr/bin/env python3
"""Run Hotspot3D logical artifacts in Cyclotron's no-timing model."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv64(words) -> int:
    value = 0xCBF29CE484222325
    for word in words:
        value = ((value ^ (int(word) & 0xFFFFFFFF)) *
                 0x100000001B3) & ((1 << 64) - 1)
    return value


def symbol_address(device_elf: Path, name: str) -> int:
    output = subprocess.check_output(
        [str(MUON_NM), "-n", str(device_elf)], text=True)
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2] == name:
            return int(fields[0], 16)
    raise ValueError(f"missing {name} in {device_elf}")


def make_config(source: dict, elf: Path, addresses: dict[str, int],
                timeout: int, iteration: int) -> str:
    sms = int(source["sm_count"])
    tags = ",\n  ".join(
        f"[[0x80038, 0x{0xC1D00000 | cluster:08x}], "
        f"[0x8003c, 0x{0xC1D00000 | cluster:08x}]]"
        for cluster in range(sms)
    )
    guard_digests = {
        "before": fnv64(0x483D0000 ^ index for index in range(16)),
        "middle_a": fnv64(0x483D1000 ^ index for index in range(16)),
        "middle_b": fnv64(0x483D2000 ^ index for index in range(16)),
        "after": fnv64(0x483D3000 ^ index for index in range(16)),
    }
    checks = [
        "  { label = \"hotspot3d-output\", address = "
        f"0x{addresses['hotspot3d_output']:08x}, "
        f"words = {int(source['output_element_count'])}, expected_fnv64 = "
        f"\"{source['stage_expected_digests'][iteration]}\" }},",
        "  { label = \"hotspot3d-owners\", address = "
        f"0x{addresses['hotspot3d_owners']:08x}, "
        f"words = {int(source['columns'])}, expected_fnv64 = "
        f"\"{source['owner_digest']}\" }},",
    ]
    for suffix, label in (("before", "before"),
                          ("middle_a", "middle-a"),
                          ("middle_b", "middle-b"),
                          ("after", "after")):
        checks.append(
            f"  {{ label = \"hotspot3d-guard-{label}\", address = "
            f"0x{addresses['hotspot3d_guard_' + suffix]:08x}, words = 16, "
            f"expected_fnv64 = \"{guard_digests[suffix]:016x}\" }},"
        )
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
        f"  [0x80008, 0x{MODEL_STAGE_FLAG | iteration:08x}],",
        f"  [0x8000c, 0x{MODEL_STAGE_FLAG | iteration:08x}],", "]",
        "smem_init_per_cluster = [", f"  {tags}", "]",
        "gmem_checks = [", *checks, "]", "smem_checks = [", "]", "",
    ))


def run_stage(cyclotron: Path, source: dict, result_path: Path, elf: Path,
              addresses: dict[str, int], out: Path, timeout: int,
              iteration: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.toml").write_text(
        make_config(source, elf, addresses, timeout, iteration))
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
    labels = (
        "hotspot3d-output", "hotspot3d-owners",
        "hotspot3d-guard-before", "hotspot3d-guard-middle-a",
        "hotspot3d-guard-middle-b", "hotspot3d-guard-after",
    )
    checks = {
        label: bool(re.search(rf"gmem check {label}: PASS\b", proc.stdout))
        for label in labels
    }
    passed = (
        proc.returncode == 0 and finished is not None and kernel is not None
        and all(checks.values())
    )
    stage = {
        "schema": "radiance-hotspot3d-cyclotron-stage-result-v1",
        "status": "passed" if passed else "failed",
        "memory_model": "functional-no-timing", "timing": False,
        "sm_count": int(source["sm_count"]), "profile": source["profile"],
        "name": source["name"], "iteration": iteration,
        "input_source": "embedded-independent-cpu-oracle-state",
        "expected_digest": source["stage_expected_digests"][iteration],
        "kernel_cycles": int(kernel.group(1)) if kernel else None,
        "correctness_checks": checks,
        "simulation_cycles": int(finished.group(1)) if finished else None,
        "returncode": proc.returncode, "wall_seconds": time.time() - started,
        "elf": str(elf), "elf_sha256": source["elf_sha256"],
        "source_result": str(result_path.resolve()),
        "source_result_sha256": sha256(result_path),
    }
    (out / "result.json").write_text(json.dumps(stage, indent=2) + "\n")
    return stage


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
    symbols = (
        "hotspot3d_output", "hotspot3d_owners", "hotspot3d_guard_before",
        "hotspot3d_guard_middle_a", "hotspot3d_guard_middle_b",
        "hotspot3d_guard_after",
    )
    addresses = {name: symbol_address(device, name) for name in symbols}
    stages = []
    for iteration in range(int(source["iterations"])):
        stages.append(run_stage(
            cyclotron, source, result_path, elf, addresses,
            out / f"iteration-{iteration}", timeout, iteration,
        ))
    passed = all(stage["status"] == "passed" for stage in stages)
    result = {
        "schema": "radiance-hotspot3d-cyclotron-functional-result-v1",
        "status": "passed" if passed else "failed",
        "memory_model": "functional-no-timing", "timing": False,
        "sm_count": int(source["sm_count"]), "profile": source["profile"],
        "name": source["name"], "stage_count": len(stages),
        "passed_stages": sum(stage["status"] == "passed" for stage in stages),
        "total_kernel_cycles": sum(
            int(stage["kernel_cycles"] or 0) for stage in stages),
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
        if (candidate.get("family") == "hotspot3d" and
                candidate.get("loader") == "logical" and
                candidate.get("status") == "built"):
            paths.append(path)
    if not paths:
        parser.error("no built logical Hotspot3D results found")
    records = []
    for path in paths:
        source = json.loads(path.read_text())
        job = (args.out.resolve() / source["profile"] /
               f"{int(source['sm_count'])}sm")
        result = run_one(cyclotron, path, job, args.timeout)
        records.append(result)
        cycles = result["total_kernel_cycles"]
        print(f"{result['status']:6} {result['name']} cycles={cycles}",
              flush=True)
    summary = {
        "schema": "radiance-hotspot3d-cyclotron-functional-summary-v1",
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
