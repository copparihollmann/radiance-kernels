#!/usr/bin/env python3
"""Run a built logical sfilter artifact on Cyclotron without touching hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parent
DEFAULT_CYCLOTRON = (
    ROOT.parents[3] / "src/radiance/cyclotron/target-native/release/cyclotron"
)
DEFAULT_NM = ROOT.parents[1] / "llvm/llvm-muon/bin/llvm-nm"
FINISHED = re.compile(r"simulation finished after (\d+) cycles")
KERNEL_CYCLES = re.compile(r"nightly_kernel_cycles=(\d+)")
CLUSTER_CYCLES = re.compile(r"nightly_cluster_cycles cluster=(\d+) cycles=(\d+)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symbol_address(nm: Path, elf: Path, symbol: str) -> int:
    output = subprocess.check_output([str(nm), "-n", str(elf)], text=True)
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2] == symbol:
            return int(fields[0], 16)
    raise ValueError(f"missing {symbol} in {elf}")


def config_text(manifest: dict, address: int, elf: Path, timeout: int) -> str:
    sms = int(manifest["sm_count"])
    tags = ",\n".join(
        "  [[0x80038, 0x%08x], [0x8003c, 0x%08x]]" %
        (0xC1D00000 | cluster, 0xC1D00000 | cluster)
        for cluster in range(sms)
    )
    return f"""[muon]
num_lanes = 16
num_warps = {int(manifest['muon_warps_per_core'])}
num_cores = 2
num_regs = 256
start_pc = 0x10000000

[mem]
io_cout_addr = 0xFF080000
io_cout_size = 64

[sim]
elf = {json.dumps(str(elf))}
timeout = {timeout}
log_level = 0
trace = false
timing = false
num_clusters = {sms}
report_nightly_cycles = true
smem_init = [
  [0x80000, 0x6e1a0c11],
  [0x80004, 0x6e1a0c11],
  [0x800c0, 0x60c0ffee],
  [0x800c4, 0x60c0ffee],
]
smem_init_per_cluster = [
{tags},
]
gmem_checks = [
  {{ label = "sfilter-complete-output", address = 0x{address:08x}, words = {int(manifest['output_element_count'])}, expected_fnv64 = {json.dumps(manifest['expected_digest'])} }},
]
smem_checks = []
"""


def run(artifact: Path, cyclotron: Path, nm: Path, timeout: int) -> dict:
    manifest_path = artifact / "result.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "built" or manifest.get("loader") != "logical":
        raise ValueError("functional smoke requires a built logical artifact")
    if not manifest.get("scalar_only") or not manifest.get("no_vpu"):
        raise ValueError("artifact does not assert scalar no-VPU execution")
    elf = Path(manifest["elf"]).resolve()
    device_elf = Path(manifest["device_elf"]).resolve()
    if sha256(elf) != manifest["elf_sha256"]:
        raise ValueError("fused ELF hash mismatch")
    if sha256(device_elf) != manifest["device_elf_sha256"]:
        raise ValueError("device ELF hash mismatch")
    address = symbol_address(nm, device_elf, "sfilter_output")
    config = artifact / "cyclotron-functional.toml"
    config.write_text(config_text(manifest, address, elf, timeout))

    started = time.time()
    proc = subprocess.run([str(cyclotron), str(config)], cwd=artifact,
                          capture_output=True, text=True)
    (artifact / "cyclotron-functional.stdout.log").write_text(proc.stdout)
    (artifact / "cyclotron-functional.stderr.log").write_text(proc.stderr)
    finished = FINISHED.search(proc.stdout)
    kernel = KERNEL_CYCLES.search(proc.stdout)
    clusters = [
        {"cluster": int(cluster), "cycles": int(cycles)}
        for cluster, cycles in CLUSTER_CYCLES.findall(proc.stdout)
    ]
    digest_line = (
        f"gmem check sfilter-complete-output: PASS address=0x{address:08x} "
        f"words={manifest['output_element_count']} fnv64={manifest['expected_digest']}"
    )
    passed = (
        proc.returncode == 0 and finished is not None and kernel is not None and
        digest_line in proc.stdout and len(clusters) == manifest["sm_count"] and
        {row["cluster"] for row in clusters} == set(range(manifest["sm_count"]))
    )
    result = {
        "schema": "radiance-sfilter-cyclotron-functional-v1",
        "status": "passed" if passed else "failed",
        "artifact": str(artifact),
        "artifact_manifest_sha256": sha256(manifest_path),
        "elf_sha256": manifest["elf_sha256"],
        "device_elf_sha256": manifest["device_elf_sha256"],
        "cyclotron_sha256": sha256(cyclotron),
        "sm_count": manifest["sm_count"],
        "size": manifest["size"],
        "output_element_count": manifest["output_element_count"],
        "output_address": f"0x{address:08x}",
        "expected_digest": manifest["expected_digest"],
        "kernel_cycles": int(kernel.group(1)) if kernel else None,
        "simulation_cycles": int(finished.group(1)) if finished else None,
        "cluster_cycles": clusters,
        "returncode": proc.returncode,
        "wall_seconds": time.time() - started,
        "memory_model": "functional-no-timing",
        "hardware_run": False,
    }
    result_path = artifact / "functional-result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--cyclotron", type=Path, default=DEFAULT_CYCLOTRON)
    parser.add_argument("--nm", type=Path, default=DEFAULT_NM)
    parser.add_argument("--timeout", type=int, default=200_000_000)
    args = parser.parse_args()
    run(args.artifact.resolve(), args.cyclotron.resolve(), args.nm.resolve(),
        args.timeout)


if __name__ == "__main__":
    main()
