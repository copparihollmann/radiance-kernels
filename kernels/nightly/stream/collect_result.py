#!/usr/bin/env python3
"""Bind one real nightly STREAM FireSim UART report to its ELF and run files."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique(pattern, text, label):
    matches = re.findall(pattern, text, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"expected one {label}, found {len(matches)}")
    return matches[0]


def parse_uart(text, kind, elements, clusters, cores_per_cluster,
               logical_bytes):
    header = unique(r"^stream: kind=(\d+) N=(\d+) clusters=(\d+)\s*$",
                    text, "STREAM header")
    expected_kind = {"copy": 0, "scale": 1, "add": 2, "triad": 3}[kind]
    if tuple(map(int, header)) != (expected_kind, elements, clusters):
        raise ValueError("UART operation, length, or cluster count differs")
    if len(re.findall(r"^PASS\s*$", text, re.MULTILINE)) != 1 or re.search(
            r"^FAIL\s*$", text, re.MULTILINE):
        raise ValueError("UART lacks a unique PASS without FAIL")
    core_rows = re.findall(
        r"^cluster (\d+) core (\d+): start=(\d+) end=(\d+) cycles=(\d+)\s*$",
        text, re.MULTILINE)
    expected_cores = {(cl, core) for cl in range(clusters)
                      for core in range(cores_per_cluster)}
    if len(core_rows) != len(expected_cores):
        raise ValueError("incomplete or duplicate per-core timing rows")
    cores = []
    for row in core_rows:
        cl, core, start, end, cycles = map(int, row)
        if end <= start or end - start != cycles:
            raise ValueError("per-core timing is inconsistent or wrapped")
        cores.append({"cluster": cl, "core": core, "start": start,
                      "end": end, "cycles": cycles})
    if {(row["cluster"], row["core"]) for row in cores} != expected_cores:
        raise ValueError("unexpected cluster or core in timing rows")
    cycles = int(unique(r"^kernel_cycles=(\d+)\s*$", text,
                        "kernel cycle count"))
    if cycles != max(row["end"] for row in cores) - min(
            row["start"] for row in cores):
        raise ValueError("kernel cycle count differs from core stamps")
    verify_rows = re.findall(
        r"^stream: cluster=(\d+) exact=(\d+)/(\d+) nonfinite=(\d+)\s*$",
        text, re.MULTILINE)
    if len(verify_rows) != clusters:
        raise ValueError("incomplete per-cluster full-output checks")
    verified = set()
    for row in verify_rows:
        cl, exact, total, nonfinite = map(int, row)
        if cl in verified or cl >= clusters or (exact, total, nonfinite) != (
                elements, elements, 0):
            raise ValueError("full-output check failed or cluster duplicated")
        verified.add(cl)
    samples, total_samples = map(int, unique(
        r"^stream: host_samples=(\d+)/(\d+)\s*$", text,
        "host sample check"))
    if (samples, total_samples) != (min(64, elements), min(64, elements)):
        raise ValueError("host sample check failed")
    if int(unique(r"^logical_bytes=(\d+)\s*$", text,
                  "logical byte count")) != logical_bytes:
        raise ValueError("logical payload byte count differs")
    reported_rate = int(unique(r"^bytes_per_gpu_cycle_x100=(\d+)\s*$",
                               text, "bytes per cycle"))
    if reported_rate != logical_bytes * 100 // cycles:
        raise ValueError("reported bytes per cycle differs from raw values")
    return {"kernel_cycles": cycles, "logical_payload_bytes": logical_bytes,
            "bytes_per_gpu_cycle_x100": reported_rate,
            "full_output_exact": True, "host_samples_exact": samples,
            "per_core": cores}


def collect(args):
    if args.sm_count not in (1, 2) or args.cores_per_cluster < 1:
        raise ValueError("SM count must be 1 or 2 and cores per cluster positive")
    if args.firesim_exit_code != 0:
        raise ValueError("FireSim exited unsuccessfully")
    receipt = json.loads(args.build_receipt.read_text())
    if receipt.get("schema") != "radiance.nightly.stream.hbm-builds.v1":
        raise ValueError("wrong build receipt schema")
    if args.cores_per_cluster != receipt.get("compiled_cores_per_cluster"):
        raise ValueError("reported cores per cluster differs from the host ELF")
    cases = [c for c in receipt["cases"] if c["operation"] == args.operation]
    if len(cases) != 1:
        raise ValueError("operation missing or duplicated in build receipt")
    case = cases[0]
    data = json.loads(args.data_receipt.read_text())
    if (sha256(args.data_receipt) != case["data_receipt_sha256"] or
            data["operation"] != args.operation or
            data["elements"] != receipt["elements"] or
            data["golden_sha256"] != case["golden_sha256"]):
        raise ValueError("data receipt differs from the compiled case")
    if sha256(args.elf) != case["elfs"][f"{args.sm_count}sm"]["sha256"]:
        raise ValueError("ELF differs from the compiled case")
    uart = args.uart_log.read_text(errors="replace")
    measurement = parse_uart(uart, args.operation, data["elements"],
                             args.sm_count, args.cores_per_cluster,
                             case["logical_payload_bytes"])
    return {"schema": "radiance.nightly.stream.firesim-result.v1",
            "status": "passed", "operation": args.operation,
            "elements": data["elements"], "sm_count": args.sm_count,
            "cores_per_cluster": args.cores_per_cluster,
            "bitstream_id": args.bitstream_id,
            "hwdb_config_sha256": sha256(args.hwdb_config),
            "workload_config_sha256": sha256(args.workload_config),
            "build_receipt_sha256": sha256(args.build_receipt),
            "data_receipt_sha256": sha256(args.data_receipt),
            "elf_sha256": sha256(args.elf),
            "uart_log_sha256": sha256(args.uart_log),
            "firesim_exit_code": args.firesim_exit_code,
            "backend": "FireSim", "measurement": measurement}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("copy", "scale", "add", "triad"),
                        required=True)
    parser.add_argument("--sm-count", type=int, required=True)
    parser.add_argument("--cores-per-cluster", type=int, required=True)
    parser.add_argument("--bitstream-id", required=True)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--data-receipt", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--hwdb-config", type=Path, required=True)
    parser.add_argument("--workload-config", type=Path, required=True)
    parser.add_argument("--uart-log", type=Path, required=True)
    parser.add_argument("--firesim-exit-code", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = collect(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
