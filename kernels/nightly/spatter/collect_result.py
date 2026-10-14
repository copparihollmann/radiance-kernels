#!/usr/bin/env python3
"""Bind one nightly Spatter FireSim report to its exact ELF and input deck."""

import argparse
import hashlib
import json
from pathlib import Path
import re


KINDS = ("gather", "scatter", "gs", "multigather", "multiscatter")


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


def parse_uart(text, case, clusters, cores_per_cluster):
    kind, length, count, printed_clusters = map(int, unique(
        r"^spatter: kind=(\d+) length=(\d+) count=(\d+) clusters=(\d+)\s*$",
        text, "Spatter header"))
    if (kind, length, count, printed_clusters) != (
            KINDS.index(case["kind"]), case["pattern_length"],
            case["count"], clusters):
        raise ValueError("UART case or cluster count differs")
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
    output_elements = case["output_elements"]
    verify_rows = re.findall(
        r"^spatter: cluster=(\d+) exact=(\d+)/(\d+)\s*$",
        text, re.MULTILINE)
    if len(verify_rows) != clusters:
        raise ValueError("incomplete per-cluster full-output checks")
    verified = set()
    for row in verify_rows:
        cl, exact, total = map(int, row)
        if cl in verified or cl >= clusters or (exact, total) != (
                output_elements, output_elements):
            raise ValueError("full-output check failed or cluster duplicated")
        verified.add(cl)
    samples, total_samples = map(int, unique(
        r"^spatter: host_samples=(\d+)/(\d+)\s*$", text,
        "host sample check"))
    if (samples, total_samples) != (min(64, output_elements),) * 2:
        raise ValueError("host sample check failed")
    logical_bytes = case["logical_payload_bytes"]
    if int(unique(r"^logical_bytes=(\d+)\s*$", text,
                  "logical byte count")) != logical_bytes:
        raise ValueError("logical byte count differs")
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
    if receipt.get("schema") != "radiance.nightly.spatter.hbm-builds.v1":
        raise ValueError("wrong build receipt schema")
    if args.cores_per_cluster != receipt.get("compiled_cores_per_cluster"):
        raise ValueError("cores per cluster differ from compiled host")
    cases = [case for case in receipt["cases"] if case["case"] == args.case]
    if len(cases) != 1:
        raise ValueError("case missing or duplicated in build receipt")
    case = cases[0]
    data = json.loads(args.data_receipt.read_text())
    if (sha256(args.data_receipt) != case["data_receipt_sha256"] or
            data["case"] != args.case or data["kind"] != case["kind"] or
            data["golden_sha256"] != case["golden_sha256"]):
        raise ValueError("data receipt differs from compiled case")
    if sha256(args.elf) != case["elfs"][f"{args.sm_count}sm"]["sha256"]:
        raise ValueError("ELF differs from compiled case")
    measurement = parse_uart(args.uart_log.read_text(errors="replace"), case,
                             args.sm_count, args.cores_per_cluster)
    return {"schema": "radiance.nightly.spatter.firesim-result.v1",
            "status": "passed", "case": args.case, "kind": case["kind"],
            "pattern_length": case["pattern_length"], "count": case["count"],
            "sm_count": args.sm_count,
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
    parser.add_argument("--case", type=int, required=True)
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
    result = collect(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
