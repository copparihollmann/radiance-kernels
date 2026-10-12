#!/usr/bin/env python3
"""Summarize completed Spatter VCS or Verilator runs without assuming a GPU clock."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPORT = re.compile(
    r"Muon Performance Report.*?Cluster\s+(\d+)\s+Core\s+(\d+)"
    r".*?Instructions:\s*(\d+).*?Cycles:\s*(\d+).*?IPC:\s*([0-9.]+)",
    re.DOTALL,
)
DONE = {"passed", "exploratory"}
FIELDS = (
    "run", "simulator", "status", "correctness", "kind", "pattern_length", "count",
    "logical_payload_bytes", "gpu_cycles", "core_instructions", "core_ipc",
    "payload_bytes_per_cycle", "spatter_one_sided_bytes_per_cycle",
    "payload_gbps", "spatter_one_sided_gbps",
)


def row_for(path: Path, gpu_mhz: float | None) -> dict:
    result = json.loads((path / "result.json").read_text())
    row = {field: "" for field in FIELDS}
    row.update({
        "run": path.name,
        "simulator": result.get("simulator", ""),
        "status": result["status"],
        "correctness": result["correctness"],
        "kind": result["kind"],
        "pattern_length": result["pattern_length"],
        "count": result["count"],
        "logical_payload_bytes": result["logical_payload_bytes"],
    })
    if result["status"] not in DONE:
        return row
    simulator = result.get("simulator", "")
    if simulator == "Cyclotron functional model":
        log = (path / "cyclotron.log").read_text(errors="replace")
        if "SPATTER_CHECK" not in log or "simulation finished after" not in log:
            raise ValueError(f"{path}: functional model lacks output readback")
        return row
    if simulator.startswith("Cyclotron"):
        log_name = "cyclotron.log"
    elif simulator.startswith("Verilator"):
        log_name = "verilator.log"
    else:
        log_name = "vcs.log"
    log_path = path / log_name
    if not log_path.exists() and log_name == "vcs.log":
        log_path = path / "previous-vcs.log"
    log = log_path.read_text(errors="replace")
    finish_marker = (
        "SPATTER_CHECK" if log_name == "cyclotron.log" else
        "Verilog $finish" if log_name == "verilator.log" else "$finish called"
    )
    if finish_marker not in log or "*** FAILED ***" in log or "%Error" in log:
        raise ValueError(f"{path}: completed run lacks clean simulator finish")
    reports = [] if log_name == "cyclotron.log" else [
        (int(cluster), int(core), int(instructions), int(cycles), float(ipc))
        for cluster, core, instructions, cycles, ipc in REPORT.findall(log)
    ]
    if log_name == "cyclotron.log":
        match = re.search(r"simulation finished after (\d+) cycles", log)
        if not match:
            raise ValueError(f"{path}: completed Cyclotron run lacks cycle count")
        gpu_cycles = int(match.group(1))
    else:
        if not reports:
            raise ValueError(f"{path}: completed RTL run lacks Muon performance reports")
        reports.sort()
        gpu_cycles = max(report[3] for report in reports)
    if result.get("gpu_cycles") != gpu_cycles:
        raise ValueError(f"{path}: result.json GPU cycles disagree with {log_name} reports")
    payload = result["logical_payload_bytes"]
    row["gpu_cycles"] = gpu_cycles
    row["core_instructions"] = ";".join(
        f"{cluster}.{core}:{instructions}" for cluster, core, instructions, _, _ in reports
    )
    row["core_ipc"] = ";".join(
        f"{cluster}.{core}:{ipc:.3f}" for cluster, core, _, _, ipc in reports
    )
    row["payload_bytes_per_cycle"] = round(payload / gpu_cycles, 6)
    if not result["kind"].startswith("stream-"):
        row["spatter_one_sided_bytes_per_cycle"] = round(payload / (2 * gpu_cycles), 6)
    if gpu_mhz is not None:
        row["payload_gbps"] = round(payload * gpu_mhz / (1000 * gpu_cycles), 6)
        if not result["kind"].startswith("stream-"):
            row["spatter_one_sided_gbps"] = round(payload * gpu_mhz / (2000 * gpu_cycles), 6)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--gpu-mhz", type=float,
                        help="measured GPU clock for optional GB/s conversion")
    args = parser.parse_args()
    if args.gpu_mhz is not None and args.gpu_mhz <= 0:
        parser.error("--gpu-mhz must be positive")
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS,
                            lineterminator="\n")
    writer.writeheader()
    for path in args.runs:
        writer.writerow(row_for(path, args.gpu_mhz))


if __name__ == "__main__":
    main()
