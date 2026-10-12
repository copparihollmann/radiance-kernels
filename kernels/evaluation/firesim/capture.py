#!/usr/bin/env python3
"""Snapshot FireSim queue evidence and check each planned UART outcome."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
HWDB_SHA256 = "d4015580a0f7d58c0cd2606832d9579981feef98f4c5c1a29d25f7ba4f3c6bc4"
PASS = re.compile(r"\*\*\* PASSED \*\*\* after (\d+) cycles")
FAIL = re.compile(r"\*\*\* FAILED \*\*\* \(code = (\d+)\) after (\d+) cycles")
DRIVER_EXIT = re.compile(r'COMMAND_EXIT_CODE="(\d+)"')


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def queue_states():
    output = subprocess.check_output(
        ["firesim-queue", "status", "--all", "--user", "agustin"], text=True
    )
    states = {}
    for line in output.splitlines():
        fields = line.split(maxsplit=7)
        if len(fields) >= 7 and fields[0].isdigit():
            states[int(fields[0])] = (fields[3], fields[4])
    return states


def snapshot(job_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "runworkload-full.json": job_dir / "runworkload-full.json",
        "stdout.log": job_dir / "stdout.log",
        "stderr.log": job_dir / "stderr.log",
        "uartlog": job_dir / "simulation/sim_slot_0/uartlog",
        "memory_stats0.csv": job_dir / "simulation/sim_slot_0/memory_stats0.csv",
        "heartbeat.csv": job_dir / "simulation/sim_slot_0/heartbeat.csv",
        "latency_histogram.csv": job_dir / "simulation/sim_slot_0/latency_histogram.csv",
    }
    for name, path in files.items():
        if path.is_file():
            shutil.copy2(path, output_dir / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-root", type=Path, default=Path("/scratch/firesim_queue/jobs"))
    parser.add_argument("--elf-root", type=Path,
                        default=Path("/scratch/agustin/projects/chipyard/firesim-hpc-inputs"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    states = queue_states()
    records = []
    problems = []
    with (HERE / "plan.csv").open(newline="") as stream:
        plan = list(csv.DictReader(stream))
    by_case = {row["case"]: row for row in plan}
    original = args.elf_root / by_case["xrage9-ordered"]["elf_snapshot_path"]
    guard_only = args.elf_root / by_case["xrage9-guard-diagnostic"]["elf_snapshot_path"]
    if original.exists() and guard_only.exists():
        before, after = original.read_bytes(), guard_only.read_bytes()
        offset = 0x1774
        if (len(before) != len(after) or before[:offset] != after[:offset]
                or before[offset:offset + 4] != bytes.fromhex("638ce700")
                or after[offset:offset + 4] != bytes.fromhex("6f008001")
                or before[offset + 4:] != after[offset + 4:]):
            problems.append("xRAGE9 guard-only diagnostic changes more than the host sample branch")
    elif args.require_complete:
        problems.append("xRAGE9 original or guard-only diagnostic ELF is missing")
    for case in plan:
        job_id = int(case["job_id"])
        queue_state, queue_phase = states.get(job_id, ("unknown", "unknown"))
        if args.require_complete and queue_state != "DONE":
            problems.append(f"{job_id}: queue did not finish cleanly ({queue_state}/{queue_phase})")
        job_dir = args.jobs_root / str(job_id)
        manifest = job_dir / "runworkload-full.json"
        if manifest.exists():
            job = json.loads(manifest.read_text())
            if job["job_id"] != job_id or job["hwdb_config_artifact_sha256"] != HWDB_SHA256:
                problems.append(f"{job_id}: queue manifest or HWDB hash changed")
            source = Path(job["stage_from"])
            if args.require_complete and not source.exists():
                problems.append(f"{job_id}: staged source ELF is missing")
            elif source.exists() and sha256(source) != case["elf_sha256"]:
                problems.append(f"{job_id}: staged source ELF hash differs from plan")
            staged = (Path(job["chipyard"]) / "sims/firesim/deploy/workloads" /
                      job["workload"] / job["bootbinary"])
            if args.require_complete and not staged.exists():
                problems.append(f"{job_id}: FireSim workload ELF is missing")
            elif staged.exists() and sha256(staged) != case["elf_sha256"]:
                problems.append(f"{job_id}: FireSim workload ELF hash differs from plan")
        local_elf = args.elf_root / case["elf_snapshot_path"]
        if args.require_complete and not local_elf.exists():
            problems.append(f"{job_id}: local archive ELF is missing")
        elif local_elf.exists() and sha256(local_elf) != case["elf_sha256"]:
            problems.append(f"{job_id}: archived ELF hash differs from plan")
        uart = job_dir / "simulation/sim_slot_0/uartlog"
        observed = "pending"
        code = ""
        cycles = ""
        driver_exit = ""
        uart_sha = ""
        if uart.exists():
            content = uart.read_text(errors="replace")
            uart_sha = sha256(uart)
            passing = PASS.search(content)
            failing = FAIL.search(content)
            exit_match = DRIVER_EXIT.search(content)
            if passing and not failing:
                observed, cycles = "pass", passing.group(1)
            elif failing and not passing:
                observed, code, cycles = "fail", failing.group(1), failing.group(2)
            if exit_match:
                driver_exit = exit_match.group(1)
        if observed == "pending" and queue_state in {"DONE", "FAILED"}:
            validation = "infrastructure-failed"
        elif observed == "pending":
            validation = "pending"
        elif observed == "fail" and case["expected_uart"] == "pass" and driver_exit == "1":
            validation = (("diagnostic-failed" if case["group"] == "diagnostic"
                           else "workload-failed") if queue_state == "DONE" else "pending")
        elif observed == case["expected_uart"] and driver_exit == ("0" if observed == "pass" else "1"):
            if queue_state == "DONE":
                validation = "passed" if observed == "pass" else "expected-control-failure"
            elif queue_state == "FAILED":
                validation = "infrastructure-failed"
            else:
                validation = "pending"
        else:
            validation = "mismatch"
            problems.append(f"{job_id}: expected {case['expected_uart']}, saw {observed}, driver exit {driver_exit}")
        if queue_state in {"DONE", "FAILED"} or validation in {"passed", "expected-control-failure", "workload-failed"}:
            snapshot(job_dir, HERE / "raw" / str(job_id))
        records.append({**case, "queue_state": queue_state, "queue_phase": queue_phase,
                        "observed_uart": observed, "guest_failure_code": code,
                        "driver_exit_code": driver_exit, "target_cycles": cycles,
                        "validation": validation, "uart_sha256": uart_sha})
    with (HERE / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    artifacts = []
    for path in sorted((HERE / "raw").rglob("*")):
        if path.is_file():
            artifacts.append({"path": str(path.relative_to(HERE)),
                              "bytes": path.stat().st_size, "sha256": sha256(path)})
    with (HERE / "artifacts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(artifacts)
    print(f"{sum(r['validation'] in ('passed', 'expected-control-failure') for r in records)} "
          f"validated, {sum(r['validation'] == 'pending' for r in records)} pending, "
          f"{sum(r['validation'] == 'workload-failed' for r in records)} workload failures, "
          f"{sum(r['validation'] == 'diagnostic-failed' for r in records)} diagnostic failures, "
          f"{sum(r['validation'] == 'infrastructure-failed' for r in records)} infrastructure failures")
    if args.require_complete and any(r["validation"] in ("pending", "infrastructure-failed") for r in records):
        problems.append("not all planned jobs produced a UART result")
    if problems:
        raise SystemExit("\n".join(problems))


if __name__ == "__main__":
    main()
