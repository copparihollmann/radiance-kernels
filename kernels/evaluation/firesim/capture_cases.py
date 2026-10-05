#!/usr/bin/env python3
"""Capture a planned U250 case series with ELF and UART verification."""

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from capture import HWDB_SHA256, queue_states, sha256, snapshot  # noqa: E402
from splice_device import device_image  # noqa: E402

PASS = re.compile(r"\*\*\* PASSED \*\*\* after (\d+) cycles")
FAIL = re.compile(r"\*\*\* FAILED \*\*\* \(code = (\d+)\) after (\d+) cycles")
EXIT = re.compile(r'COMMAND_EXIT_CODE="(\d+)"')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs-root", type=Path, default=Path("/scratch/firesim_queue/jobs"))
    parser.add_argument("--elf-root", type=Path,
                        default=Path("/scratch/agustin/projects/chipyard/firesim-hpc-inputs"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with args.plan.open(newline="") as stream:
        plan = list(csv.DictReader(stream))
    states = queue_states()
    results = []
    problems = []
    for row in plan:
        job_id = int(row["job_id"])
        job_dir = args.jobs_root / str(job_id)
        source = args.elf_root / row["elf_path"]
        if not source.is_file() or sha256(source) != row["elf_sha256"]:
            problems.append(f"{job_id}: source ELF missing or changed")
            continue
        build = json.loads((source.parent / "result.json").read_text())
        if (str(build.get("expected_digest")) != row["expected_digest"] or
                str(build.get("count")) != row["count"] or
                str(build.get("original_count")) != row["original_count"]):
            problems.append(f"{job_id}: build manifest differs from plan")
        if row["original_device_elf"]:
            reference = args.elf_root / row["original_device_elf"]
            if not reference.is_file() or device_image(source) != device_image(reference):
                problems.append(f"{job_id}: RV32 device LOAD image differs from original")
        request_file = job_dir / "runworkload-full.json"
        if not request_file.is_file():
            problems.append(f"{job_id}: queue request missing")
            continue
        request = json.loads(request_file.read_text())
        if (request["job_id"] != job_id or
                request["hwdb_config_artifact_sha256"] != HWDB_SHA256 or
                Path(request["stage_from"]).resolve() != source.resolve()):
            problems.append(f"{job_id}: queue request differs from plan")
        state, phase = states.get(job_id, ("unknown", "unknown"))
        uart_file = job_dir / "simulation/sim_slot_0/uartlog"
        observed, cycles, code, driver_exit, uart_hash = "pending", "", "", "", ""
        if uart_file.is_file():
            uart = uart_file.read_text(errors="replace")
            passing, failing, exit_match = PASS.search(uart), FAIL.search(uart), EXIT.search(uart)
            uart_hash = sha256(uart_file)
            if passing and not failing:
                observed, cycles = "pass", passing.group(1)
            elif failing and not passing:
                observed, code, cycles = "fail", failing.group(1), failing.group(2)
            if exit_match:
                driver_exit = exit_match.group(1)
        outcome_ok = (observed == row["expected_uart"] and
                      driver_exit == ("0" if observed == "pass" else "1"))
        if state == "DONE" and outcome_ok:
            validation = "passed" if observed == "pass" else "expected-diagnostic-failure"
        elif outcome_ok and state not in ("DONE", "FAILED"):
            validation = "pending"
        elif state in ("DONE", "FAILED") and observed == "pending":
            validation = "infrastructure-failed"
        elif observed == "pending":
            validation = "pending"
        else:
            validation = "mismatch"
            problems.append(f"{job_id}: expected {row['expected_uart']}, got {observed}/{driver_exit}")
        if observed != "pending" or state in ("DONE", "FAILED"):
            raw = out / "raw" / str(job_id)
            snapshot(job_dir, raw)
            shutil.copy2(source.parent / "result.json", raw / "build-result.json")
            shutil.copy2(source.parent / "build.log", raw / "build.log")
            sample_comparison = source.parent / "sample-comparison.json"
            if sample_comparison.is_file():
                shutil.copy2(sample_comparison, raw / "sample-comparison.json")
        results.append({**row, "queue_state": state, "queue_phase": phase,
                        "observed_uart": observed, "guest_failure_code": code,
                        "driver_exit_code": driver_exit, "target_cycles": cycles,
                        "validation": validation, "uart_sha256": uart_hash})
    if results:
        with (out / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=results[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(results)
    artifacts = []
    for path in sorted((out / "raw").rglob("*")):
        if path.is_file():
            artifacts.append({"path": path.relative_to(out).as_posix(),
                              "bytes": path.stat().st_size, "sha256": sha256(path)})
    with (out / "artifacts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(artifacts)
    print(f"{len(results)} rows: " + ", ".join(f"{r['job_id']}={r['validation']}" for r in results))
    if args.require_complete:
        if any(r["validation"] == "pending" for r in results):
            problems.append("planned jobs are still pending")
        for case in {r["case"] for r in results}:
            attempts = [r for r in results if r["case"] == case]
            if not any(r["validation"] in ("passed", "expected-diagnostic-failure")
                       for r in attempts):
                problems.append(f"{case}: no completed guest check")
    if problems:
        raise SystemExit("\n".join(problems))


if __name__ == "__main__":
    main()
