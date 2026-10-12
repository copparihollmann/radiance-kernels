#!/usr/bin/env python3
"""Capture planned U250 host intervals as their queue jobs finish."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from capture import queue_states, snapshot  # noqa: E402


def write_results(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE / "plan.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs-root", type=Path, default=Path("/scratch/firesim_queue/jobs"))
    parser.add_argument("--watch", action="store_true", help="poll until every job is terminal")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    with args.plan.open(newline="") as stream:
        plan = list(csv.DictReader(stream))
    if not plan or len({r["job_id"] for r in plan}) != len(plan):
        parser.error("plan is empty or has duplicate job IDs")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    while True:
        states = queue_states()
        rows = []
        for item in plan:
            job_id = int(item["job_id"])
            state, phase = states.get(job_id, ("unknown", "unknown"))
            case_out = out / str(job_id)
            status = "pending"
            if (case_out / "result.json").is_file():
                status = "captured"
            elif state == "DONE":
                command = [sys.executable, str(HERE.parent / "host_interval.py"),
                           "--job", str(job_id), "--elf", item["timed_elf"],
                           "--untimed-elf", item["untimed_elf"],
                           "--plan", str(args.plan.resolve()),
                           "--jobs-root", str(args.jobs_root.resolve()),
                           "--out", str(case_out)]
                result = subprocess.run(command, text=True, capture_output=True, check=False)
                if result.returncode == 0:
                    status = "captured"
                else:
                    status = "capture-failed"
                    case_out.mkdir(parents=True, exist_ok=True)
                    (case_out / "capture-error.txt").write_text(result.stdout + result.stderr)
                    snapshot(args.jobs_root / str(job_id), case_out / "raw")
            elif state in ("FAILED", "TIMEOUT", "CANCELLED"):
                status = "queue-failed"
                snapshot(args.jobs_root / str(job_id), case_out / "raw")
            rows.append({"job_id": job_id, "case": item["case"],
                         "queue_state": state, "queue_phase": phase,
                         "capture_status": status})
        write_results(out / "results.csv", rows)
        print(" ".join(f"{r['job_id']}={r['capture_status']}" for r in rows), flush=True)
        if not args.watch or all(r["capture_status"] != "pending" for r in rows):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
