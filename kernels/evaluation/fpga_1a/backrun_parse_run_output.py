#!/usr/bin/env python3
"""Back-run parse_run_output.py over an archived campaign corpus.

Usage:
    backrun_parse_run_output.py RUNS_ROOT --csv OUT.csv [--jsonl OUT.jsonl]
                                [--sidecar/--no-sidecar] [--coverage OUT.json]
                                [--quiet-seconds N]

Walks RUNS_ROOT for every directory that holds a ``result.json``, parses its
sibling ``stdout.log`` (and ``uartlog``) with parse_run_output.parse_job_dir,
and reports what came out.

It never touches an existing ``result.json``, ``plan.json``, ELF or manifest.
The only thing it may write inside the corpus is a NEW ``parsed.json`` side car
next to each ``result.json``; ``--no-sidecar`` turns even that off.  Job
directories whose ``result.json`` was modified within ``--quiet-seconds``
(default 900) are skipped entirely, so a live campaign's in-flight directories
are left alone.

Also cross-checks the per-core rows this parser extracts against the
``core_kernel_cycles`` campaign.py already records, and reports every
disagreement.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_run_output import parse_job_dir  # noqa: E402

# Lists that are kept whole in parsed.json / the JSONL but not exploded into
# CSV columns (a 46-row llama phase table would add hundreds of columns).
BULK_LISTS = ("core_cycles", "cyclotron_cluster_cycles", "gmem_checks",
              "phase_table", "bf16_checks", "byte_checks", "per_op", "sources",
              "compute_launches", "phase_stamps")

RESULT_COLUMNS = ("id", "status", "returncode", "kernel_cycles",
                  "per_sm_timing_complete", "wall_seconds", "parser_error")

# parsed["compute"] key -> CSV column.  The column is the name the device
# actually printed, so a CSV cell can be grepped straight out of a stdout.log;
# keys not listed here already carry that name.
COMPUTE_CSV_COLUMNS = {
    "op_unit": "compute_op_unit",
    "datapath": "compute_datapath",
    "util_bp": "compute_util_bp",
    "util_pct": "compute_util_pct",
    "window_cycles": "compute_window_cycles",
}


def flatten(parsed: dict) -> dict:
    """One flat row of scalars for the CSV."""
    row: dict = {}
    for key, value in parsed.items():
        if key in BULK_LISTS or key == "schema":
            continue
        if key == "model_geometry" and isinstance(value, dict):
            for sub, item in value.items():
                row[f"geometry_{sub}"] = item
            continue
        if key == "compute" and isinstance(value, dict):
            for sub, item in value.items():
                row[COMPUTE_CSV_COLUMNS.get(sub, sub)] = item
            continue
        if key == "compute_detail" and isinstance(value, dict):
            # Already namespaced per kernel (btree_lookups, spmv_executed_macs)
            # and disjoint across kernels, so carried through unprefixed.
            row.update(value)
            continue
        if key == "parse_errors":
            row["parse_errors"] = "; ".join(str(v) for v in value)
            continue
        if isinstance(value, (dict, list)):
            continue
        row[key] = value
    row["sources"] = "+".join(parsed.get("sources", []))
    for name, count in (("core_cycles", "core_cycles_rows"),
                        ("cyclotron_cluster_cycles", "cyclotron_cluster_rows"),
                        ("gmem_checks", "gmem_checks_rows"),
                        ("bf16_checks", "bf16_checks_rows"),
                        ("byte_checks", "byte_checks_rows")):
        if name in parsed:
            row[count] = len(parsed[name])
    for op in parsed.get("per_op", []):
        for field in ("count", "cycles", "share_pct", "mesh_util_pct"):
            if field in op:
                row[f"op_{op['op']}_{field}"] = op[field]
    # hotspot3d's four host-sequenced launches: kept whole in parsed.json and
    # exploded here, since four launches x four fields is a bounded widening
    # and the per-launch cycle split is the point of the field.
    for launch in parsed.get("compute_launches", []):
        for field, value in launch.items():
            if field != "index":
                row[f"launch{launch['index']}_{field}"] = value
    for stamp in parsed.get("phase_stamps", []):
        row[f"phase_stamp_c{stamp['cluster']}_{stamp['index']}"] = \
            stamp["cycles"]
    return row


def core_cross_check(parsed: dict, result: dict) -> tuple[str, str]:
    """Compare parsed per-core rows with result.json's core_kernel_cycles."""
    mine = parsed.get("core_cycles")
    theirs = result.get("core_kernel_cycles")
    if mine is None and not theirs:
        return "both-absent", ""
    if mine is None:
        return "disagree", "parser found no per-core rows, result.json has " \
                           f"{len(theirs)}"
    if theirs is None:
        return "disagree", "result.json has no core_kernel_cycles"

    def key(row):
        return (row["cluster"], row["core"], row["start"], row["end"],
                row["cycles"])

    a = sorted(key(r) for r in mine)
    b = sorted(key(r) for r in theirs)
    if a == b:
        return "agree", ""
    only_mine = [r for r in a if r not in b]
    only_theirs = [r for r in b if r not in a]
    return "disagree", (f"parser-only={only_mine[:4]} "
                        f"result-only={only_theirs[:4]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_root")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--jsonl")
    ap.add_argument("--coverage")
    ap.add_argument("--sidecar", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--quiet-seconds", type=float, default=900.0,
                    help="skip job dirs whose result.json was modified more "
                         "recently than this (default 900)")
    args = ap.parse_args(argv)

    root = Path(args.runs_root).resolve()
    now = time.time()

    job_dirs: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "result.json" in filenames:
            job_dirs.append(Path(dirpath))
    job_dirs.sort()

    rows: list[dict] = []
    records: list[dict] = []
    coverage: dict[str, int] = {}
    disagreements: list[dict] = []
    skipped_recent: list[str] = []
    unreadable_result: list[str] = []
    no_log = 0
    sidecars = 0

    for job in job_dirs:
        relative = job.relative_to(root).as_posix()
        result_path = job / "result.json"
        try:
            age = now - result_path.stat().st_mtime
        except OSError:
            age = float("inf")
        if age < args.quiet_seconds:
            skipped_recent.append(relative)
            continue
        try:
            result = json.loads(result_path.read_text())
            if not isinstance(result, dict):
                raise ValueError("result.json is not an object")
        except (OSError, ValueError) as error:
            unreadable_result.append(f"{relative}: {error}")
            result = {}

        parsed = parse_job_dir(job)
        if not parsed.get("sources"):
            no_log += 1
        for key in parsed:
            if key != "schema":
                coverage[key] = coverage.get(key, 0) + 1

        verdict, detail = core_cross_check(parsed, result)
        if verdict == "disagree":
            disagreements.append({"job": relative, "detail": detail})

        row = {"job_dir": relative}
        for column in RESULT_COLUMNS:
            if column in result:
                row[f"result_{column}"] = result[column]
        row["core_cycles_cross_check"] = verdict
        if detail:
            row["core_cycles_cross_check_detail"] = detail
        row.update(flatten(parsed))
        rows.append(row)
        records.append({"job_dir": relative, "parsed": parsed})

        if args.sidecar:
            try:
                (job / "parsed.json").write_text(
                    json.dumps(parsed, indent=2, sort_keys=True) + "\n")
                sidecars += 1
            except OSError as error:
                print(f"sidecar failed for {relative}: {error}",
                      file=sys.stderr)

    columns: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    lead = [c for c in ("job_dir",) if c in seen]
    columns = lead + sorted(c for c in columns if c not in lead)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "schema": "radiance-fpga-1a-run-output-backrun-v1",
        "runs_root": str(root),
        "job_dirs_found": len(job_dirs),
        "job_dirs_parsed": len(rows),
        "job_dirs_skipped_recent": skipped_recent,
        "job_dirs_without_console_log": no_log,
        "unreadable_result_json": unreadable_result,
        "sidecars_written": sidecars,
        "field_coverage": dict(sorted(coverage.items())),
        "core_cycles_disagreements": disagreements,
        "csv": str(csv_path),
        "csv_columns": len(columns),
    }
    if args.coverage:
        coverage_path = Path(args.coverage)
        coverage_path.parent.mkdir(parents=True, exist_ok=True)
        coverage_path.write_text(json.dumps(summary, indent=2) + "\n")
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
