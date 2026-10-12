#!/usr/bin/env python3
"""Summarize Cyclotron memory transactions for completed Radiance kernel runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


FIELDS = ("run", "kind", "gpu_cycles", "logical_payload_bytes",
          "model_gmem_bytes_issued", "model_gmem_transactions",
          "model_gmem_bytes_per_cycle", "model_lsu_global_loads_issued",
          "model_lsu_global_stores_issued", "l0_accesses", "l0_hits",
          "gmem_queue_full_rejects")


def row_for(path: Path) -> dict:
    result = json.loads((path / "result.json").read_text())
    if result["status"] not in ("passed", "exploratory"):
        raise ValueError(f"{path}: Cyclotron run is not complete")
    summaries = list((path / "performance_logs").glob("*/summary.json"))
    matches = []
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        elapsed = max(core["scheduler"]["cycles"] for core in summary["per_core"])
        if abs(elapsed - result["gpu_cycles"]) <= 1:
            matches.append(summary["total"])
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one matching Cyclotron summary, found {len(matches)}")
    total = matches[0]
    gmem = total["gmem_stats"]
    lsu = total["lsu_stats"]
    hits = total["gmem_hits"]
    cycles = result["gpu_cycles"]
    if gmem["issued"] != gmem["completed"]:
        raise ValueError(f"{path}: GPU memory transactions did not drain")
    if (lsu["global_ldq_issued"] != lsu["global_ldq_completed"] or
            lsu["global_stq_issued"] != lsu["global_stq_completed"] or
            sum(lsu[f"{queue}_issued"] for queue in
                ("global_ldq", "global_stq", "shared_ldq", "shared_stq")) != lsu["issued"]):
        raise ValueError(f"{path}: GPU LSU queue counts are inconsistent")
    return dict(run=path.name, kind=result["kind"], gpu_cycles=cycles,
                logical_payload_bytes=result["logical_payload_bytes"],
                model_gmem_bytes_issued=gmem["bytes_issued"],
                model_gmem_transactions=gmem["issued"],
                model_gmem_bytes_per_cycle=round(gmem["bytes_issued"] / cycles, 6),
                model_lsu_global_loads_issued=lsu["global_ldq_issued"],
                model_lsu_global_stores_issued=lsu["global_stq_issued"],
                l0_accesses=hits["l0_accesses"], l0_hits=hits["l0_hits"],
                gmem_queue_full_rejects=gmem["queue_full_rejects"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS,
                            lineterminator="\n")
    writer.writeheader()
    for path in args.runs:
        writer.writerow(row_for(path))


if __name__ == "__main__":
    main()
