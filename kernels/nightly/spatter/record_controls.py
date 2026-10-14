#!/usr/bin/env python3
"""Summarize full-output Cyclotron controls without treating cycles as performance."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
UPSTREAM = ROOT / "kernels/spatter/evaluation/upstream-golden-results.csv"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--standard", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with UPSTREAM.open(newline="") as stream:
        oracle = {row["run"]: row for row in csv.DictReader(stream)}
    groups = {}
    for name, path, ncases in (("smoke", args.smoke, 6),
                               ("standard_gpu_stream", args.standard, 5)):
        records = json.loads(path.read_text())
        if {(item["case"], item["logical_partitions"]) for item in records} != {
                (case, partitions) for case in range(ncases)
                for partitions in (1, 2)}:
            raise ValueError(f"incomplete or duplicate {name} controls")
        for item in records:
            if (item["status"] != "passed" or
                    item["exact_output_bytes"] != item["output_elements"] * 8 or
                    item["output_sha256"] != item["reference_sha256"] or
                    item["rtl_execution"] or item["performance_measurement"] or
                    item["two_sm_execution"] or
                    item["spatter_body_sha256"] != sha256(HERE / "spatter_body.hpp")):
                raise ValueError(f"failed or stale {name} control: {item['case']}")
            if name == "standard_gpu_stream":
                prior = oracle[f"rebuild-gpu-stream-{item['case']}"]
                if (item["reference_sha256"] != prior["golden_output_sha256"]
                        or item["kind"] != prior["kind"]):
                    raise ValueError("standard control differs from upstream oracle")
        groups[name] = {"case_count": ncases, "runs": len(records),
                        "raw_results_sha256": sha256(path), "cases": records}
    output = {"schema": "radiance.nightly.spatter.device-controls.v1",
              "backend": "Cyclotron functional ISA simulation",
              "scope": "shared device body; two partitions run sequentially on one simulated cluster",
              "pinned_upstream_oracle_sha256": sha256(UPSTREAM),
              "all_passed": True, "rtl_execution": False,
              "performance_measurement": False, "groups": groups}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print("recorded 12 smoke and 10 original-size complete-output controls")


if __name__ == "__main__":
    main()
