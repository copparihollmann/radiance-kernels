#!/usr/bin/env python3
"""Join the FireSim UART checks with the existing RTL and model results."""

import argparse
import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
KERNELS = EVAL.parent


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def issue_slot_use(row):
    counts = [int(part.split(":")[1]) for part in row["core_instructions"].split(";")]
    return f"{sum(counts) / (2 * int(row['gpu_cycles'])):.6f}"


def build_rows():
    plan = {row["case"]: row for row in read_csv(HERE / "plan.csv")}
    firesim = {row["case"]: row for row in read_csv(HERE / "results.csv")}
    model = {row["run"]: row for row in read_csv(EVAL / "workload-results.csv")}
    composition_model = {
        row["run"]: row for row in read_csv(KERNELS / "spatter/evaluation/composition-fullsize-model-results.csv")
    }
    rtl = {}
    for rel in ("stream/evaluation/current-build-results.csv",
                "spatter/evaluation/current-build-results.csv",
                "spatter/evaluation/composition-fullsize-rtl-results.csv"):
        for row in read_csv(KERNELS / rel):
            if row["simulator"].startswith("Verilator") and row["status"] == "passed":
                rtl[row["run"]] = row
    inventory = {row["snapshot_path"]: row for row in read_csv(EVAL / "elf-snapshot.csv")}
    mapping = [
        ("stream-copy", "copy-1048576", "copy-1048576", "not-applicable"),
        ("stream-scale", "scale-1048576", "scale-1048576", "not-applicable"),
        ("stream-add", "add-1048576", "add-1048576", "not-applicable"),
        ("stream-triad", "triad-1048576", "triad-1048576", "not-applicable"),
        ("gpu-stream-gather", "rebuild-gpu-stream-0", "gpu-stream-0", "passed"),
        ("gpu-stream-scatter", "rebuild-gpu-stream-1", "rebuild-gpu-stream-1", "passed"),
        ("gpu-stream-gs", "rebuild-gpu-stream-2", "rebuild-gpu-stream-2", "passed"),
        ("gpu-stream-multiscatter", "rebuild-gpu-stream-3", "rebuild-gpu-stream-3", "passed"),
        ("gpu-stream-multigather", "rebuild-gpu-stream-4", "rebuild-gpu-stream-4", "passed"),
        ("xrage5-gather", "xrage5", None, "passed"),
        ("xrage9-parallel", "rebuild-xrage9", None, "no-deterministic-golden"),
        ("xrage9-ordered", "xrage9-ordered", None, "passed-serial-order"),
        ("lulesh-gather", "lulesh-gather", None, "passed"),
        ("lulesh-scatter-ordered", "lulesh-scatter-ordered", None, "passed-serial-order"),
        ("amg-scaled", "amg-gpu-scaled-1024", None, "passed-scaled"),
        ("chain-materialized", "chain-gpu-stream-no-fence", "chain-gpu-stream-no-fence", "passed"),
        ("chain-fused", "composed-gpu-stream-current", "composed-gpu-stream-current", "not-directly-run"),
    ]
    rows = []
    for case, model_run, rtl_run, golden in mapping:
        planned = plan[case]
        fpga = firesim[case]
        m = model.get(model_run, composition_model.get(model_run))
        expected_model_status = "exploratory" if case == "xrage9-parallel" else "passed"
        if m is None or m["status"] != expected_model_status:
            raise ValueError(f"missing {expected_model_status} model row: {model_run}")
        if case in ("chain-materialized", "chain-fused"):
            model_elf = inventory.get(
                f"elf-snapshot/spatter-current/model/{model_run}/kernel.soc.elf"
            )
            if model_elf is None or model_elf["sha256"] != planned["elf_sha256"]:
                raise ValueError(f"model and FireSim ELF hashes disagree: {case}")
        elif m["elf_sha256"] != planned["elf_sha256"]:
            raise ValueError(f"model and FireSim ELF hashes disagree: {case}")
        r = rtl.get(rtl_run) if rtl_run else None
        if rtl_run and r is None:
            raise ValueError(f"missing passed RTL row: {rtl_run}")
        rows.append({
            "case": case,
            "group": planned["group"],
            "size": planned["size"],
            "logical_payload_bytes": m["logical_payload_bytes"],
            "rtl_gpu_cycles": r["gpu_cycles"] if r else "",
            "rtl_output_check": r["correctness"] if r else "not-run",
            "rtl_issue_slot_use": issue_slot_use(r) if r else "",
            "model_gpu_cycles": m["gpu_cycles"],
            "model_status": m["status"],
            "model_gmem_bytes_issued": m.get("model_gmem_bytes_issued", ""),
            "model_gmem_bytes_per_cycle": m.get("model_gmem_bytes_per_cycle", ""),
            "upstream_spatter_golden": golden,
            "firesim_uart": fpga["observed_uart"],
            "firesim_host_check": planned["host_check"],
            "firesim_target_cycles_whole_program": fpga["target_cycles"],
            "firesim_job_id": fpga["job_id"],
            "elf_sha256": planned["elf_sha256"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare with the saved CSV")
    args = parser.parse_args()
    rows = build_rows()
    if args.check:
        if read_csv(HERE / "comparison.csv") != rows:
            raise SystemExit("comparison.csv differs from the source results")
        print(f"verified {len(rows)} representative workload rows")
    else:
        with (HERE / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} representative workload rows")


if __name__ == "__main__":
    main()
