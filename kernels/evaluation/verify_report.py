#!/usr/bin/env python3
"""Check the published workload table against completed model run CSVs."""

from __future__ import annotations

import csv
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re


KERNELS = Path(__file__).resolve().parents[1]
TOOLS = KERNELS / "spatter/tools"
CHECK = re.compile(
    r"SPATTER_CHECK digest=([0-9a-f]{16}) expected=(\w+) "
    r"guards_intact=(\w+) nonzero_words=(\d+) output_elements=(\d+)"
)
ROWS = {
    "STREAM Copy": ("stream", "copy-1048576", "passed"),
    "STREAM Scale": ("stream", "scale-1048576", "passed"),
    "STREAM Add": ("stream", "add-1048576", "passed"),
    "STREAM Triad": ("stream", "triad-1048576", "passed"),
    "Spatter GPU STREAM Gather": ("spatter", "rebuild-gpu-stream-0", "passed"),
    "Spatter GPU STREAM Scatter": ("spatter", "rebuild-gpu-stream-1", "passed"),
    "Spatter GPU STREAM GS": ("spatter", "rebuild-gpu-stream-2", "passed"),
    "Spatter GPU STREAM MultiScatter": ("spatter", "rebuild-gpu-stream-3", "passed"),
    "Spatter GPU STREAM MultiGather": ("spatter", "rebuild-gpu-stream-4", "passed"),
    "xRAGE asteroid pattern 5, Gather": ("spatter", "xrage5", "passed"),
    "xRAGE asteroid pattern 9, parallel Scatter":
        ("spatter", "rebuild-xrage9", "exploratory"),
    "xRAGE asteroid pattern 9, ordered Scatter":
        ("spatter", "xrage9-ordered", "passed"),
    "LULESH app trace case 1, Gather": ("spatter", "lulesh-gather", "passed"),
    "LULESH app trace case 3, ordered Scatter":
        ("spatter", "lulesh-scatter-ordered", "passed"),
    "AMG GPU Gather, scaled to 1,024 repetitions":
        ("spatter", "amg-gpu-scaled-1024", "passed"),
}
REPORT_CSV = KERNELS / "evaluation/workload-results.csv"
REPORT_FIELDS = (
    "workload", "family", "run", "simulator", "status", "correctness", "elements_or_transfers",
    "gpu_cycles", "logical_payload_bytes", "payload_bytes_per_cycle",
    "model_gmem_bytes_issued", "model_gmem_transactions",
    "model_gmem_bytes_per_cycle", "model_lsu_global_loads_issued",
    "model_lsu_global_stores_issued", "output_digest", "expected_digest",
    "elf_sha256", "kernel_source_sha256", "suite_sha256",
    "timing_config_sha256", "checker_source_sha256",
)
RTL_CASES = {
    "spatter": ("gpu-stream-0", "rebuild-gpu-stream-1", "rebuild-gpu-stream-2",
                "rebuild-gpu-stream-3", "rebuild-gpu-stream-4"),
    "stream": ("copy-1048576", "scale-1048576", "add-1048576", "triad-1048576"),
}
SMALL_RTL_TABLES = (
    ("spatter", "current-build-smoke-results.csv", "rtl-single",
     ("default", "smoke-1", "smoke-2", "smoke-3", "smoke-4", "smoke-5",
      "composed", "composed-wrap", "ordered-overlap", "materialized-chain")),
    ("stream", "current-build-smoke-results.csv", "rtl",
     ("copy-256", "scale-256", "add-256", "triad-256")),
    ("spatter", "app-trace-smoke-results.csv", "rtl",
     ("lulesh-gather-smoke", "lulesh-scatter-smoke")),
    ("spatter", "composition-ordered-pair.csv", "rtl",
     ("materialized-chain-ordered", "composed-ordered")),
    ("spatter", "mt8-smoke-pair.csv", "rtl-mt-probe", ("smoke-1",)),
    ("spatter", "mt16-smoke-pair.csv", "rtl-mt16-probe", ("smoke-1",)),
)


def indexed_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    indexed = {row["run"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate run in {path}")
    return indexed


def number(value: str) -> int:
    return int(value.replace(",", ""))


def load_tool(filename: str):
    spec = importlib.util.spec_from_file_location(filename, TOOLS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_runs(records: dict) -> None:
    summarize = load_tool("spatter-summarize.py")
    memory_summary = load_tool("spatter-memory-summary.py")
    config_hash = hash_file(TOOLS / "spatter-config.toml")
    checker_hash = hash_file(TOOLS / "spatter_check.rs")
    for (family, run), (cycle, memory) in records.items():
        build_path = KERNELS / family / "runs" / run
        model_path = KERNELS / family / "runs/model" / run
        build = json.loads((build_path / "result.json").read_text())
        model = json.loads((model_path / "result.json").read_text())
        status = "exploratory" if (family, run) == ("spatter", "rebuild-xrage9") else "passed"
        if model["status"] != status or model["correctness"] != (
                "guards-and-nonzero-exploratory" if status == "exploratory"
                else "digest-checked"):
            raise ValueError(f"{family}/{run}: unsupported completion status")
        if (model["timing_config_sha256"] != config_hash or
                model["checker_source_sha256"] != checker_hash):
            raise ValueError(f"{family}/{run}: retained timing config or checker differs")
        for field in ("suite_sha256", "kernel_source_sha256", "expected_digest",
                      "case", "address_plan"):
            if build[field] != model[field]:
                raise ValueError(f"{family}/{run}: build/model {field} differs")
        elf_hash = hash_file(build_path / "kernel.soc.elf")
        if model["elf_sha256"] != elf_hash or hash_file(
                model_path / "kernel.soc.elf") != elf_hash:
            raise ValueError(f"{family}/{run}: model ELF differs from build")
        log = (model_path / "cyclotron.log").read_text(errors="replace")
        check = CHECK.search(log)
        if (not check or check.group(3) != "true" or int(check.group(4)) == 0 or
                int(check.group(5)) != model["output_elements"] or
                check.group(1) != model["output_digest"]):
            raise ValueError(f"{family}/{run}: complete GPU output readback differs")
        if status == "passed" and (
                check.group(1) != model["expected_digest"] or
                check.group(2) != model["expected_digest"]):
            raise ValueError(f"{family}/{run}: full output digest differs")
        if status == "exploratory" and check.group(2) != "none":
            raise ValueError(f"{family}/{run}: racing output called deterministic")
        actual_cycle = summarize.row_for(model_path, None)
        actual_memory = memory_summary.row_for(model_path)
        for label, reported, actual in (("cycle", cycle, actual_cycle),
                                        ("memory", memory, actual_memory)):
            if reported != {field: str(value) for field, value in actual.items()}:
                raise ValueError(f"{family}/{run}: {label} CSV differs from raw run")


def verify_paired_rtl_tables() -> int:
    validate = load_tool("spatter-validate.py")
    summarize = load_tool("spatter-summarize.py")
    verified = 0
    for family, rtl_directory in (("spatter", "rtl"), ("stream", "rtl-full")):
        table = KERNELS / family / "evaluation/current-build-results.csv"
        with table.open(newline="") as source:
            reported = list(csv.DictReader(source))
        if len(reported) % 2:
            raise ValueError(f"{table}: incomplete RTL/model pairs")
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=summarize.FIELDS, lineterminator="\n")
        writer.writeheader()
        root = KERNELS / family / "runs"
        included = []
        for position in range(0, len(reported), 2):
            rtl_row, model_row = reported[position:position + 2]
            name = rtl_row["run"]
            included.append(name)
            if (model_row["run"] != name or
                    not rtl_row["simulator"].startswith("Verilator") or
                    model_row["simulator"] != "Cyclotron timing model"):
                raise ValueError(f"{table}: invalid RTL/model pair at row {position + 2}")
            rtl_path = root / rtl_directory / name
            log = (rtl_path / "verilator.log").read_text(errors="replace")
            cycles = [int(value) for value in re.findall(r"\bCycles:\s*(\d+)", log)]
            result = json.loads((rtl_path / "result.json").read_text())
            if ("Verilog $finish" not in log or "*** FAILED ***" in log or
                    "%Error" in log or not cycles or
                    result["gpu_cycles"] != max(cycles)):
                raise ValueError(f"{rtl_path}: RTL log differs from reported result")
            writer.writerows(validate.inspect_run(
                root, root / rtl_directory, root / "model", name, summarize))
        completed = []
        for name in RTL_CASES[family]:
            result_path = root / rtl_directory / name / "result.json"
            if (result_path.exists() and
                    json.loads(result_path.read_text())["status"] == "passed"):
                completed.append(name)
        if included != completed:
            raise ValueError(f"{table}: does not cover every completed full-size RTL case")
        if table.read_text() != output.getvalue():
            raise ValueError(f"{table}: paired CSV differs from raw RTL/model runs")
        verified += len(included)
    return verified


def verify_small_rtl_tables() -> int:
    """Rebuild every small paired table from the retained build, RTL, and model runs."""
    validate = load_tool("spatter-validate.py")
    summarize = load_tool("spatter-summarize.py")
    verified = 0
    for family, filename, rtl_directory, names in SMALL_RTL_TABLES:
        root = KERNELS / family / "runs"
        table = KERNELS / family / "evaluation" / filename
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=summarize.FIELDS,
                                lineterminator="\n")
        writer.writeheader()
        for name in names:
            rtl_path = root / rtl_directory / name
            log = (rtl_path / "verilator.log").read_text(errors="replace")
            cycles = [int(value) for value in re.findall(r"\bCycles:\s*(\d+)", log)]
            result = json.loads((rtl_path / "result.json").read_text())
            if ("Verilog $finish" not in log or "*** FAILED ***" in log or
                    "%Error" in log or not cycles or
                    result["gpu_cycles"] != max(cycles)):
                raise ValueError(f"{rtl_path}: RTL log differs from reported result")
            writer.writerows(validate.inspect_run(
                root, root / rtl_directory, root / "model", name, summarize))
        if table.read_text() != output.getvalue():
            raise ValueError(f"{table}: paired CSV differs from raw RTL/model runs")
        verified += len(names)
    return verified


def report_csv(records: dict) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=REPORT_FIELDS, lineterminator="\n")
    writer.writeheader()
    for workload, (family, run, status) in ROWS.items():
        cycle, memory = records[(family, run)]
        model = json.loads((KERNELS / family / "runs/model" / run / "result.json").read_text())
        if model["status"] != status:
            raise ValueError(f"{family}/{run}: model status differs from report")
        writer.writerow({
            "workload": workload, "family": family, "run": run,
            "simulator": model["simulator"],
            "status": status, "correctness": model["correctness"],
            "elements_or_transfers": int(cycle["pattern_length"]) * int(cycle["count"]),
            "gpu_cycles": cycle["gpu_cycles"],
            "logical_payload_bytes": cycle["logical_payload_bytes"],
            "payload_bytes_per_cycle": cycle["payload_bytes_per_cycle"],
            "model_gmem_bytes_issued": memory["model_gmem_bytes_issued"],
            "model_gmem_transactions": memory["model_gmem_transactions"],
            "model_gmem_bytes_per_cycle": memory["model_gmem_bytes_per_cycle"],
            "model_lsu_global_loads_issued": memory["model_lsu_global_loads_issued"],
            "model_lsu_global_stores_issued": memory["model_lsu_global_stores_issued"],
            "output_digest": model["output_digest"],
            "expected_digest": model["expected_digest"],
            "elf_sha256": model["elf_sha256"],
            "kernel_source_sha256": model["kernel_source_sha256"],
            "suite_sha256": model["suite_sha256"],
            "timing_config_sha256": model["timing_config_sha256"],
            "checker_source_sha256": model["checker_source_sha256"],
        })
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-csv", action="store_true",
                        help="regenerate the consolidated model result CSV after verification")
    args = parser.parse_args()
    records = {}
    for family in ("stream", "spatter"):
        directory = KERNELS / family / "evaluation"
        cycles = indexed_csv(directory / "current-build-model-results.csv")
        memory = indexed_csv(directory / "current-build-memory.csv")
        if cycles.keys() != memory.keys():
            raise ValueError(f"cycle and memory runs differ for {family}")
        for run, cycle in cycles.items():
            mem = memory[run]
            if (cycle["gpu_cycles"] != mem["gpu_cycles"] or
                    cycle["logical_payload_bytes"] != mem["logical_payload_bytes"]):
                raise ValueError(f"cycle and memory counters differ for {family}/{run}")
            records[(family, run)] = (cycle, mem)

    report = (KERNELS / "WORKLOAD_RESULTS.md").read_text()
    header = ("| Workload | Elements or transfers | Model cycles | "
              "Logical payload bytes | Model global-memory bytes issued |")
    table = report.split(header, 1)[1].split("\n\n", 1)[0]
    found = set()
    for line in table.splitlines():
        if not line.startswith("| ") or line.startswith("| ---"):
            continue
        fields = [field.strip() for field in line.strip("|").split("|")]
        if len(fields) != 5 or fields[0] not in ROWS or fields[0] in found:
            raise ValueError(f"unexpected or duplicate report row: {line}")
        found.add(fields[0])
        family, run, status = ROWS[fields[0]]
        cycle, memory = records[(family, run)]
        expected = (
            int(cycle["pattern_length"]) * int(cycle["count"]),
            int(cycle["gpu_cycles"]),
            int(cycle["logical_payload_bytes"]),
            int(memory["model_gmem_bytes_issued"]),
        )
        if cycle["status"] != status or tuple(map(number, fields[1:])) != expected:
            raise ValueError(f"report differs from model run: {fields[0]}")
    if found != ROWS.keys() or len(records) != len(ROWS):
        raise ValueError("the report does not cover every current full-size model run")
    verify_raw_runs(records)
    paired = verify_paired_rtl_tables()
    small_paired = verify_small_rtl_tables()
    expected_csv = report_csv(records)
    if args.write_csv:
        REPORT_CSV.write_text(expected_csv)
    if REPORT_CSV.read_text() != expected_csv:
        raise ValueError(f"{REPORT_CSV} differs from verified model runs")
    print(f"verified {len(found)} workload rows, {paired} full-size and "
          f"{small_paired} small paired RTL cases, and consolidated CSV")


if __name__ == "__main__":
    main()
