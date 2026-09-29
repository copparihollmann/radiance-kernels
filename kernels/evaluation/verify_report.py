#!/usr/bin/env python3
"""Check the published workload table against completed model run CSVs."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
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
}


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


def main() -> None:
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
    print(f"verified {len(found)} published workload rows")


if __name__ == "__main__":
    main()
