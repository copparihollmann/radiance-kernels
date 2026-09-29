#!/usr/bin/env python3
"""Check the published workload table against completed model run CSVs."""

from __future__ import annotations

import csv
from pathlib import Path


KERNELS = Path(__file__).resolve().parents[1]
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
    print(f"verified {len(found)} published workload rows")


if __name__ == "__main__":
    main()
