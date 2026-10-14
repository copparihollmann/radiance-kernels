#!/usr/bin/env python3
"""Generate a fully checked nightly HBM case from an existing Spatter deck."""

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parents[1] / "spatter"))
from nightly_data import Layout  # noqa: E402
from run import (KINDS, destination, fnv, normalize, payload, reference,
                 source_index)  # noqa: E402


def groups(case):
    destinations = case["dst_length"]
    tasks = case["count"] * case["length"]
    counts = array("I", [0]) * destinations
    for task in range(tasks):
        i, j = divmod(task, case["length"])
        counts[destination(case, i, j)] += 1
    offsets = array("I", [0]) * (destinations + 1)
    for index, count in enumerate(counts):
        offsets[index + 1] = offsets[index] + count
    cursor = offsets[:destinations]
    ordered = array("I", [0]) * tasks
    for task in range(tasks):
        i, j = divmod(task, case["length"])
        dst = destination(case, i, j)
        ordered[cursor[dst]] = task
        cursor[dst] += 1
    return offsets, ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--collision-policy", choices=("parallel", "ordered"))
    parser.add_argument("--max-layout-mib", type=int, default=128,
                        help="reject large case blobs before allocating them")
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    if not isinstance(suite, list) or not 0 <= args.case < len(suite):
        parser.error("case is outside the Spatter suite")
    raw = dict(suite[args.case])
    if args.collision_policy:
        raw["collision-policy"] = args.collision_policy
    case = normalize(raw)
    ordered_words = (case["dst_length"] + 1 + case["count"] * case["length"]
                     if case["collision_policy"] == "ordered" else 2)
    layout_bytes = (8 * (case["src_length"] + 2 * case["dst_length"])
                    + 4 * (sum(max(1, len(case[name])) for name in
                               ("pattern", "gather", "scatter")) + ordered_words))
    if args.max_layout_mib < 1 or layout_bytes > args.max_layout_mib * (1 << 20):
        parser.error(f"case requires about {layout_bytes / (1 << 20):.1f} MiB "
                     "of input/output blobs; raise --max-layout-mib explicitly")
    expected_digest, overlap = reference(case)
    if expected_digest is None:
        parser.error("colliding scatter destinations require --collision-policy ordered")
    source = array("Q", (payload(case["payload_tag"], i)
                         for i in range(case["src_length"])))
    golden = array("Q", [0]) * case["dst_length"]
    for iteration in range(case["count"]):
        for j in range(case["length"]):
            golden[destination(case, iteration, j)] = source[
                source_index(case, iteration, j)]
    if fnv(golden) != expected_digest:
        raise AssertionError("materialized golden differs from the existing Spatter reference")
    offsets, tasks = (groups(case) if case["collision_policy"] == "ordered"
                      else (array("I", [0]), array("I", [0])))

    layout = Layout("spatter")
    constants = {
        "SPATTER_KIND": KINDS[case["kind"]],
        "SPATTER_LENGTH": f"{case['length']}u",
        "SPATTER_COUNT": f"{case['count']}u",
        "SPATTER_WRAP": f"{case['wrap']}u",
        "SPATTER_DELTA": f"{case['delta']}u",
        "SPATTER_DELTA_GATHER": f"{case['delta_gather']}u",
        "SPATTER_DELTA_SCATTER": f"{case['delta_scatter']}u",
        "SPATTER_GATHER_FINAL_WRAP": f"{case['gather_final_wrap']}u",
        "SPATTER_OUTPUT_LENGTH": f"{case['dst_length']}u",
        "SPATTER_ORDERED": int(case["collision_policy"] == "ordered"),
        "SPATTER_LOGICAL_BYTES": f"{case['count'] * case['length'] * 16}ull",
    }
    for name, value in constants.items():
        layout.const(name, value)
    layout.add("SPATTER_SRC", 1, np.frombuffer(source, dtype="<u8"),
               "opaque 64-bit source payload")
    for name, values in (("PATTERN", case["pattern"]),
                         ("GATHER", case["gather"]),
                         ("SCATTER", case["scatter"])):
        layout.add("SPATTER_" + name, 1,
                   np.asarray(values or [0], dtype="<u4"), "pattern indices")
    layout.add("SPATTER_OFFSETS", 1, np.frombuffer(offsets, dtype="<u4"),
               "ordered destination offsets")
    layout.add("SPATTER_TASKS", 1, np.frombuffer(tasks, dtype="<u4"),
               "ordered transfer tasks")
    layout.reserve("SPATTER_OUT", 2, case["dst_length"] * 8,
                   fill=0, comment="zero-initialized 64-bit destination")
    layout.add("SPATTER_GOLD", 3, np.frombuffer(golden, dtype="<u8"),
               "complete serial transfer reference")
    layout.write("data", "spatter_data.h")
    receipt = {
        "suite": str(args.suite.resolve()),
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "case": args.case,
        "kind": case["kind"], "pattern_length": case["length"],
        "count": case["count"], "wrap": case["wrap"],
        "source_elements": case["src_length"],
        "output_elements": case["dst_length"],
        "collision_policy": case["collision_policy"],
        "destination_overlap": overlap,
        "address_plan": case["_plan"].as_dict(),
        "logical_payload_bytes": case["count"] * case["length"] * 16,
        "reference_source": "kernels/spatter/run.py and plan.py",
        "reference_digest": f"{expected_digest:016x}",
        "source_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
        "golden_sha256": hashlib.sha256(golden.tobytes()).hexdigest(),
        "ordered_offsets_sha256": hashlib.sha256(offsets.tobytes()).hexdigest(),
        "ordered_tasks_sha256": hashlib.sha256(tasks.tobytes()).hexdigest(),
    }
    Path("data/spatter.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
