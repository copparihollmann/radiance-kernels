#!/usr/bin/env python3
"""Bind original-size LANL xRAGE HBM ELFs to upstream and device checks."""

import argparse
import csv
import json
from pathlib import Path
import re

from extract_cases import LANL_XRAGE_SHA256
from record_builds import elf_record, sha256, SOURCE_FILES, UPSTREAM, ROOT


CASES = (
    (0, 4, "xrage5", "parallel"),
    (1, 8, "xrage9-ordered", "ordered"),
)


def record(cases_root, controls_root, suite, extraction, simulator):
    with UPSTREAM.open(newline="") as stream:
        upstream = {row["run"]: row for row in csv.DictReader(stream)}
    extracted = json.loads(extraction.read_text())
    if (extracted.get("source_sha256") != LANL_XRAGE_SHA256 or
            extracted.get("selected_cases") != [4, 8] or
            extracted.get("output_sha256") != sha256(suite)):
        raise ValueError("selected suite differs from pinned LANL xRAGE source")
    simulator_sha256 = sha256(simulator)
    rows = []
    for case_id, original_index, upstream_run, collision in CASES:
        case_dir = cases_root / ("xrage5" if case_id == 0 else "xrage9")
        data_path = case_dir / "spatter.json"
        data = json.loads(data_path.read_text())
        prior = upstream[upstream_run]
        if (data["case"] != case_id or
                data["suite_sha256"] != extracted["output_sha256"] or
                data["collision_policy"] != collision or
                data["kind"] != prior["kind"] or
                str(data["pattern_length"]) != prior["pattern_length"] or
                str(data["count"]) != prior["count"] or
                data["source_sha256"] != prior["source_sha256"] or
                data["golden_sha256"] != prior["golden_output_sha256"] or
                data["reference_digest"] != prior["golden_digest"]):
            raise ValueError(f"case {case_id} differs from upstream xRAGE oracle")
        controls = {}
        for partitions in (1, 2):
            path = controls_root / ("device-xrage5" if case_id == 0 else
                                    "device-xrage9") / f"case-{case_id}" / str(partitions) / "result.json"
            result = json.loads(path.read_text())
            run_dir = path.parent
            if (result["case"] != case_id or result["status"] != "passed" or
                    result["kind"] != data["kind"] or
                    result["collision_policy"] != collision or
                    result["logical_partitions"] != partitions or
                    result["output_elements"] != data["output_elements"] or
                    result["exact_output_bytes"] != data["output_elements"] * 8 or
                    result["output_sha256"] != data["golden_sha256"] or
                    result["reference_sha256"] != data["golden_sha256"] or
                    result["data_receipt_sha256"] != sha256(data_path) or
                    result["data_receipt_sha256"] != sha256(
                        run_dir / "data/spatter.json") or
                    result["spatter_body_sha256"] != sha256(
                        ROOT / "kernels/nightly/spatter/spatter_body.hpp") or
                    result["simulator_sha256"] != simulator_sha256 or
                    result["elf_sha256"] != sha256(run_dir / "kernel.radiance.elf") or
                    result["log_sha256"] != sha256(run_dir / "cyclotron.log") or
                    result["config_sha256"] != sha256(
                        run_dir / "cyclotron-config.toml") or
                    result["output_sha256"] != sha256(run_dir / "output.bin") or
                    result["cycles_functional"] is None or
                    result["cycles_functional"] <= 0 or
                    result["rtl_execution"] or result["performance_measurement"] or
                    result["two_sm_execution"]):
                raise ValueError(f"case {case_id}/{partitions} lacks an exact device control")
            log = (run_dir / "cyclotron.log").read_text(errors="replace")
            cycles = re.findall(r"simulation finished after (\d+) cycles", log)
            if (log.count("isa-test passed with tohost=0") != 1 or
                    cycles != [str(result["cycles_functional"])]):
                raise ValueError(f"case {case_id}/{partitions} completion log differs")
            controls[str(partitions)] = {
                "result_sha256": sha256(path),
                "output_sha256": result["output_sha256"],
                "elf_sha256": result["elf_sha256"],
                "log_sha256": result["log_sha256"],
                "cycles_functional": result["cycles_functional"],
                "exact_output_bytes": result["exact_output_bytes"],
            }
        elfs = {f"{sm}sm": elf_record(case_dir / f"spatter-{sm}sm.soc.elf")
                for sm in (1, 2)}
        if elfs["1sm"]["sha256"] == elfs["2sm"]["sha256"]:
            raise ValueError(f"case {case_id} has identical one- and two-SM images")
        rows.append({
            "case": case_id, "upstream_case": original_index,
            "upstream_run": upstream_run, "kind": data["kind"],
            "collision_policy": collision,
            "pattern_length": data["pattern_length"], "count": data["count"],
            "output_elements": data["output_elements"],
            "logical_payload_bytes": data["logical_payload_bytes"],
            "data_receipt_sha256": sha256(data_path),
            "source_sha256": data["source_sha256"],
            "golden_sha256": data["golden_sha256"],
            "golden_digest": data["reference_digest"],
            "functional_controls": controls, "elfs": elfs,
        })
    sources = SOURCE_FILES + (
        "kernels/nightly/spatter/extract_cases.py",
        "kernels/nightly/spatter/record_lanl_builds.py",
        "kernels/nightly/spatter/test_device.py",
        "lib/libmuonrt.a",
    )
    return {
        "schema": "radiance.nightly.spatter.hbm-builds.v1",
        "suite": "lanl-xrage-asteroid-patterns-5-and-9",
        "source_trace_sha256": LANL_XRAGE_SHA256,
        "selected_suite_sha256": extracted["output_sha256"],
        "extraction_receipt_sha256": sha256(extraction),
        "upstream_revision": upstream["xrage5"]["upstream_revision"],
        "upstream_table_sha256": sha256(UPSTREAM),
        "simulator_sha256": simulator_sha256,
        "memory_hash": {"gpu_size_bytes": 0x80000000,
                        "unit_bytes": 32, "slices": 4,
                        "load_path": "direct LoadMem"},
        "compiled_cores_per_cluster": 2,
        "source_sha256": {name: sha256(ROOT / name) for name in sources},
        "local_checks": [
            "source trace and selected cases match pinned LANL SHA-256",
            "full source and golden hashes match the pinned Spatter serial oracle",
            "one- and two-partition device bodies match every golden output byte",
            "one- and two-SM SoC ELFs fit the declared 2 GiB GPU map",
        ],
        "rtl_execution": False, "fpga_execution": False,
        "performance_measurement": False, "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--controls-root", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = record(args.cases_root.resolve(), args.controls_root.resolve(),
                    args.suite.resolve(), args.extraction.resolve(),
                    args.simulator.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"recorded {len(result['cases'])} xRAGE cases with four HBM ELFs")


if __name__ == "__main__":
    main()
