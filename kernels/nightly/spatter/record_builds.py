#!/usr/bin/env python3
"""Record and check the five original GPU STREAM Spatter HBM builds."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
UPSTREAM = ROOT / "kernels/spatter/evaluation/upstream-golden-results.csv"
SOURCE_FILES = (
    "kernels/nightly/spatter/kernel.cpp",
    "kernels/nightly/spatter/host.cpp",
    "kernels/nightly/spatter/spatter_body.hpp",
    "kernels/nightly/spatter/gen_data.py",
    "kernels/nightly/spatter/Makefile",
    "kernels/nightly/nightly.mk",
    "kernels/spatter/run.py",
    "kernels/spatter/plan.py",
    "kernels/spatter/spatter_ops.hpp",
    "soc/scramble_gpu_elf.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def elf_record(path):
    sections = subprocess.check_output(["readelf", "-W", "-p",
                                        ".radiance_addr_hash", str(path)],
                                       text=True)
    if "unit=32 slices=4 size=0x80000000 gpu_base=0x100000000" not in sections:
        raise ValueError(f"wrong or missing GPU address hash: {path}")
    segments = subprocess.check_output(["readelf", "-W", "-l", str(path)],
                                       text=True)
    intervals = []
    loaded = 0
    for line in segments.splitlines():
        fields = line.split()
        if not fields or fields[0] != "LOAD":
            continue
        address = int(fields[3], 16)
        if address < 0x100000000:
            continue
        memory_bytes = int(fields[5], 16)
        file_bytes = int(fields[4], 16)
        if not 0x100000000 <= address < 0x180000000 or (
                address + memory_bytes > 0x180000000):
            raise ValueError(f"GPU segment outside 2 GiB map: {path}")
        intervals.append((address, address + memory_bytes))
        loaded += file_bytes
    if not intervals:
        raise ValueError(f"no GPU load segments: {path}")
    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError(f"overlapping GPU segments: {path}")
    return {"file_size_bytes": path.stat().st_size, "sha256": sha256(path),
            "address_hash_section_present": True,
            "gpu_load_segments": len(intervals), "gpu_loaded_bytes": loaded,
            "gpu_region_in_bounds": True}


def record(cases_root):
    with UPSTREAM.open(newline="") as stream:
        upstream = {row["run"]: row for row in csv.DictReader(stream)}
    rows = []
    for case_id in range(5):
        case_dir = cases_root / f"standard-{case_id}"
        data_path = case_dir / "spatter.json"
        data = json.loads(data_path.read_text())
        prior = upstream[f"rebuild-gpu-stream-{case_id}"]
        if (data["case"] != case_id or data["suite_sha256"] != prior["suite_sha256"]
                or data["kind"] != prior["kind"]
                or data["source_sha256"] != prior["source_sha256"]
                or data["golden_sha256"] != prior["golden_output_sha256"]
                or data["reference_digest"] != prior["golden_digest"]):
            raise ValueError(f"case {case_id} differs from pinned upstream oracle")
        rows.append({"case": case_id, "kind": data["kind"],
                     "pattern_length": data["pattern_length"],
                     "count": data["count"],
                     "output_elements": data["output_elements"],
                     "logical_payload_bytes": data["logical_payload_bytes"],
                     "data_receipt_sha256": sha256(data_path),
                     "source_sha256": data["source_sha256"],
                     "golden_sha256": data["golden_sha256"],
                     "upstream_run": prior["run"],
                     "elfs": {f"{sm}sm": elf_record(
                         case_dir / f"spatter-{sm}sm.soc.elf")
                              for sm in (1, 2)}})
    return {"schema": "radiance.nightly.spatter.hbm-builds.v1",
            "backend": "nightly HBM SoC ELF build",
            "memory_hash": {"gpu_size_bytes": 0x80000000,
                            "unit_bytes": 32, "slices": 4,
                            "load_path": "direct LoadMem"},
            "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
            "upstream_revision": upstream["rebuild-gpu-stream-0"]["upstream_revision"],
            "upstream_table_sha256": sha256(UPSTREAM),
            "compiled_cores_per_cluster": 2,
            "local_checks": [
                "all ten SoC ELFs linked with address hash metadata",
                "GPU load segments fit the declared 2 GiB map without overlap",
                "source and full golden hashes match pinned upstream serial oracle",
                "one-SM and two-SM ELF hashes differ for every case",
            ],
            "rtl_execution": False, "fpga_execution": False,
            "performance_measurement": False, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = record(args.cases_root.resolve())
    for case in result["cases"]:
        if case["elfs"]["1sm"]["sha256"] == case["elfs"]["2sm"]["sha256"]:
            raise ValueError(f"case {case['case']} produced identical SM images")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"recorded {len(result['cases'])} cases and ten distinct images")


if __name__ == "__main__":
    main()
