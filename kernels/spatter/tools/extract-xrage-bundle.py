#!/usr/bin/env python3
"""Extract the two original xRAGE u32 patterns from a portable HPC bundle.

The portable bundle stores each pattern inside a content-addressed global-memory
load image.  This tool uses the recorded addresses and generated header instead
of hard-coded blob names, verifies every source receipt, and emits a compact
Spatter suite whose patterns remain binary rather than multi-million-entry JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


DEFINE = re.compile(r"^#define\s+(SPATTER_[A-Z_]+)\s+([^\s]+)", re.MULTILINE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_blob(bundle: Path, record: dict) -> Path:
    path = bundle / record["path"]
    if not path.is_file():
        raise ValueError(f"bundle blob is missing: {record['path']}")
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"bundle blob differs from its receipt: {record['path']}")
    return path


def generated_header(bundle: Path, workload: dict) -> dict[str, int]:
    source = checked_blob(bundle, workload["sources"]["generated_sources"])
    with tarfile.open(source, "r:gz") as archive:
        member = archive.getmember("spatter_data.h")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f"{workload['name']}: generated source has no spatter_data.h")
        text = stream.read().decode()
    values = {}
    for name, raw in DEFINE.findall(text):
        raw = raw.rstrip("uUlL")
        try:
            values[name] = int(raw, 0)
        except ValueError:
            pass
    required = {"SPATTER_PATTERN_ADDR", "SPATTER_PATTERN_BYTES", "SPATTER_LENGTH",
                "SPATTER_COUNT", "SPATTER_DELTA", "SPATTER_WRAP"}
    missing = required - values.keys()
    if missing:
        raise ValueError(f"{workload['name']}: generated header lacks {sorted(missing)}")
    return values


def primary(bundle_manifest: dict, operation: str) -> dict:
    matches = [
        row for row in bundle_manifest["workloads"]
        if row.get("benchmark") == "xrage"
        and row.get("run_group") == "primary"
        and row.get("sm_count") == 1
        and row.get("logical_partitions") == 1
        and row["name"].startswith(f"xrage-{operation}--")
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one primary 1-SM xRAGE {operation}, found {len(matches)}")
    return matches[0]


def extract_pattern(bundle: Path, workload: dict, out: Path) -> dict:
    header = generated_header(bundle, workload)
    address = header["SPATTER_PATTERN_ADDR"]
    size = header["SPATTER_PATTERN_BYTES"]
    if size != header["SPATTER_LENGTH"] * 4:
        raise ValueError(f"{workload['name']}: pattern byte count is inconsistent")
    containing = [
        load for load in workload["loads"]
        if load["address"] <= address
        and address + size <= load["address"] + load["bytes"]
    ]
    if len(containing) != 1:
        raise ValueError(f"{workload['name']}: pattern is not in exactly one load region")
    load = containing[0]
    source = checked_blob(bundle, load)
    offset = address - load["address"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, out.open("wb") as dst:
        src.seek(offset)
        remaining = size
        while remaining:
            block = src.read(min(8 << 20, remaining))
            if not block:
                raise ValueError(f"{workload['name']}: load ended inside the pattern")
            dst.write(block)
            remaining -= len(block)
    return {
        "workload": workload["name"],
        "source_load": load,
        "source_offset": offset,
        "pattern_address": address,
        "pattern_bytes": size,
        "pattern_length": header["SPATTER_LENGTH"],
        "pattern_sha256": sha256(out),
        "count": header["SPATTER_COUNT"],
        "delta": header["SPATTER_DELTA"],
        "wrap": header["SPATTER_WRAP"],
    }


def descriptor(path: Path, receipt: dict) -> dict:
    return {
        "file": path.name,
        "format": "u32le",
        "length": receipt["pattern_length"],
        "bytes": receipt["pattern_bytes"],
        "sha256": receipt["pattern_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="extracted fpga-handoff-hpc-all-sm package")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    out = args.out.resolve()
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"bundle manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    gather = primary(manifest, "gather")
    scatter = primary(manifest, "scatter")
    out.mkdir(parents=True, exist_ok=True)
    gather_path = out / "xrage-gather.u32le"
    scatter_path = out / "xrage-scatter.u32le"
    gather_receipt = extract_pattern(bundle, gather, gather_path)
    scatter_receipt = extract_pattern(bundle, scatter, scatter_path)
    suite = [
        {
            "kernel": "Gather", "pattern": descriptor(gather_path, gather_receipt),
            "count": gather_receipt["count"], "delta": gather_receipt["delta"],
            "wrap": gather_receipt["wrap"], "source-tag": 1,
        },
        {
            "kernel": "Scatter", "pattern": descriptor(scatter_path, scatter_receipt),
            "count": scatter_receipt["count"], "delta": scatter_receipt["delta"],
            "wrap": scatter_receipt["wrap"], "source-tag": 2,
            "collision-policy": "ordered",
        },
    ]
    suite_path = out / "xrage-suite.json"
    suite_path.write_text(json.dumps(suite, indent=2) + "\n")
    receipt = {
        "schema": "radiance-xrage-bundle-extraction-v1",
        "bundle": str(bundle),
        "bundle_manifest_sha256": sha256(manifest_path),
        "suite": suite_path.name,
        "suite_sha256": sha256(suite_path),
        "patterns": {"gather": gather_receipt, "scatter": scatter_receipt},
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"wrote {suite_path} with {gather_receipt['pattern_length']:,} gather and "
          f"{scatter_receipt['pattern_length']:,} scatter addresses")


if __name__ == "__main__":
    main()
