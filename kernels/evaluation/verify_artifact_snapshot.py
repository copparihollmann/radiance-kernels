#!/usr/bin/env python3
"""Verify the committed raw-record archive and local binary snapshots.

The archive and CSVs are a point-in-time capture. Active run logs may have
grown since that capture, so this checks archived bytes rather than comparing
them with mutable files in the run directories.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tarfile

from make_inventory import HERE, sha256


def rows(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="") as source:
        return list(csv.DictReader(source))


def check_file(path: Path, size: str, digest: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing snapshot: {path}")
    if path.stat().st_size != int(size) or sha256(path) != digest:
        raise ValueError(f"snapshot differs from manifest: {path}")


def main() -> None:
    files = rows("artifacts.csv")
    runs = rows("runs.csv")
    elfs = rows("elf-snapshot.csv")
    dependencies = rows("dependency-snapshot.csv")
    by_file = {(row["root"], row["path"]): row for row in files}
    if len(by_file) != len(files):
        raise ValueError("duplicate artifact path")
    if len({(row["root"], row["run"]) for row in runs}) != len(runs):
        raise ValueError("duplicate run path")

    expected_archive = {
        "inventory/artifacts.csv": (HERE / "artifacts.csv").read_bytes(),
        "inventory/runs.csv": (HERE / "runs.csv").read_bytes(),
    }
    archived_results = {}
    for row in files:
        name = f"{row['root']}/{row['path']}"
        if row["changed_during_capture"] == "true":
            if row["sha256"] or row["in_metadata_archive"] != "false":
                raise ValueError(f"changing file has a captured hash: {name}")
        elif row["in_metadata_archive"] == "true":
            expected_archive[name] = None
        elif not row["path"].endswith("kernel.soc.elf"):
            raise ValueError(f"stable non-ELF missing from archive: {name}")

    seen = set()
    with tarfile.open(HERE / "raw-metadata.tar.gz", "r:gz") as archive:
        for member in archive:
            name = member.name
            if name in seen or name not in expected_archive or not member.isfile():
                raise ValueError(f"unexpected or duplicate archive member: {name}")
            seen.add(name)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"archive member cannot be read: {name}")
            contents = stream.read()
            manifest_bytes = expected_archive[name]
            if manifest_bytes is not None:
                if contents != manifest_bytes:
                    raise ValueError(f"archived inventory differs: {name}")
            else:
                root, path = name.split("/", 1)
                row = by_file[(root, path)]
                if (len(contents) != int(row["bytes"]) or
                        hashlib.sha256(contents).hexdigest() != row["sha256"]):
                    raise ValueError(f"archived file differs: {name}")
                if path.endswith("/result.json"):
                    archived_results[(root, path.removesuffix("/result.json"))] = (
                        json.loads(contents)
                    )
    if seen != expected_archive.keys():
        raise ValueError(f"archive misses {sorted(expected_archive.keys() - seen)}")

    for row in runs:
        key = (row["root"], row["run"])
        result = archived_results.get(key)
        if result is None:
            raise ValueError(f"run has no archived result JSON: {key}")
        for field in ("status", "gpu_cycles", "elf_sha256", "suite_sha256"):
            recorded = result.get(field)
            if row[field] != ("" if recorded is None else str(recorded)):
                # Older records can omit the ELF digest; the inventory then
                # supplies the digest measured from the indexed ELF itself.
                if field != "elf_sha256" or recorded:
                    raise ValueError(f"run index differs from result JSON: {key}/{field}")
        artifact = by_file[key[0], f"{key[1]}/result.json"]
        if row["result_sha256"] != artifact["sha256"]:
            raise ValueError(f"run index has another result digest: {key}")

    expected_elfs = {
        key for key, row in by_file.items()
        if row["path"].endswith("kernel.soc.elf") and row["sha256"]
    }
    seen_elfs = set()
    for row in elfs:
        key = (row["root"], row["path"])
        if key in seen_elfs or key not in expected_elfs:
            raise ValueError(f"unexpected or duplicate ELF snapshot: {key}")
        seen_elfs.add(key)
        artifact = by_file[key]
        if (row["sha256"], row["bytes"]) != (artifact["sha256"], artifact["bytes"]):
            raise ValueError(f"ELF manifest disagrees with artifact: {key}")
        check_file(HERE / row["snapshot_path"], row["bytes"], row["sha256"])
    if seen_elfs != expected_elfs:
        raise ValueError(f"ELF snapshots missing: {sorted(expected_elfs - seen_elfs)}")

    seen_dependencies = set()
    dependency_by_source = {}
    for row in dependencies:
        key = (row["kind"], row["original_path"], row["sha256"])
        if key in seen_dependencies:
            raise ValueError(f"duplicate dependency: {key}")
        seen_dependencies.add(key)
        dependency_by_source[(row["kind"], row["original_path"])] = row
        check_file(HERE / row["snapshot_path"], row["bytes"], row["sha256"])

    for row in runs:
        suite = row["suite"]
        paths = json.loads(suite) if suite.startswith("[") else [suite] if suite else []
        for path in paths:
            if ("input", path) not in dependency_by_source:
                raise ValueError(f"input snapshot missing for {row['root']}/{row['run']}: {path}")
        if len(paths) == 1 and row["suite_sha256"]:
            dependency = dependency_by_source["input", paths[0]]
            if dependency["sha256"] != row["suite_sha256"]:
                raise ValueError(f"input digest differs for {row['root']}/{row['run']}")
        simulator = row["simulator_path"]
        if simulator and row["simulator_sha256"]:
            dependency = dependency_by_source.get(("simulator", simulator))
            if dependency is None or dependency["sha256"] != row["simulator_sha256"]:
                raise ValueError(f"simulator snapshot differs for {row['root']}/{row['run']}")

    print(f"verified {len(files)} indexed artifacts, {len(expected_archive)} "
          f"archive members, {len(elfs)} ELF snapshots, and "
          f"{len(dependencies)} dependency snapshots for {len(runs)} runs")


if __name__ == "__main__":
    main()
