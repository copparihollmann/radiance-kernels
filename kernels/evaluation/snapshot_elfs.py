#!/usr/bin/env python3
"""Preserve indexed ELF binaries as local hardlinks without copying their data."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from make_inventory import HERE, KERNELS, PRIOR_ROOTS, sha256


def main(workspace: Path) -> None:
    roots = {
        "spatter-current": KERNELS / "spatter/runs",
        "stream-current": KERNELS / "stream/runs",
    }
    roots.update((key, workspace / name) for key, name in PRIOR_ROOTS.items())
    with (HERE / "artifacts.csv").open(newline="") as source:
        files = list(csv.DictReader(source))
    rows = []
    for file in files:
        if not file["path"].endswith("kernel.soc.elf"):
            continue
        root = roots[file["root"]]
        original = root / file["path"]
        snapshot = HERE / "elf-snapshot" / file["root"] / file["path"]
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            if not original.exists():
                raise FileNotFoundError(f"ELF missing from both locations: {original}")
            os.link(original, snapshot)
        if original.exists() and not original.samefile(snapshot):
            raise ValueError(f"snapshot points to another inode: {snapshot}")
        digest = sha256(snapshot)
        if digest != file["sha256"]:
            raise ValueError(f"snapshot hash differs from inventory: {snapshot}")
        rows.append({
            "root": file["root"], "path": file["path"],
            "snapshot_path": snapshot.relative_to(HERE).as_posix(),
            "bytes": file["bytes"], "sha256": digest,
            "same_inode_as_run": "true" if original.exists() else "source-missing",
        })
    output = HERE / "elf-snapshot.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "root", "path", "snapshot_path", "bytes", "sha256", "same_inode_as_run",
        ), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"verified {len(rows)} local ELF hardlinks against artifacts.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=KERNELS.parent.parent)
    args = parser.parse_args()
    main(args.workspace.resolve())
