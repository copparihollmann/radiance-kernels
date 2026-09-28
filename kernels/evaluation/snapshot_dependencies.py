#!/usr/bin/env python3
"""Preserve input decks and simulator binaries referenced by the run index.

Use hardlinks where permitted and verified copies across sandbox boundaries.
The CSV records original paths and hashes for later result reproduction.
"""

from __future__ import annotations

import csv
import errno
import json
import os
from pathlib import Path
import shutil

from make_inventory import HERE, sha256


FIELDS = ("kind", "original_path", "snapshot_path", "bytes", "sha256",
          "same_inode_as_source")


def suite_paths(value: str) -> list[str]:
    if not value:
        return []
    if value.startswith("["):
        paths = json.loads(value)
        return paths if isinstance(paths, list) else []
    return [value]


def main() -> None:
    with (HERE / "runs.csv").open(newline="") as source:
        runs = list(csv.DictReader(source))
    dependencies: set[tuple[str, str, str]] = set()
    for run in runs:
        suite = run["suite"]
        paths = suite_paths(suite)
        # A compound suite hash describes the composition, not either file.
        suite_digest = run["suite_sha256"] if len(paths) == 1 else ""
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(f"input suite missing: {path}")
            dependencies.add(("input", path, suite_digest))
        binary = run["simulator_path"]
        if binary and run["simulator_sha256"]:
            dependencies.add(("simulator", binary, run["simulator_sha256"]))

    rows = []
    for kind, path_text, declared in sorted(dependencies):
        original = Path(path_text)
        if not original.is_file():
            raise FileNotFoundError(original)
        digest = sha256(original)
        if declared and digest != declared:
            raise ValueError(f"{kind} changed since the run: {original}")
        snapshot = HERE / "dependency-snapshot" / kind / f"{digest}-{original.name}"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            try:
                os.link(original, snapshot)
            except OSError as error:
                # Sandboxed source roots can reject links across their boundary.
                if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES):
                    raise
                temporary = snapshot.with_name(snapshot.name + ".partial")
                shutil.copy2(original, temporary)
                if sha256(temporary) != digest:
                    raise ValueError(f"copy changed during capture: {original}")
                os.replace(temporary, snapshot)
        if sha256(snapshot) != digest:
            raise ValueError(f"snapshot hash differs: {snapshot}")
        rows.append({
            "kind": kind, "original_path": str(original),
            "snapshot_path": snapshot.relative_to(HERE).as_posix(),
            "bytes": original.stat().st_size, "sha256": digest,
            "same_inode_as_source": str(original.samefile(snapshot)).lower(),
        })
    with (HERE / "dependency-snapshot.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"verified {len(rows)} local input and simulator snapshots")


if __name__ == "__main__":
    main()
