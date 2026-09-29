#!/usr/bin/env python3
"""Package committed kernel sources and verified raw result snapshots.

The bundle supports later reporting from another machine. It includes Git
bundles for the kernel branch and pinned upstream Spatter, plus local ELF,
input, simulator, log, and golden-output snapshots. A draft is explicitly
labeled while required full-size RTL cases are missing or incomplete.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

from make_inventory import HERE, KERNELS, PRIOR_ROOTS, sha256
from verify_report import COMPOSITION_FULLSIZE_NAMES, RTL_CASES

REPO = KERNELS.parent
SPATTER = KERNELS / "spatter"
PINNED_SPATTER = "ec8923711f8dc21eedff7189f12b02eb06845d2f"
RTL_ROOTS = {
    "spatter": ("spatter-current", "rtl"),
    "stream": ("stream-current", "rtl-full"),
}


def command(*args: str, cwd: Path = REPO) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def table(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="") as source:
        return list(csv.DictReader(source))


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    archive.addfile(info, io.BytesIO(data))


def add_directory(archive: tarfile.TarFile, root: Path, label: str) -> int:
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            archive.add(path, f"{label}/{path.relative_to(root).as_posix()}",
                        recursive=False)
            count += 1
    return count


def verify_current_run_roots(files: list[dict[str, str]]) -> None:
    """Require a fresh inventory before calling an archive complete."""
    workspace = REPO.parent
    roots = {
        "spatter-current": SPATTER / "runs",
        "stream-current": KERNELS / "stream/runs",
    }
    roots.update((name, workspace / relative) for name, relative in PRIOR_ROOTS.items())
    indexed = {(row["root"], row["path"]): row for row in files}
    actual = {}
    for name, root in roots.items():
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                actual[name, path.relative_to(root).as_posix()] = path
    if actual.keys() != indexed.keys():
        raise ValueError("run files changed since the inventory; refresh snapshots")
    for key, path in actual.items():
        row = indexed[key]
        if (not row["sha256"] or path.stat().st_size != int(row["bytes"]) or
                sha256(path) != row["sha256"]):
            raise ValueError(f"run file changed since the inventory: {key}")


def pending_required_rtl(runs: list[dict[str, str]]) -> list[str]:
    statuses = {(row["root"], row["run"]): row["status"] for row in runs}
    pending = []
    for family, cases in RTL_CASES.items():
        root, directory = RTL_ROOTS[family]
        for case in cases:
            path = f"{directory}/{case}"
            status = statuses.get((root, path), "missing")
            if status != "passed":
                pending.append(f"{root}/{path}: {status}")
    for case, _ in COMPOSITION_FULLSIZE_NAMES:
        path = f"rtl-full-composition/{case}"
        status = statuses.get(("spatter-current", path), "missing")
        if status != "passed":
            pending.append(f"spatter-current/{path}: {status}")
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--draft", action="store_true",
                        help="allow an explicitly incomplete bundle while required RTL runs remain")
    parser.add_argument("--dry-run", action="store_true",
                        help="verify contents and report bundle scope without writing it")
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    if command("git", "status", "--porcelain"):
        raise ValueError("commit kernel changes before exporting their source")
    if command("git", "-C", str(upstream), "rev-parse", "HEAD") != PINNED_SPATTER:
        raise ValueError("upstream Spatter is not at the pinned revision")
    subprocess.run(["python3", str(HERE / "verify_report.py")], check=True)
    subprocess.run(["python3", str(HERE / "verify_artifact_snapshot.py")], check=True)
    subprocess.run(["python3", str(SPATTER / "tools/verify-upstream-golden.py"),
                    "--upstream", str(upstream)], check=True)
    runs = table("runs.csv")
    running = [f"{row['root']}/{row['run']}" for row in runs
               if row["status"] == "running"]
    pending_rtl = pending_required_rtl(runs)
    if (running or pending_rtl) and not args.draft:
        raise ValueError("full-size RTL evaluation is incomplete; use --draft "
                         "for a provisional archive: "
                         f"running={running}, required_rtl={pending_rtl}")
    if not args.output and not args.dry_run:
        parser.error("--output is required unless --dry-run is set")
    files = table("artifacts.csv")
    if not running and not pending_rtl:
        verify_current_run_roots(files)
    goldens = []
    with (SPATTER / "evaluation/upstream-golden-artifacts.csv").open(newline="") as source:
        goldens = list(csv.DictReader(source))
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "draft" if running or pending_rtl else "complete-snapshot",
        "running": running,
        "pending_required_rtl": pending_rtl,
        "radiance_kernels_commit": command("git", "rev-parse", "HEAD"),
        "upstream_spatter_commit": PINNED_SPATTER,
        "runs": len(runs), "indexed_run_files": len(files),
        "golden_files": len(goldens),
        "raw_metadata_sha256": sha256(HERE / "raw-metadata.tar.gz"),
        "artifact_index_sha256": sha256(HERE / "artifacts.csv"),
        "golden_index_sha256": sha256(
            SPATTER / "evaluation/upstream-golden-artifacts.csv"),
    }
    if args.dry_run:
        print(json.dumps(info, indent=2))
        return

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            kernel_bundle = scratch / "radiance-kernels.bundle"
            upstream_bundle = scratch / "upstream-spatter.bundle"
            subprocess.run(["git", "bundle", "create", str(kernel_bundle), "HEAD"],
                           cwd=REPO, check=True)
            subprocess.run(["git", "bundle", "create", str(upstream_bundle), "HEAD"],
                           cwd=upstream, check=True)
            with temporary.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                   compresslevel=1, mtime=0) as zipped:
                    with tarfile.open(fileobj=zipped, mode="w") as archive:
                        add_bytes(archive, "bundle-info.json",
                                  (json.dumps(info, indent=2) + "\n").encode())
                        archive.add(kernel_bundle, "radiance-kernels.bundle")
                        archive.add(upstream_bundle, "upstream-spatter.bundle")
                        for name in ("README.md", "runs.csv", "artifacts.csv",
                                     "elf-snapshot.csv", "dependency-snapshot.csv",
                                     "raw-metadata.tar.gz"):
                            archive.add(HERE / name, f"evaluation/{name}")
                        add_directory(archive, HERE / "elf-snapshot", "elf-snapshot")
                        add_directory(archive, HERE / "dependency-snapshot",
                                      "dependency-snapshot")
                        add_directory(archive, SPATTER / "golden-runs", "golden-runs")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"wrote {output}: {output.stat().st_size} bytes, "
          f"sha256={sha256(output)}, status={info['status']}")


if __name__ == "__main__":
    main()
