#!/usr/bin/env python3
"""Index workload run artifacts and preserve their small raw records.

The CSVs cover every file in the configured run roots. The archive contains
all non-ELF files, including result JSON, logs, and model counter summaries.
ELFs stay in the run roots and are identified by SHA-256 in the file index.
Rerun after active simulations finish to capture their final records.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


KERNELS = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PRIOR_ROOTS = {
    "prior-vcs": "spatter-vcs-runs",
    "prior-verilator": "spatter-verilator-runs",
    "prior-sampled-rtl": "spatter-sampled-rtl-runs",
    "prior-cyclotron": "spatter-cyclotron-runs",
    "prior-sampled-builds": "spatter-sampled-builds",
}
RESULT_TABLES = (
    ("spatter-current", "spatter/evaluation/current-build-model-results.csv", "model", "model"),
    ("spatter-current", "spatter/evaluation/current-build-results.csv", "model", "rtl"),
    ("spatter-current", "spatter/evaluation/gpu-stream-revision-comparison.csv", "model", "model"),
    ("spatter-current", "spatter/evaluation/xrage9-revision-comparison.csv", "model", "model"),
    ("stream-current", "stream/evaluation/current-build-model-results.csv", "model", "model"),
    ("stream-current", "stream/evaluation/current-build-results.csv", "model", "rtl-full"),
    ("spatter-current", "spatter/evaluation/current-build-smoke-results.csv", "model", "rtl-single"),
    ("stream-current", "stream/evaluation/current-build-smoke-results.csv", "model", "rtl"),
    ("spatter-current", "spatter/evaluation/app-trace-smoke-results.csv", "model", "rtl"),
    ("spatter-current", "spatter/evaluation/composition-ordered-pair.csv", "model", "rtl"),
    ("spatter-current", "spatter/evaluation/composition-model-results.csv", "model", "model"),
    ("spatter-current", "spatter/evaluation/current-build-memory.csv", "model", "model"),
    ("spatter-current", "spatter/evaluation/gpu-stream-revision-memory.csv", "model", "model"),
    ("spatter-current", "spatter/evaluation/xrage9-revision-memory.csv", "model", "model"),
    ("stream-current", "stream/evaluation/current-build-memory.csv", "model", "model"),
    ("spatter-current", "spatter/evaluation/composition-memory.csv", "model", "model"),
)
FILE_FIELDS = (
    "root", "path", "run", "run_status", "bytes", "sha256",
    "changed_during_capture", "in_metadata_archive",
)
RUN_FIELDS = (
    "root", "run", "status", "simulator", "correctness", "kind",
    "pattern_length", "count", "original_count", "count_overridden",
    "logical_payload_bytes", "gpu_cycles", "model_steps", "expected_digest",
    "output_digest", "elf_sha256", "elf_hash_matches", "suite", "suite_sha256",
    "suite_hash_matches", "kernel_source_sha256", "radiance_revision",
    "cyclotron_revision", "simulator_path", "simulator_sha256",
    "checker_source_sha256", "timing_config_sha256", "result_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_bytes(fields: tuple[str, ...], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def add_archive_member(archive: tarfile.TarFile, name: str, contents: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(contents)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(contents))


def nearest_run(path: Path, root: Path, results: dict[str, dict]) -> str:
    for parent in (path.parent, *path.parents):
        if parent == root:
            break
        key = parent.relative_to(root).as_posix()
        if key in results:
            return key
    return ""


def verify_reported_rows(runs: list[dict]) -> int:
    indexed = {(row["root"], row["run"]): row for row in runs}
    count = 0
    for root, table, model_phase, rtl_phase in RESULT_TABLES:
        with (KERNELS / table).open(newline="") as source:
            for reported in csv.DictReader(source):
                phase = (rtl_phase if "Verilator" in reported.get("simulator", "")
                         or "VCS" in reported.get("simulator", "") else model_phase)
                key = (root, f"{phase}/{reported['run']}")
                actual = indexed.get(key)
                if actual is None:
                    raise ValueError(f"{table}: missing run {key}")
                if (reported.get("status") and
                        reported["status"] != actual["status"]):
                    raise ValueError(f"{table}: status differs for {key}")
                if reported["gpu_cycles"] != str(actual["gpu_cycles"]):
                    raise ValueError(f"{table}: cycles differ for {key}")
                count += 1
    return count


def main(workspace: Path) -> None:
    roots = {
        "spatter-current": KERNELS / "spatter/runs",
        "stream-current": KERNELS / "stream/runs",
    }
    roots.update((key, workspace / name) for key, name in PRIOR_ROOTS.items())
    roots = {key: value for key, value in roots.items() if value.is_dir()}
    if not {"spatter-current", "stream-current"}.issubset(roots):
        raise FileNotFoundError("current Spatter and STREAM run roots are required")

    files: list[dict] = []
    runs: list[dict] = []
    archived: list[tuple[str, bytes]] = []
    outside_hashes: dict[Path, str] = {}
    mismatches: list[str] = []
    for root_name, root in roots.items():
        results: dict[str, dict] = {}
        for result in sorted(root.rglob("result.json")):
            if result.is_symlink():
                continue
            key = result.parent.relative_to(root).as_posix()
            results[key] = json.loads(result.read_text())
        file_hashes: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            run = nearest_run(path, root, results)
            before = path.stat()
            digest = sha256(path)
            after = path.stat()
            changed = (before.st_size, before.st_mtime_ns) != (
                after.st_size, after.st_mtime_ns
            )
            if changed:
                digest = ""
            archive_this = not changed and path.name != "kernel.soc.elf"
            if archive_this:
                contents = path.read_bytes()
                if hashlib.sha256(contents).hexdigest() != digest:
                    archive_this = False
                else:
                    archived.append((f"{root_name}/{relative}", contents))
            files.append({
                "root": root_name, "path": relative, "run": run,
                "run_status": results.get(run, {}).get("status", ""),
                "bytes": after.st_size, "sha256": digest,
                "changed_during_capture": str(changed).lower(),
                "in_metadata_archive": str(archive_this).lower(),
            })
            file_hashes[relative] = digest
        for run, data in sorted(results.items()):
            elf_hash = file_hashes.get(f"{run}/kernel.soc.elf", "")
            declared_elf = data.get("elf_sha256") or ""
            elf_match = ("yes" if declared_elf and elf_hash == declared_elf else
                         "no" if declared_elf and elf_hash else "unverified")
            if elf_match == "no":
                mismatches.append(f"{root_name}/{run}: ELF SHA-256 mismatch")
            suite_value = data.get("suite")
            suite = (suite_value if isinstance(suite_value, str) else
                     json.dumps(suite_value, separators=(",", ":"))
                     if suite_value is not None else "")
            suite_hash = data.get("suite_sha256") or ""
            suite_match = "unverified"
            if isinstance(suite_value, str) and suite_hash and Path(suite).is_file():
                suite_path = Path(suite)
                if suite_path not in outside_hashes:
                    outside_hashes[suite_path] = sha256(suite_path)
                suite_match = "yes" if outside_hashes[suite_path] == suite_hash else "no"
                if suite_match == "no":
                    mismatches.append(f"{root_name}/{run}: input suite SHA-256 mismatch")
            simulator = data.get("simulator_path") or ""
            simulator_hash = ""
            if simulator and Path(simulator).is_file():
                simulator_path = Path(simulator)
                if simulator_path not in outside_hashes:
                    outside_hashes[simulator_path] = sha256(simulator_path)
                simulator_hash = outside_hashes[simulator_path]
            runs.append({
                "root": root_name, "run": run, "status": data.get("status", ""),
                "simulator": data.get("simulator", ""),
                "correctness": data.get("correctness", ""),
                "kind": data.get("kind", ""),
                "pattern_length": data.get("pattern_length", ""),
                "count": data.get("count", ""),
                "original_count": data.get("original_count", ""),
                "count_overridden": data.get("count_overridden", ""),
                "logical_payload_bytes": data.get("logical_payload_bytes", ""),
                "gpu_cycles": data.get("gpu_cycles", ""),
                "model_steps": data.get("model_steps", ""),
                "expected_digest": data.get("expected_digest", ""),
                "output_digest": data.get("output_digest", ""),
                "elf_sha256": declared_elf or elf_hash,
                "elf_hash_matches": elf_match,
                "suite": suite, "suite_sha256": suite_hash,
                "suite_hash_matches": suite_match,
                "kernel_source_sha256": data.get("kernel_source_sha256", ""),
                "radiance_revision": data.get("radiance_revision", ""),
                "cyclotron_revision": data.get("cyclotron_revision", ""),
                "simulator_path": simulator, "simulator_sha256": simulator_hash,
                "checker_source_sha256": data.get("checker_source_sha256", ""),
                "timing_config_sha256": data.get("timing_config_sha256", ""),
                "result_sha256": file_hashes.get(f"{run}/result.json", ""),
            })
    if mismatches:
        raise ValueError("\n".join(mismatches))
    validated_rows = verify_reported_rows(runs)
    file_csv = csv_bytes(FILE_FIELDS, files)
    run_csv = csv_bytes(RUN_FIELDS, runs)
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "artifacts.csv").write_bytes(file_csv)
    (HERE / "runs.csv").write_bytes(run_csv)
    with (HERE / "raw-metadata.tar.gz").open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                add_archive_member(archive, "inventory/artifacts.csv", file_csv)
                add_archive_member(archive, "inventory/runs.csv", run_csv)
                for name, contents in sorted(archived):
                    add_archive_member(archive, name, contents)
    print(f"indexed {len(runs)} runs and {len(files)} files across {len(roots)} roots; "
          f"archived {len(archived)} non-ELF files; "
          f"validated {validated_rows} published current-build rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=KERNELS.parent.parent,
                        help="Chipyard workspace containing historical Spatter roots")
    args = parser.parse_args()
    main(args.workspace.resolve())
