#!/usr/bin/env python3
"""Export the completed U250 follow-up with exact ELFs and upstream arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CAMPAIGNS = {
    "output-initialization": None,
    "native-traces": REPO / "kernels/spatter/evaluation/native-upstream-golden.csv",
    "gather-chunking": REPO / "kernels/spatter/evaluation/chunk-smoke-upstream.csv",
    "amg-gpu-chunks": REPO / "kernels/spatter/evaluation/amg-gpu-chunks-upstream.csv",
    "gpu-stream-full": None,
    "xrage-full": None,
}
INPUT_DIRS = (
    "diagnostic", "native", "native-golden", "chunk-smoke",
    "chunk-smoke-golden", "chunk-smoke-full-golden",
    "amg-gpu-native-chunks", "amg-gpu-native-golden", "amg-gpu-full-golden",
    "source-snapshots",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    args = parser.parse_args()
    inputs = args.inputs_root.resolve()
    upstream = args.upstream.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"archive already exists: {output}")
    if not args.baseline_archive.is_file():
        raise ValueError("baseline U250 archive is missing")
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != "ec8923711f8dc21eedff7189f12b02eb06845d2f":
        raise ValueError("upstream Spatter revision differs")
    subprocess.run([
        "python3", str(REPO / "kernels/spatter/tools/verify-native-evidence.py"),
        "--upstream", str(upstream), "--inputs-root", str(inputs),
    ], check=True)
    for name, golden in CAMPAIGNS.items():
        directory = HERE / name
        subprocess.run([
            "python3", str(HERE / "capture_cases.py"), "--plan", str(directory / "plan.csv"),
            "--out", str(directory), "--elf-root", str(inputs), "--require-complete",
        ], check=True)
        command = ["python3", str(HERE / "verify_cases.py"), str(directory)]
        if golden is not None:
            command.extend(["--golden-table", str(golden)])
        subprocess.run(command, check=True)
    subprocess.run(["python3", str(HERE / "verify.py")], check=True)
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"],
                               text=True).strip():
        raise ValueError("commit the completed records before exporting")
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   text=True).strip()
    files: dict[str, Path] = {}
    for directory, prefix in ((HERE, "evidence/firesim"),
                              (REPO / "kernels/spatter/evaluation", "evidence/spatter"),
                              (REPO / "kernels/spatter/inputs/standard-suite", "inputs/standard-suite")):
        for path in directory.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[f"{prefix}/{path.relative_to(directory).as_posix()}"] = path
    for directory_name in INPUT_DIRS:
        directory = inputs / directory_name
        if not directory.is_dir():
            raise ValueError(f"missing local artifact directory: {directory}")
        for path in directory.rglob("*"):
            if path.is_file():
                files[f"inputs/{path.relative_to(inputs).as_posix()}"] = path
    for name in CAMPAIGNS:
        for row in read_csv(HERE / name / "plan.csv"):
            for field in ("elf_path", "original_device_elf"):
                if not row[field]:
                    continue
                path = inputs / row[field]
                if not path.is_file():
                    raise ValueError(f"planned ELF missing: {path}")
                if field == "elf_path" and sha256(path) != row["elf_sha256"]:
                    raise ValueError(f"planned ELF hash changed: {path}")
                files[f"inputs/{row[field]}"] = path
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        feature_bundle = temp / "radiance-kernels-spatter-workloads.bundle"
        upstream_bundle = temp / "spatter-upstream.bundle"
        subprocess.run(["git", "-C", str(REPO), "bundle", "create",
                        str(feature_bundle), "HEAD"], check=True)
        subprocess.run(["git", "-C", str(upstream), "bundle", "create",
                        str(upstream_bundle), "HEAD"], check=True)
        files["source/radiance-kernels-spatter-workloads.bundle"] = feature_bundle
        files["source/spatter-upstream.bundle"] = upstream_bundle
        manifest = ["path,bytes,sha256"]
        for name, path in sorted(files.items()):
            manifest.append(f"{name},{path.stat().st_size},{sha256(path)}")
        readme = (
            f"Radiance STREAM/Spatter U250 follow-up.\n"
            f"radiance-kernels commit: {head}\n"
            f"upstream Spatter commit: {revision}\n"
            f"The earlier baseline U250 archive is separate: {args.baseline_archive.resolve()}\n"
            f"Baseline archive SHA-256: {sha256(args.baseline_archive)}\n"
            "Keep both archives. The baseline archive contains the bitstream and original campaign.\n"
            "This archive contains the later exact ELFs, upstream inputs/outputs, source snapshots,\n"
            "queue logs, and Git bundles. Use evidence/firesim/*/plan.csv and results.csv to distinguish\n"
            "completed guest checks from infrastructure failures.\n"
        ).encode()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w") as archive:
            for name, path in sorted(files.items()):
                archive.add(path, arcname=name, recursive=False)
            for name, data in (("README.txt", readme),
                               ("manifest.csv", ("\n".join(manifest) + "\n").encode())):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(data))
    print(f"{output}\nSHA-256 {sha256(output)}")


if __name__ == "__main__":
    main()
