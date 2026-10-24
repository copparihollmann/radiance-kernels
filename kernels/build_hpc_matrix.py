#!/usr/bin/env python3
"""Build the reproducible FPGA-1A HPC ELF matrix without rebuilding hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STREAM = HERE / "stream"
SPATTER = HERE / "spatter"
GEMM = HERE / "nightly/gemm_mx"
GPU_STREAM = SPATTER / "inputs/standard-suite/basic-tests/gpu-stream.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def source_state() -> dict:
    status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain=v1", "-z"]
    )
    digest = hashlib.sha256()
    digest.update(subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", "--binary", "HEAD", "--"]
    ))
    untracked = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "--others", "--exclude-standard", "-z"]
    ).split(b"\0")
    for raw in sorted(path for path in untracked if path):
        path = ROOT / raw.decode()
        digest.update(raw + b"\0")
        digest.update(path.read_bytes())
    return {
        "revision": revision(),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "worktree_sha256": digest.hexdigest(),
        "status": status.decode(errors="replace").replace("\0", "\n").splitlines(),
    }


def run(argv: list[str], cwd: Path = ROOT) -> None:
    subprocess.run(argv, cwd=cwd, check=True)


def record(path: Path, *, name: str, family: str, sms: int,
           loader: str, extra: dict | None = None) -> dict:
    result = json.loads((path / "result.json").read_text())
    elf = path / "kernel.soc.elf"
    row = {
        "name": name,
        "family": family,
        "sm_count": sms,
        "loader": loader,
        "elf": str(elf),
        "elf_sha256": sha256(elf),
        "bytes": elf.stat().st_size,
        "correctness": result.get("correctness"),
        "status": result["status"],
    }
    if extra:
        row.update(extra)
    return row


def build_stream(out: Path, sms: list[int], loaders: list[str],
                 elements: int) -> list[dict]:
    rows = []
    for loader in loaders:
        for sm in sms:
            for kind in ("copy", "scale", "add", "triad"):
                name = f"stream--{kind}--{sm}sm"
                target = out / loader / name
                run(["python3", str(STREAM / "run.py"), "--kind", kind,
                     "--elements", str(elements), "--sms", str(sm),
                     "--loader", loader, "--full-host-check", "--out", str(target)])
                rows.append(record(target, name=name, family="stream",
                                   sms=sm, loader=loader,
                                   extra={"operation": kind, "elements": elements}))
    return rows


def build_spatter(out: Path, sms: list[int], loaders: list[str]) -> list[dict]:
    rows = []
    for loader in loaders:
        for sm in sms:
            for case in range(5):
                name = f"spatter--case-{case}--{sm}sm"
                target = out / loader / name
                run(["python3", str(SPATTER / "run.py"), "--suite", str(GPU_STREAM),
                     "--case", str(case), "--sms", str(sm), "--loader", loader,
                     "--full-host-check", "--out", str(target), "--build-only"])
                rows.append(record(target, name=name, family="spatter",
                                   sms=sm, loader=loader, extra={"case": case}))
    return rows


def build_xrage(out: Path, sms: list[int], loaders: list[str],
                suite: Path) -> list[dict]:
    rows = []
    variants = (("gather", 0, None),
                ("scatter-ordered", 1, "ordered"),
                ("scatter-parallel", 1, "parallel"))
    for loader in loaders:
        for sm in sms:
            for variant, case, policy in variants:
                name = f"xrage--{variant}--{sm}sm"
                target = out / loader / name
                argv = ["python3", str(SPATTER / "run.py"), "--suite", str(suite),
                        "--case", str(case), "--sms", str(sm), "--loader", loader,
                        "--out", str(target), "--build-only"]
                if policy is not None:
                    argv += ["--collision-policy", policy]
                if policy != "parallel":
                    argv.append("--full-host-check")
                run(argv)
                rows.append(record(
                    target, name=name, family="xrage", sms=sm, loader=loader,
                    extra={"operation": variant, "case": case,
                           "collision_policy": policy or "parallel"},
                ))
    return rows


def build_gemm(out: Path, sms: list[int], loaders: list[str]) -> list[dict]:
    rows = []
    args = "--M 512 --N 512 --K 512 --TM 128 --TN 64 --TK 256 --pretile --bdup"
    for loader in loaders:
        run(["make", "clean"], GEMM)
        run(["make", "-j1", f"SMS={' '.join(map(str, sms))}", "FMT=fp8",
             f"ARGS={args}",
             f"MU_ADDR_HASH={1 if loader == 'direct-hashed' else 0}"], GEMM)
        for sm in sms:
            source = GEMM / f"build/{sm}sm/gemm_mx.soc.elf"
            target = out / loader / f"gemm--fp8--{sm}sm"
            target.mkdir(parents=True, exist_ok=True)
            elf = target / "kernel.soc.elf"
            shutil.copy2(source, elf)
            result = {
                "name": f"gemm--fp8--{sm}sm", "family": "fp8_gemm",
                "sm_count": sm, "loader": loader,
                "gpu_address_hash": loader == "direct-hashed",
                "shape": {"m": 512, "n": 512, "k": 512},
                "format": "mxfp8-e4m3", "vpu": False,
                "correctness": "nightly-device-full-and-host-spot-check",
                "status": "built", "elf_sha256": sha256(elf),
            }
            (target / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            rows.append({**result, "elf": str(elf), "bytes": elf.stat().st_size})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sms", nargs="+", type=int, choices=(1, 2, 4),
                        default=(1, 2, 4))
    parser.add_argument("--loaders", nargs="+",
                        choices=("logical", "direct-hashed"),
                        default=("logical", "direct-hashed"))
    parser.add_argument("--stream-elements", type=int, default=4102)
    parser.add_argument("--xrage-suite", type=Path,
                        help="suite emitted by spatter/tools/extract-xrage-bundle.py")
    parser.add_argument("--require-xrage", action="store_true",
                        help="fail instead of producing a matrix without xRAGE")
    parser.add_argument("--skip-gemm", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sms = sorted(set(args.sms))
    loaders = list(dict.fromkeys(args.loaders))
    rows = build_stream(out, sms, loaders, args.stream_elements)
    rows.extend(build_spatter(out, sms, loaders))
    if args.xrage_suite is not None:
        suite = args.xrage_suite.resolve()
        if not suite.is_file():
            parser.error(f"--xrage-suite is not a file: {suite}")
        rows.extend(build_xrage(out, sms, loaders, suite))
    elif args.require_xrage:
        parser.error("--require-xrage needs --xrage-suite")
    if not args.skip_gemm:
        rows.extend(build_gemm(out, sms, loaders))
    source = source_state()
    manifest = {
        "schema": "radiance-fpga-1a-hpc-build-v1",
        "source_revision": source["revision"],
        "source_state": source,
        "no_vpu": True,
        "hardware_rebuilt": False,
        "sms": sms,
        "loaders": loaders,
        "xrage_suite": str(args.xrage_suite.resolve()) if args.xrage_suite else None,
        "workloads": rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"built {len(rows)} ELF configurations under {out}")


if __name__ == "__main__":
    main()
