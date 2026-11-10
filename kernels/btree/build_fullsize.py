#!/usr/bin/env python3
"""Build and functionally gate the deterministic full-size B+Tree profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import model_check
import run


ROOT = Path(__file__).resolve().parent
SMS = (1, 2, 4)
ROBUST_POST_SPIN = 20_000
ADDR_HASH_CONTRACT = (
    "unit=32 slices=4 size=0x80000000 gpu_base=0x100000000"
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def confined(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"output must remain under {ROOT}: {resolved}")
    return resolved


def build_case(out: Path, sms: int, loader: str, *, robust: bool,
               spin_until_reset: bool) -> dict:
    parameters = run.profile_parameters("performance")
    case = run.prepare(
        *parameters, out, sms=sms, loader=loader,
        robust_end_drain=robust,
        robust_spin_until_reset=spin_until_reset,
        robust_serial_fence=robust,
        robust_post_spin=ROBUST_POST_SPIN if robust else 500,
    )
    return run.build(case, out)


def section_text(elf: Path, section: str) -> str:
    proc = subprocess.run(
        ["readelf", "-p", section, str(elf)], capture_output=True, text=True,
    )
    if proc.returncode:
        raise ValueError(f"missing {section} in {elf}: {proc.stderr.strip()}")
    return proc.stdout


def artifact_record(result: dict) -> dict:
    result_path = Path(result["elf"]).parent / "result.json"
    return {
        "sm_count": result["sm_count"],
        "loader": result["loader"],
        "robust_end_drain": result["robust_end_drain"],
        "robust_spin_until_reset": result["robust_spin_until_reset"],
        "robust_serial_fence": result["robust_serial_fence"],
        "elf": result["elf"],
        "elf_sha256": result["elf_sha256"],
        "device_elf": result["device_elf"],
        "device_elf_sha256": result["device_elf_sha256"],
        "result": str(result_path.resolve()),
        "result_sha256": sha256(result_path),
    }


def model_gate(cyclotron: Path, result: dict, out: Path,
               timeout: int) -> dict:
    source = Path(result["elf"]).parent / "result.json"
    checked = model_check.run_one(cyclotron, source, out, timeout)
    if checked["status"] != "passed":
        raise RuntimeError(f"functional model failed: {source}")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cyclotron", type=Path,
        default=ROOT.parents[2] / "radiance/cyclotron/target-native/release/cyclotron",
    )
    parser.add_argument(
        "--riscv64-toolchain", type=Path,
        default=Path(os.environ.get(
            "RISCV64_TOOLCHAIN_PATH",
            ROOT.parents[3] / "toolchains/riscv-tools-prefix/riscv-tools",
        )),
    )
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runs/fullsize-performance")
    parser.add_argument("--timeout", type=int, default=2_000_000_000)
    args = parser.parse_args()
    out = confined(args.out)
    cyclotron = args.cyclotron.resolve()
    if not cyclotron.is_file():
        parser.error(f"missing Cyclotron binary: {cyclotron}")
    toolchain = args.riscv64_toolchain.resolve()
    if not (toolchain / "bin/riscv64-unknown-elf-g++").is_file():
        parser.error(f"missing RV64 compiler under toolchain: {toolchain}")
    os.environ["RISCV64_TOOLCHAIN_PATH"] = str(toolchain)

    clean_artifacts = []
    clean_models = []
    for sms in SMS:
        artifact = build_case(out / "logical" / f"{sms}sm", sms,
                              "logical", robust=False,
                              spin_until_reset=False)
        clean_artifacts.append(artifact_record(artifact))
        clean_models.append(model_gate(
            cyclotron, artifact, out / "model-logical" / f"{sms}sm",
            args.timeout,
        ))

    # The direct-hashed FPGA image must spin until host reset, which a pure
    # functional model cannot terminate. Build the same robust epilogue with
    # only that FPGA handshake disabled and gate all useful work, all outputs,
    # three guards, serialized fences, and cache-capacity drains in Cyclotron.
    robust_model_artifacts = []
    robust_models = []
    for sms in SMS:
        artifact = build_case(out / "logical-robust-model" / f"{sms}sm", sms,
                              "logical", robust=True,
                              spin_until_reset=False)
        robust_model_artifacts.append(artifact_record(artifact))
        robust_models.append(model_gate(
            cyclotron, artifact,
            out / "model-logical-robust" / f"{sms}sm", args.timeout,
        ))

    direct_artifacts = []
    for sms in SMS:
        artifact = build_case(
            out / "direct-hashed-robust" / f"{sms}sm", sms,
            "direct-hashed", robust=True, spin_until_reset=True,
        )
        if not artifact["gpu_address_hash"]:
            raise RuntimeError("direct-hashed artifact lacks hash metadata")
        if ADDR_HASH_CONTRACT not in section_text(
                Path(artifact["elf"]), ".radiance_addr_hash"):
            raise RuntimeError("direct-hashed artifact has wrong hash contract")
        direct_artifacts.append(artifact_record(artifact))

    manifest = {
        "schema": "radiance-btree-fullsize-campaign-v1",
        "status": "passed",
        "workload_profile": "derived-full-size-performance",
        "parameters": dict(zip(
            ("keys", "point_queries", "range_queries", "range_span"),
            run.profile_parameters("performance"),
        )),
        "semantics": "exact-point-and-inclusive-exact-endpoint-range",
        "oracle": "independent-python-bisect-left-lower-bound",
        "sm_counts": list(SMS),
        "logical_artifacts": clean_artifacts,
        "logical_model_results": clean_models,
        "robust_model_artifacts": robust_model_artifacts,
        "robust_model_results": robust_models,
        "direct_hashed_robust_artifacts": direct_artifacts,
        "direct_hash_contract": ADDR_HASH_CONTRACT,
        "direct_hashed_model_note": (
            "The FPGA-only spin-until-reset image cannot terminate in "
            "Cyclotron; the otherwise identical robust logical variant is "
            "functionally gated with that handshake disabled."
        ),
        "fpga_touched": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "campaign.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "logical_cycles": [item["kernel_cycles"] for item in clean_models],
        "robust_model_cycles": [item["kernel_cycles"]
                                for item in robust_models],
    }, indent=2))


if __name__ == "__main__":
    main()
