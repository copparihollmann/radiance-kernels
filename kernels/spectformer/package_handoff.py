#!/usr/bin/env python3
"""Collect SpectFormer device artifacts and an explicit GMEM load map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_receipt_sources(name: str, receipt_path: Path, build: dict,
                            export: dict, output_kind: str,
                            output_count: int) -> dict:
    """Bind a numerical PASS to its raw simulator run and reference bytes."""
    receipt = json.loads(receipt_path.read_text())
    comparison = receipt.get("output_comparison")
    finite = receipt.get("all_finite")
    if finite is None and isinstance(comparison, dict):
        finite = comparison.get("all_finite")
    if (receipt.get("status") != "passed" or
            receipt.get("reference_checked") is not True or
            finite is not True):
        raise ValueError(f"{name}: simulator receipt lacks a checked numerical pass")
    if receipt.get("elf_sha256") != build["radiance_elf_sha256"]:
        raise ValueError(f"{name}: receipt used another ELF")
    if receipt.get("weights_sha256") != export["weights_sha256"]:
        raise ValueError(f"{name}: receipt used other weights")
    if receipt.get("input_sha256") != export["input_sha256"]:
        raise ValueError(f"{name}: receipt used another input")
    if receipt.get("output_kind", receipt.get("scope")) != output_kind:
        raise ValueError(f"{name}: receipt checked another output")
    if receipt.get("output_count") != output_count:
        raise ValueError(f"{name}: receipt checked another output length")

    if "run_result_sha256" in receipt:
        if receipt.get("out_of_tolerance") != 0:
            raise ValueError(f"{name}: verified output contains mismatches")
        raw_path = receipt_path.parent / "result.json"
        raw = json.loads(raw_path.read_text())
        if sha256(raw_path) != receipt["run_result_sha256"]:
            raise ValueError(f"{name}: raw simulator run differs from verification")
        if (raw.get("status") != "passed" or raw.get("reference_checked") or
                raw.get("elf_sha256") != receipt["elf_sha256"] or
                raw.get("weights_sha256") != receipt["weights_sha256"] or
                raw.get("input_sha256") != receipt["input_sha256"]):
            raise ValueError(f"{name}: verification is not bound to the raw run")
        output_file = "logits.bin" if output_kind == "logits" else "output.bin"
        if sha256(receipt_path.parent / output_file) != receipt["output_sha256"]:
            raise ValueError(f"{name}: device output differs from verification")
        reference = receipt_path.parent / "verified-reference.bin"
        if sha256(reference) != receipt["reference_sha256"]:
            raise ValueError(f"{name}: reference differs from verification")
        sources = {
            "simulator-result.json": (raw_path, None),
            "simulator-verification.json": (receipt_path, None),
            "reference.bin": (reference, receipt["reference_sha256"]),
        }
    else:
        comparison = receipt.get("output_comparison")
        if not isinstance(comparison, dict) or comparison.get("all_close") is not True:
            raise ValueError(f"{name}: simulator result lacks an upstream comparison")
        output_file = "logits.bin" if output_kind == "logits" else "output.bin"
        if sha256(receipt_path.parent / output_file) != comparison["logits_sha256"]:
            raise ValueError(f"{name}: device output differs from receipt")
        reference_file = ("reference-logits.bin" if output_kind == "logits"
                          else "reference-output.bin")
        reference = receipt_path.parent / reference_file
        sources = {
            "simulator-result.json": (receipt_path, None),
            "reference.bin": (reference, None),
        }
    if reference.stat().st_size != output_count * 4:
        raise ValueError(f"{name}: wrong reference size")
    return sources


def package(name: str, build_dir: Path, export_dir: Path, receipt_path: Path | None,
            out: Path, nm: Path) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError(f"invalid case name: {name}")
    build = json.loads((build_dir / "manifest.json").read_text())
    export = json.loads((export_dir / "manifest.json").read_text())
    if not build.get("device_elf_built") or not build.get("runtime_input"):
        raise ValueError(f"{name}: requires a built runtime-input ELF")
    if Path(build["source_image_manifest"]).resolve() != (export_dir / "manifest.json").resolve():
        raise ValueError(f"{name}: build and export do not match")
    if build["weights_sha256"] != export["weights_sha256"]:
        raise ValueError(f"{name}: weight hashes differ")
    if build["image_sha256"] != export["image_sha256"]:
        raise ValueError(f"{name}: image hashes differ")
    if build["input_address"] != export["entries"]["input.image"]["address"]:
        raise ValueError(f"{name}: input addresses differ")
    if build["weights_base_address"] != export["base_address"]:
        raise ValueError(f"{name}: weight addresses differ")

    elf = build_dir / "kernel.radiance.elf"
    symbols = subprocess.check_output([str(nm), "-n", str(elf)], text=True)
    arena = re.findall(r"(?m)^([0-9a-fA-F]+) [A-Za-z] v_arena$", symbols)
    if len(arena) != 1:
        raise ValueError(f"{name}: expected one v_arena in device ELF")
    output_address = int(arena[0], 16) + build.get("output_offset", build["logits_offset"])
    output_count = build.get("output_count", export["num_classes"])
    output_kind = build.get("output_kind", "logits")
    case_dir = out / name
    sources = {
        "kernel.radiance.elf": (elf, build["radiance_elf_sha256"]),
        "kernel.soc.elf": (build_dir / "kernel.soc.elf", build["soc_elf_sha256"]),
        "weights.bin": (export_dir / "weights.bin", export["weights_sha256"]),
        "input.bin": (export_dir / "input.bin", export["input_sha256"]),
        "build-manifest.json": (build_dir / "manifest.json", None),
        "export-manifest.json": (export_dir / "manifest.json", None),
    }
    if output_kind == "logits":
        sources["reference.bin"] = (export_dir / "golden.bin", export["golden_sha256"])
    if receipt_path is not None:
        receipt = json.loads(receipt_path.read_text())
        sources.update(checked_receipt_sources(
            name, receipt_path, build, export, output_kind, output_count))
    else:
        receipt = None
    source_inventory = {}
    for filename, (source, expected_hash) in sources.items():
        actual_hash = sha256(source)
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError(f"{name}: {filename} differs from manifest")
        source_inventory[filename] = {
            "sha256": actual_hash, "bytes": source.stat().st_size}
    if source_inventory["input.bin"]["bytes"] != export["entries"]["input.image"]["size_bytes"]:
        raise ValueError(f"{name}: input size differs from export")
    if ("reference.bin" in source_inventory and
            source_inventory["reference.bin"]["bytes"] != output_count * 4):
        raise ValueError(f"{name}: output reference size differs from build")
    case_dir.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for filename, (source, _) in sources.items():
        actual_hash = source_inventory[filename]["sha256"]
        target = case_dir / filename
        shutil.copyfile(source, target)
        if sha256(target) != actual_hash:
            raise ValueError(f"{name}: copy of {filename} is corrupt")
        inventory[filename] = {"sha256": actual_hash, "bytes": target.stat().st_size}
    record = {
        "name": name,
        "model": build["model"],
        "variant": export.get("variant", "vanilla"),
        "image_size": export["image_size"],
        "weight_provenance": export["weight_provenance"],
        "output_kind": output_kind,
        "stage_count": build["stages"],
        "gmem_loads": [
            {"file": "weights.bin", "gpu_address": f"0x{build['weights_base_address']:08x}",
             "bytes": inventory["weights.bin"]["bytes"]},
            {"file": "input.bin", "gpu_address": f"0x{build['input_address']:08x}",
             "bytes": inventory["input.bin"]["bytes"]},
        ],
        "output": {"gpu_address": f"0x{output_address:08x}",
                   "host_address": f"0x{(output_address | 0x100000000):x}",
                   "dtype": "float32_le", "elements": output_count,
                   "bytes": output_count * 4},
        "comparison": {"reference_file": "reference.bin" if "reference.bin" in inventory else None,
                       "rtol": build["logit_rtol"], "atol": build["logit_atol"]},
        "simulator_status": receipt["status"] if receipt is not None else "no_completed_result",
        "simulator_kind": "Cyclotron" if receipt is not None else None,
        "fpga_status": "not_run",
        "self_contained_soc_elf": False,
        "host_checks_output": False,
        "files": inventory,
    }
    (case_dir / "load-map.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", nargs=4, action="append", required=True,
                        metavar=("NAME", "BUILD", "EXPORT", "RECEIPT_OR_DASH"))
    parser.add_argument("--nm", type=Path, default=Path(
        "/scratch/agustin/projects/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-nm"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for name, build, export, receipt in args.case:
        records.append(package(name, Path(build).resolve(), Path(export).resolve(),
                               None if receipt == "-" else Path(receipt).resolve(),
                               args.out, args.nm))
    (args.out / "manifest.json").write_text(json.dumps({
        "schema": "spectformer.fpga.handoff.v1", "cases": records,
        "fpga_execution": False,
    }, indent=2) + "\n")
    shutil.copyfile(Path(__file__).with_name("verify_handoff.py"),
                    args.out / "verify.py")
    lines = ["# SpectFormer FPGA handoff", "",
             "Each case has a fused SoC ELF, a GPU ELF, weights, one normalized",
             "FP32 NCHW input, a reference output, and `load-map.json`.", "",
             "The current fused SoC ELF contains the host and GPU code only.",
             "Before starting the GPU, load both binary segments at the GPU",
             "addresses in `gmem_loads`. Read the FP32 output at the address",
             "and length in `output`; compare every element with `reference.bin`",
             "using the case tolerance. The host program only waits for GPU",
             "completion and does not check or print the model output.", "",
             "The existing U250 FireSim queue used self-contained fused ELFs.",
             "That submission route needs a memory preload facility or a new",
             "self-contained ELF and a host readback check. These cases are",
             "therefore handoff inputs, not directly runnable queue jobs.", "",
             "Simulation pass/fail is recorded per case. A passing prefix",
             "checks only that intermediate tensor; it does not validate",
             "full-model inference. No case here has an FPGA result.", "",
             "Run `python3 verify.py .` after unpacking and before loading.", ""]
    (args.out / "README.md").write_text("\n".join(lines))
    print(json.dumps({"out": str(args.out.resolve()),
                      "cases": [(r["name"], r["simulator_status"]) for r in records]},
                     indent=2))


if __name__ == "__main__":
    main()
