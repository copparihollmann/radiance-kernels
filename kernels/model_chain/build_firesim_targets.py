#!/usr/bin/env python3
"""Build target-specific model ELFs from a portable FireSim handoff directory.

The caller must supply the address-hash parameters from the actual bitstream
configuration. This script cannot infer or certify that hardware configuration.
"""

import argparse
import json
from pathlib import Path

from package_firesim import digest, hash_folds, package_hashed, verify_package


def selected_models(bundle, requested, hash_size):
    models = json.loads((bundle / "bundle.json").read_text())["models"]
    by_name = {model["model"]: model for model in models}
    names = list(by_name) if not requested or "all" in requested else requested
    if len(names) != len(set(names)):
        raise ValueError("each model may be selected only once")
    unknown = set(names) - set(by_name)
    if unknown:
        raise ValueError(f"unknown models: {', '.join(sorted(unknown))}")
    selected = [by_name[name] for name in names]
    for model in selected:
        required = int(model["max_gpu_address_exclusive"], 0)
        if required > hash_size:
            raise ValueError(
                f"{model['model']} needs GPU addresses through {required:#x}; "
                f"declared hash size is only {hash_size:#x}")
        source = bundle / model["hbm_source_elf"]
        if not source.is_file():
            raise FileNotFoundError(source)
        if digest(source) != model["hbm_source_elf_sha256"]:
            raise ValueError(f"{source}: SHA-256 differs from bundle.json")
        donor = bundle / model["donor_elf"]
        if not donor.is_file() or not donor.with_suffix(donor.suffix + ".json").is_file():
            raise FileNotFoundError(f"missing donor ELF or receipt: {donor}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path.cwd(),
                        help="directory containing bundle.json and the donor ELFs")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory for target-specific ELFs and receipts")
    parser.add_argument("--model", action="append",
                        help="model key from bundle.json; repeat or omit for all")
    parser.add_argument("--hash-size", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--hash-unit", type=int, required=True)
    parser.add_argument("--hash-slices", type=int, required=True)
    parser.add_argument("--compact-hash", action="store_true",
                        help="store only occupied physical GPU load regions")
    parser.add_argument("--config-file", type=Path,
                        help="optional bitstream/HWDB config to hash into the report")
    parser.add_argument("--bitstream-id",
                        help="optional HWDB entry or bitstream identifier for the report")
    parser.add_argument("--dry-run", action="store_true",
                        help="check input files and capacity without writing ELFs")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    output = args.out.resolve()
    try:
        hash_folds(args.hash_size, args.hash_unit, args.hash_slices)
        models = selected_models(bundle, args.model, args.hash_size)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    config_hash = digest(args.config_file) if args.config_file else None
    report = {
        "bundle_manifest_sha256": digest(bundle / "bundle.json"),
        "declared_memory_map": {"hash_size": args.hash_size,
                                "unit": args.hash_unit,
                                "slices": args.hash_slices,
                                "compact_file_layout": args.compact_hash},
        "bitstream_id": args.bitstream_id,
        "config_file_sha256": config_hash,
        "config_match": "caller supplied; not checked against RTL or bitstream",
        "fpga_execution": False,
        "models": [],
    }
    for model in models:
        report["models"].append({
            "model": model["model"],
            "max_gpu_address_exclusive": model["max_gpu_address_exclusive"],
            "source_elf": model["hbm_source_elf"],
            "source_elf_sha256": model["hbm_source_elf_sha256"],
            "donor_elf": model["donor_elf"],
            "donor_elf_sha256": model["donor_elf_sha256"],
        })
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    for item in report["models"]:
        stem = item["donor_elf"].removesuffix(".soc.elf")
        target = output / f"{stem}.hbm.soc.elf"
        if target.exists() or target.with_suffix(target.suffix + ".json").exists():
            raise FileExistsError(f"refusing to replace existing output: {target}")
    for item in report["models"]:
        stem = item["donor_elf"].removesuffix(".soc.elf")
        target = output / f"{stem}.hbm.soc.elf"
        receipt = bundle / (item["donor_elf"] + ".json")
        package_hashed(bundle / item["source_elf"], [], target,
                       args.hash_size, args.hash_unit, args.hash_slices,
                       receipts=[receipt], compact=args.compact_hash)
        checked = verify_package(target.with_suffix(target.suffix + ".json"))
        item["target_elf"] = target.name
        item["target_elf_sha256"] = checked["sha256"]
        item["embedded_images_verified"] = checked["embedded_images_verified"]
        (output / "target-bundle.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"verified {item['model']}: {target}", flush=True)


if __name__ == "__main__":
    main()
