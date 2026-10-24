#!/usr/bin/env python3
"""Derive a registry-driven, single-card asset manifest from a copied handoff."""

import argparse
import json
from pathlib import Path


CONFIGS = {
    "firesim-1sm-dram": "1SM_DRAM",
    "firesim-2sm-dram": "2SM_DRAM",
    "firesim-2sm-dram-baseline": "2SM_DRAM_baseline",
    "firesim-2sm-hbm": "2SM_HBM",
    "firesim-4sm-dram": "4SM_DRAM",
}


def command(config: str) -> list[str]:
    return [
        "timeout", "--signal=TERM", "--kill-after=30", "1800",
        "bash", "{runner}", "-b", "{bitstream_config}",
        "--bdf", "{bdf}", "--fastloadmem", "{elf}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bdf", default="1a:00.0")
    args = parser.parse_args()

    spec = json.loads(args.input.read_text())
    ids = {target["id"] for target in spec["targets"]}
    if ids != set(CONFIGS):
        raise ValueError(f"unexpected target set: {sorted(ids)}")
    spec["source_manifest"] = str(args.input.resolve())
    spec["single_fpga_bdf"] = args.bdf
    spec["selection"] = (
        "run.sh -b selects the copied registry's matching archive and driver; "
        "the positional path is the workload ELF"
    )
    for target in spec["targets"]:
        config = CONFIGS[target["id"]]
        target["bdf"] = args.bdf
        target["bitstream_config"] = config
        target["run_argv"] = command(config)
        target.pop("reuse_argv", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    main()
