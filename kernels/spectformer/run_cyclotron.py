#!/usr/bin/env python3
"""Run a SpectFormer Radiance ELF against its pinned FP32 upstream golden."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from prepare import sha256


def run(target: Path, simulator_root: Path, timeout: int, sim_cycles: int) -> dict:
    target = target.resolve()
    manifest = json.loads((target / "manifest.json").read_text())
    if not manifest["device_elf_built"]:
        raise ValueError("the complete SpectFormer ELF has not been built")
    elf = target / "kernel.radiance.elf"
    if sha256(elf) != manifest["radiance_elf_sha256"]:
        raise ValueError("ELF differs from build manifest")
    image_manifest = Path(manifest["source_image_manifest"])
    image = json.loads(image_manifest.read_text())
    data = image_manifest.parent / "image.bin"
    if sha256(data) != image["image_sha256"] or image["image_sha256"] != manifest["image_sha256"]:
        raise ValueError("weight/input/golden image differs from build")
    if image["base_address"] < 0x30000000 or (image["base_address"] +
        image["image_size_bytes"] > 0x7EE00000):
        raise ValueError("image overlaps reserved device memory")
    simulator_root = simulator_root.resolve()
    simulator = simulator_root / "target/release/cyclotron"
    source = (simulator_root / "config.toml").read_text()
    config_text, replacements = re.subn(r"(?m)^timeout\s*=\s*\d+",
                                        f"timeout = {sim_cycles}", source, count=1)
    if replacements != 1:
        raise ValueError("Cyclotron config has no unique timeout")
    config_text = config_text.replace('"config/timing/',
                                      f'"{simulator_root}/config/timing/')
    config = target / "cyclotron-config.toml"
    config.write_text(config_text)
    environment = os.environ.copy()
    environment["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
    environment["CYCLOTRON_WEIGHTS"] = f"0x{image['base_address']:x}:{data.resolve()}"
    toolchain = Path(environment.get(
        "RISCV_TOOLCHAIN_PATH", "/scratch/agustin/projects/chipyard/.conda-env/riscv-tools"))
    symbols = subprocess.check_output(
        [str(toolchain / "bin/riscv64-unknown-elf-nm"), "-n", str(elf)], text=True)
    arenas = re.findall(r"(?m)^([0-9a-fA-F]+) [A-Za-z] v_arena$", symbols)
    if len(arenas) != 1:
        raise ValueError("ELF must contain exactly one v_arena symbol")
    logits_address = int(arenas[0], 16) + manifest["logits_offset"]
    dump = target / "cyclotron-logits.bin"
    dump.unlink(missing_ok=True)
    environment["CYCLOTRON_DUMP_GMEM"] = (
        f"0x{logits_address:x}:{image['num_classes'] * 4}:{dump}")
    try:
        result = subprocess.run([str(simulator), str(config), "--binary-path", str(elf),
                                 "--gen-trace", "false"], cwd=simulator_root,
                                env=environment, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=timeout)
        output, exit_code = result.stdout, result.returncode
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        exit_code = 124
    log = target / "cyclotron.log"
    log.write_text(output)
    cycles = re.findall(r"simulation finished after (\d+) cycles", output)
    match = re.search(r"isa-test failed with tohost=(\d+)", output)
    tohost = int(match.group(1)) if match else None
    passed = (exit_code == 0 and "isa-test passed with tohost=0" in output
              and len(cycles) == 1 and
              f"preloaded {image['image_size_bytes']} bytes of weights "
              f"@0x{image['base_address']:x}" in output)
    error = None
    if dump.is_file() and dump.stat().st_size == image["num_classes"] * 4:
        actual = np.fromfile(dump, dtype="<f4")
        golden_entry = image["entries"]["output.logits"]
        with data.open("rb") as stream:
            stream.seek(golden_entry["offset_bytes"])
            expected = np.frombuffer(stream.read(golden_entry["size_bytes"]), dtype="<f4")
        finite = bool(np.isfinite(actual).all())
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        error = {
            "all_finite": finite,
            "max_abs_error": float(np.max(np.abs(difference))) if finite else None,
            "rms_error": float(np.sqrt(np.mean(difference * difference))) if finite else None,
            "top1_agrees": bool(np.argmax(actual) == np.argmax(expected)) if finite else False,
            "logits_dump_sha256": sha256(dump),
        }
    record = {
        "model": "spectformer_vanilla", "shape": [3, image["image_size"], image["image_size"]],
        "depth": image["depth"], "embed_dim": image["embed_dim"],
        "weight_provenance": image["weight_provenance"],
        "input_provenance": image.get("input_provenance", "torch.linspace(-1,1)"),
        "stage_count": manifest["stages"], "status": "passed" if passed else "failed",
        "failure_reason": None if passed else ("wall_clock_timeout" if exit_code == 124
                                         else "output_mismatch" if tohost is not None and tohost & 1
                                         else "simulator_or_program_failure"),
        "cycles_functional": int(cycles[0]) if cycles else None,
        "tohost": 0 if passed else tohost,
        "failed_logit_index": tohost >> 1 if tohost and tohost & 1 else None,
        "logit_comparison": error,
        "logit_rtol": manifest["logit_rtol"],
        "logit_atol": manifest["logit_atol"],
        "validated_against": "upstream SpectFormer.eval() FP32 CPU",
        "rtl_execution": False, "performance_measurement": False,
        "device_elf_sha256": sha256(elf), "image_sha256": sha256(data),
        "simulator_sha256": sha256(simulator), "config_sha256": sha256(config),
        "log_sha256": sha256(log), "process_exit_code": exit_code,
    }
    (target / "cyclotron-result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--cyclotron-root", type=Path,
                        default=Path("/scratch/agustin/projects/chipyard/generators/radiance/cyclotron"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--sim-cycles", type=int, default=100_000_000)
    args = parser.parse_args()
    result = run(args.target, args.cyclotron_root, args.timeout, args.sim_cycles)
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
