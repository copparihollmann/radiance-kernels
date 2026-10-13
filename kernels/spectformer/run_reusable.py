#!/usr/bin/env python3
"""Run one input through a reusable SpectFormer Radiance ELF in Cyclotron."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from prepare import sha256
from reference import load_input, logits


def run(target: Path, input_path: Path, out: Path, simulator_root: Path,
        timeout: int, sim_cycles: int, check_reference: bool = True) -> dict:
    target = target.resolve()
    out.mkdir(parents=True, exist_ok=True)
    out = out.resolve()
    build = json.loads((target / "manifest.json").read_text())
    if not build.get("runtime_input") or not build.get("device_elf_built"):
        raise ValueError("build a reusable ELF with --runtime-input --build")
    elf = target / "kernel.radiance.elf"
    if sha256(elf) != build["radiance_elf_sha256"]:
        raise ValueError("ELF differs from build manifest")
    image_manifest = Path(build["source_image_manifest"])
    image = json.loads(image_manifest.read_text())
    weights = image_manifest.parent / "weights.bin"
    if (sha256(weights) != image["weights_sha256"] or
            sha256(weights) != build["weights_sha256"]):
        raise ValueError("weights differ from export")
    pixels = load_input(input_path, image["image_size"])
    input_bytes = out / "input.bin"
    input_bytes.write_bytes(pixels.tobytes())
    expected = None
    if check_reference:
        if image["format"] == "spectformer.hierarchical.fp32.v1":
            from hierarchical_prepare import reference_logits
            expected = reference_logits(image, pixels).reshape(-1)
        else:
            expected = logits(image, pixels).reshape(-1)
        (out / "reference-logits.bin").write_bytes(expected.tobytes())

    simulator_root = simulator_root.resolve()
    simulator = simulator_root / "target/release/cyclotron"
    source = (simulator_root / "config.toml").read_text()
    config_text, count = re.subn(r"(?m)^timeout\s*=\s*\d+",
                                 f"timeout = {sim_cycles}", source, count=1)
    if count != 1:
        raise ValueError("Cyclotron config has no unique timeout")
    config_text = config_text.replace('"config/timing/',
                                      f'"{simulator_root}/config/timing/')
    config = out / "cyclotron-config.toml"
    config.write_text(config_text)
    toolchain = Path(os.environ.get(
        "RISCV_TOOLCHAIN_PATH", "/scratch/agustin/projects/chipyard/.conda-env/riscv-tools"))
    symbols = subprocess.check_output(
        [str(toolchain / "bin/riscv64-unknown-elf-nm"), "-n", str(elf)], text=True)
    arenas = re.findall(r"(?m)^([0-9a-fA-F]+) [A-Za-z] v_arena$", symbols)
    if len(arenas) != 1:
        raise ValueError("ELF must contain one v_arena")
    dump = out / "logits.bin"
    dump.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
    environment["CYCLOTRON_WEIGHTS"] = ",".join((
        f"0x{image['base_address']:x}:{weights.resolve()}",
        f"0x{build['input_address']:x}:{input_bytes.resolve()}",
    ))
    environment["CYCLOTRON_DUMP_GMEM"] = (
        f"0x{int(arenas[0], 16) + build['logits_offset']:x}:"
        f"{image['num_classes'] * 4}:{dump}")
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
    log = out / "cyclotron.log"
    log.write_text(output)
    complete = (exit_code == 0 and "isa-test passed with tohost=0" in output and
                dump.is_file() and dump.stat().st_size == image["num_classes"] * 4 and
                f"preloaded {weights.stat().st_size} bytes of weights "
                f"@0x{image['base_address']:x}" in output and
                f"preloaded {input_bytes.stat().st_size} bytes of weights "
                f"@0x{build['input_address']:x}" in output)
    comparison = None
    if expected is not None and dump.is_file() and dump.stat().st_size == expected.nbytes:
        actual = np.fromfile(dump, dtype="<f4")
        finite = bool(np.isfinite(actual).all())
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        close = finite and bool(np.allclose(actual, expected,
                                           rtol=build["logit_rtol"],
                                           atol=build["logit_atol"]))
        comparison = {
            "all_finite": finite, "all_close": close,
            "max_abs_error": float(np.max(np.abs(delta))) if finite else None,
            "rms_error": float(np.sqrt(np.mean(delta * delta))) if finite else None,
            "top1_agrees": bool(np.argmax(actual) == np.argmax(expected)) if finite else False,
            "logits_sha256": sha256(dump),
        }
    passed = complete and (comparison is None if not check_reference else
                           comparison is not None and comparison["all_close"])
    cycles = re.findall(r"simulation finished after (\d+) cycles", output)
    record = {
        "model": build["model"], "status": "passed" if passed else "failed",
        "failure_reason": None if passed else ("wall_clock_timeout" if exit_code == 124 else
            "logit_mismatch" if complete else "simulator_or_program_failure"),
        "variant": image.get("variant", "vanilla"),
        "variant_depth": image.get("depth"), "image_size": image["image_size"],
        "weight_provenance": image["weight_provenance"],
        "input_path": str(input_path.resolve()), "input_sha256": sha256(input_bytes),
        "checkpoint_sha256": image["checkpoint_sha256"],
        "weights_sha256": sha256(weights), "elf_sha256": sha256(elf),
        "simulator_sha256": sha256(simulator),
        "config_sha256": sha256(config), "log_sha256": sha256(log),
        "logit_comparison": comparison,
        "reference_checked": check_reference,
        "cycles_functional": int(cycles[0]) if len(cycles) == 1 else None,
        "performance_measurement": False, "rtl_execution": False,
        "exit_code": exit_code,
    }
    (out / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cyclotron-root", type=Path,
                        default=Path("/scratch/agustin/projects/chipyard/generators/radiance/cyclotron"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--sim-cycles", type=int, default=100_000_000)
    parser.add_argument("--no-reference", action="store_true",
                        help="run inference and dump logits without importing upstream PyTorch")
    args = parser.parse_args()
    result = run(args.target, args.input, args.out, args.cyclotron_root,
                 args.timeout, args.sim_cycles, not args.no_reference)
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
