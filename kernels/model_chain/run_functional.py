#!/usr/bin/env python3
"""Run connected decoder ELFs on the Radiance Cyclotron functional ISA model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = ("tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma_2_2b_it")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(model: str, generated_root: Path, simulator: Path, config: Path,
        simulator_root: Path, timeout: int) -> dict:
    target = generated_root / model
    manifest = json.loads((target / "manifest.json").read_text())
    elf = target / "kernel.radiance.elf"
    if not manifest["device_elf_built"] or not manifest["device_check_all_stages"]:
        raise ValueError(f"{model}: build with --device-check-all-stages --build")
    checked_stages = (manifest["native_verified_float_stages"]
                      + manifest.get("native_verified_integer_stages", 0))
    if len(manifest["verified_tensors"]) != checked_stages:
        raise ValueError(f"{model}: device stage check list is incomplete")
    if sha256(elf) != manifest["radiance_elf_sha256"]:
        raise ValueError(f"{model}: ELF hash differs from build manifest")
    if "errors=0" not in (target / "native.log").read_text():
        raise ValueError(f"{model}: native check did not pass")
    argv = [str(simulator), str(config), "--binary-path", str(elf),
            "--gen-trace", "false"]
    env = os.environ.copy()
    env["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
    try:
        process = subprocess.run(argv, cwd=simulator_root, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, timeout=timeout)
        output = process.stdout
        return_code = process.returncode
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return_code = 124
    (target / "functional.log").write_text(output)
    cycle_matches = re.findall(r"simulation finished after (\d+) cycles", output)
    passed = (return_code == 0 and "isa-test passed with tohost=0" in output
              and len(cycle_matches) == 1)
    failure_reason = None
    if not passed:
        if return_code == 124:
            failure_reason = "wall_clock_timeout"
        elif return_code == 1 and output.rstrip().endswith("Error: 0"):
            failure_reason = "simulator_cycle_limit_reached"
        else:
            failure_reason = "device_or_simulator_failure"
    result = {
        "model": model,
        "scope": manifest["scope"],
        "layers": manifest["layers"],
        "prefill_tokens": manifest["prefill_tokens"],
        "decode_steps": manifest["decode_steps"],
        "stage_count": manifest["stages"],
        "checked_float_stages": manifest["native_verified_float_stages"],
        "checked_integer_stages": manifest.get("native_verified_integer_stages", 0),
        "simulation": "Cyclotron functional ISA model",
        "status": "passed" if passed else "failed",
        "failure_reason": failure_reason,
        "cycles_functional": int(cycle_matches[0]) if cycle_matches else None,
        "tohost": 0 if passed else None,
        "process_exit_code": return_code,
        "device_elf_sha256": sha256(elf),
        "generated_source_sha256": sha256(target / "kernel.cpp"),
        "device_source_files_sha256": {
            name: sha256(target / name)
            for name in manifest.get("device_source_files", ["kernel.cpp"])},
        "device_optimization": manifest.get("device_optimization", "O3"),
        "stages_per_device_object": manifest.get("stages_per_device_object", 0),
        "portable_ops_sha256": sha256(HERE / "pipeline_math.hpp"),
        "numpy_reference_sha256": sha256(ROOT / "kernels/evaluation/llm/reference.py"),
        "model_specs_sha256": manifest["source_spec_sha256"],
        "simulator_sha256": sha256(simulator),
        "config_sha256": sha256(config),
        "simulator_cycle_limit": int(re.search(
            r"(?m)^timeout\s*=\s*(\d+)", config.read_text()).group(1)),
        "log_sha256": sha256(target / "functional.log"),
        "checkpoint_weights": False,
        "original_model_dimensions": False,
        "upstream_execution_equivalent": False,
        "native_check_tolerance": {"rtol": 1e-4, "atol": 1e-5},
        "device_check_tolerance": {"rtol": 5e-3, "atol": 5e-4},
        "rtl_execution": False,
        "performance_measurement": False,
    }
    (target / "functional-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--generated-root", type=Path, default=HERE / "generated")
    parser.add_argument("--cyclotron-root", type=Path,
                        default=ROOT.parent / "generators/radiance/cyclotron")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sim-cycles", type=int,
                        help="override the Cyclotron cycle limit in a local config copy")
    parser.add_argument("--out", type=Path,
                        default=HERE / "evaluation/functional-results.json")
    args = parser.parse_args()
    simulator_root = args.cyclotron_root.resolve()
    simulator = simulator_root / "target/release/cyclotron"
    config = simulator_root / "config.toml"
    if not simulator.is_file() or not config.is_file():
        parser.error("Cyclotron binary or config is missing")
    if args.sim_cycles is not None:
        if args.sim_cycles <= 0:
            parser.error("--sim-cycles must be positive")
        source = config.read_text()
        modified, replacements = re.subn(
            r"(?m)^timeout\s*=\s*\d+", f"timeout = {args.sim_cycles}",
            source, count=1)
        if replacements != 1:
            parser.error("Cyclotron config has no unique simulation timeout")
        modified = modified.replace('"config/timing/',
                                    f'"{simulator_root}/config/timing/')
        args.generated_root.mkdir(parents=True, exist_ok=True)
        config = (args.generated_root / "cyclotron-config.toml").resolve()
        config.write_text(modified)
    results = [run(model, args.generated_root.resolve(), simulator, config,
                   simulator_root, args.timeout) for model in args.models]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    for item in results:
        print(f"{item['model']}: {item['status']} "
              f"({item['cycles_functional']} functional cycles)")
    if any(item["status"] != "passed" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
