#!/usr/bin/env python3
"""Record a VCS run with the exact ELF and simulator hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
DEFAULT_SIM = (HERE.parents[2] / "sims/vcs/"
               "simv-chipyard.harness-RadianceSingleClusterFastConfig")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--elf", type=Path, default=HERE / "kernel.soc.elf")
    parser.add_argument("--manifest", type=Path,
                        help="generated decoder manifest; omit for the small handoff case")
    parser.add_argument("--sim", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--run-exit-code", type=int,
                        help="exit code of the VCS process, if available")
    parser.add_argument("--timeout-seconds", type=int,
                        help="wall-clock limit used for this run, if any")
    parser.add_argument("--out", type=Path,
                        default=HERE / "evaluation/rtl-handoff-result.json")
    args = parser.parse_args()
    log = args.log.read_text(errors="replace")
    failed = re.search(r"TEST FAILED with tohost=\s*(\d+)", log)
    finished = "$finish called from file" in log
    passed = (finished and not failed and "GPUResetAggregator.sv" in log
              and args.run_exit_code in (None, 0))
    time = re.search(r"\bTime:\s*(\d+) ps", log)
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        if not manifest["device_elf_built"] or not manifest["device_check_all_stages"]:
            parser.error("decoder must be built with --device-check-all-stages")
        checked_stages = (manifest["native_verified_float_stages"]
                          + manifest.get("native_verified_integer_stages", 0))
        if len(manifest["verified_tensors"]) != checked_stages:
            parser.error("decoder stage check list is incomplete")
        if sha256(args.elf) != manifest["soc_elf_sha256"]:
            parser.error("SoC ELF hash differs from build manifest")
        if "errors=0" not in (args.manifest.parent / "native.log").read_text():
            parser.error("native decoder check did not pass")
        case = manifest["model"]
        scope = manifest["scope"]
        verification = "device checks every stage; nonzero tohost asserts"
        checked = manifest["verified_tensors"]
        source = args.manifest.parent / "kernel.cpp"
    else:
        manifest = None
        case = "RMSNorm -> FP32 linear -> residual, M=3 K=16 N=16"
        scope = "synthetic_shared_buffer_handoff"
        verification = "device checks normalization, projection, and residual; nonzero tohost asserts"
        checked = [48, 48, 48]
        source = HERE / "kernel.cpp"
    result = {
        "case": case,
        "scope": scope,
        "simulation": "VCS RadianceSingleClusterFastConfig RTL",
        "status": "passed" if passed else "failed" if failed else "incomplete",
        "run_exit_code": args.run_exit_code,
        "timeout_seconds": args.timeout_seconds,
        "termination": "timeout" if args.run_exit_code == 124 else None,
        "tohost_failure_code": int(failed.group(1)) if failed else None,
        "verification": verification,
        "simulation_time_ps_includes_host_boot": int(time.group(1)) if time else None,
        "device_cycles": None,
        "elf_sha256": sha256(args.elf),
        "simulator_sha256": sha256(args.sim),
        "source_sha256": sha256(source),
        "log_sha256": sha256(args.log),
        "checkpoint_weights": False,
        "full_model_execution": False,
        "upstream_execution_equivalent": False,
        "performance_measurement": False,
    }
    if manifest:
        failure_code = int(failed.group(1)) if failed else None
        failure_stage = ((failure_code >> 16) - 1) if failure_code else None
        result.update({
            "layers": manifest["layers"],
            "prefill_tokens": manifest["prefill_tokens"],
            "decode_steps": manifest["decode_steps"],
            "stage_count": manifest["stages"],
            "checked_float_stages": manifest["native_verified_float_stages"],
            "checked_integer_stages": manifest.get("native_verified_integer_stages", 0),
            "failed_checked_stage": (
                checked[failure_stage]
                if failure_stage is not None and 0 <= failure_stage < len(checked)
                else None),
            "failed_element_index": ((failure_code & 0xffff) >> 1)
                if failure_code else None,
            "generated_weights": True,
            "original_model_dimensions": False,
            "numpy_reference_sha256": sha256(
                HERE.parent / "evaluation/llm/reference.py"),
            "portable_ops_sha256": sha256(HERE / "pipeline_math.hpp"),
        })
    else:
        result.update({
            "checked_output_elements_per_stage": checked,
            "input_generator_sha256": sha256(HERE / "gen_data.py"),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
