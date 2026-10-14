#!/usr/bin/env python3
"""Compare full-width Qwen3 device runs with full and last-token prefill logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verify_functional import sha


EVALUATION = Path(__file__).resolve().parent / "evaluation"


def saved(name: str) -> tuple[dict, Path]:
    path = EVALUATION / name
    return json.loads(path.read_text()), path


def verified(build_name: str, result_name: str,
             check_name: str) -> tuple[dict, dict, dict]:
    build, build_path = saved(build_name)
    result, result_path = saved(result_name)
    check, _ = saved(check_name)
    if (check["status"] != "passed" or result["status"] != "passed"
            or result["tohost"] != 0 or result["process_exit_code"] != 0
            or check["result_sha256"] != sha(result_path)
            or check["manifest_sha256"] != sha(build_path)
            or check["elf_sha256"] != result["device_elf_sha256"]
            or check["elf_sha256"] != build["radiance_elf_sha256"]
            or check["cycles_functional"] != result["cycles_functional"]
            or check["checked_stages"] != result["stage_count"]
            or not check["weight_image_checked"]
            or result["rtl_execution"] or result["performance_measurement"]):
        raise ValueError(f"unverified functional run: {result_name}")
    return build, result, check


def verify() -> dict:
    baseline, base_result, base_check = verified(
        "qwen3-full-depth-multitoken-fp16-build.json",
        "qwen3-full-depth-fp16-multitoken-functional.json",
        "qwen3-full-depth-fp16-multitoken-functional-verification.json")
    selected, last_result, last_check = verified(
        "qwen3-prefill-last-full-fp16-build.json",
        "qwen3-prefill-last-full-fp16-functional.json",
        "qwen3-prefill-last-full-fp16-verification.json")
    logit, logit_path = saved("qwen3-prefill-last-logit-check.json")
    for field in ("model", "scope", "layers", "prefill_tokens",
                  "decode_steps", "input_token_ids", "reference_generated_token_ids",
                  "source_config_sha256", "checkpoint_sha256",
                  "weight_image_sha256", "source_spec_sha256",
                  "checkpoint_weight_format", "generation"):
        if baseline[field] != selected[field]:
            raise ValueError(f"prefill selection changed {field}")
    if (baseline["model"] != "qwen3_0_6b" or baseline["layers"] != 28
            or baseline["prefill_tokens"] != 3 or baseline["decode_steps"] != 2
            or baseline["reference_generated_token_ids"] != [16, 8]
            or baseline.get("prefill_logits_policy") is not None
            or selected["prefill_logits_policy"] != "last"
            or baseline["stages"] != 1607 or selected["stages"] != 1608
            or base_result["checked_float_stages"] != 1605
            or last_result["checked_float_stages"] != 1606
            or base_result["checked_integer_stages"] != 2
            or last_result["checked_integer_stages"] != 2
            or last_result["prefill_logits_policy"] != "last"):
        raise ValueError("unexpected full-width Qwen3 mapping")
    for field in ("source_config_sha256", "checkpoint_sha256",
                  "weight_image_sha256"):
        if (base_result[field] != baseline[field]
                or last_result[field] != selected[field]):
            raise ValueError(f"device receipt changed {field}")
    if (logit["status"] != "passed"
            or logit["optimized_device_executed"] is not False
            or logit["saved_reference_output_sha256"] !=
            baseline["reference_output_sha256"]
            or logit["checkpoint_sha256"] != baseline["checkpoint_sha256"]
            or logit["weight_image_sha256"] != baseline["weight_image_sha256"]
            or logit["upstream_selected_tokens"] != [16, 8]
            or logit["selected_top1"] != 16
            or logit["saved_full_logit_top1"] != 16):
        raise ValueError("last-token arithmetic check differs from device fixture")
    before = base_result["cycles_functional"]
    after = last_result["cycles_functional"]
    if before <= after:
        raise ValueError("last-token mapping did not reduce functional cycles")
    return {
        "schema": "radiance.qwen3-full-prefill-selection-pair.v1",
        "status": "passed",
        "model": "qwen3_0_6b",
        "layers": 28,
        "prefill_tokens": 3,
        "decode_steps": 2,
        "generated_token_ids": [16, 8],
        "baseline_checked_stages": base_check["checked_stages"],
        "last_prefill_checked_stages": last_check["checked_stages"],
        "baseline_cycles_functional": before,
        "last_prefill_cycles_functional": after,
        "functional_cycle_difference": before - after,
        "functional_cycle_reduction_percent": round(100 * (before - after) / before, 6),
        "checkpoint_sha256": baseline["checkpoint_sha256"],
        "weight_image_sha256": baseline["weight_image_sha256"],
        "baseline_elf_sha256": base_check["elf_sha256"],
        "last_prefill_elf_sha256": last_check["elf_sha256"],
        "baseline_result_sha256": base_check["result_sha256"],
        "last_prefill_result_sha256": last_check["result_sha256"],
        "arithmetic_check_sha256": sha(logit_path),
        "rtl_execution": False,
        "performance_measurement": False,
        "source_sha256": sha(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verify(), indent=2) + "\n")


if __name__ == "__main__":
    main()
