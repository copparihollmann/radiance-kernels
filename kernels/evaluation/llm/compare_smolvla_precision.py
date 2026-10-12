#!/usr/bin/env python3
"""Measure the output change from casting pinned SmolVLA weights to FP32."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(checkpoint_path: Path, fp32_path: Path) -> dict:
    original = json.loads(checkpoint_path.read_text())
    promoted = json.loads(fp32_path.read_text())
    if (not original["passed"] or not promoted["passed"] or
            original["model"] != "smolvla_base" or
            promoted["model"] != "smolvla_base" or
            promoted["weights_promoted_to_fp32"] is not True or
            promoted["checkpoint_embedding_dtype"] != "torch.bfloat16" or
            promoted["execution_embedding_dtype"] != "torch.float32" or
            original["input_sha256"] != promoted["input_sha256"] or
            original["checkpoint_weight_sha256"] !=
            promoted["checkpoint_weight_sha256"]):
        raise ValueError("policy precision controls differ beyond weight dtype")
    baseline = np.asarray(original["output_values"], dtype=np.float32)
    alternate = np.asarray(promoted["output_values"], dtype=np.float32)
    if baseline.shape != (1, 50, 32) or alternate.shape != baseline.shape:
        raise ValueError("SmolVLA action-chunk shape differs")
    for record, value in ((original, baseline), (promoted, alternate)):
        if hashlib.sha256(value.tobytes()).hexdigest() != record["output_sha256"]:
            raise ValueError("policy output differs from recorded SHA-256")
    delta = alternate.astype(np.float64) - baseline.astype(np.float64)
    threshold = 1e-3 + 1e-3 * np.abs(baseline.astype(np.float64))
    legacy_threshold = 1e-2 + 1e-2 * np.abs(baseline.astype(np.float64))
    return {
        "model": "smolvla_base", "scope": "same_checkpoint_bf16_vs_fp32_policy",
        "checkpoint_weight_sha256": original["checkpoint_weight_sha256"],
        "input_sha256": original["input_sha256"],
        "checkpoint_precision_output_sha256": original["output_sha256"],
        "promoted_fp32_output_sha256": promoted["output_sha256"],
        "output_elements": int(baseline.size),
        "max_abs_error": float(np.max(np.abs(delta))),
        "rms_error": float(np.sqrt(np.mean(delta * delta))),
        "elements_outside_device_tolerance": int(np.count_nonzero(np.abs(delta) > threshold)),
        "comparison_tolerance": {"rtol": 1e-3, "atol": 1e-3},
        "elements_outside_1e_2_tolerance": int(
            np.count_nonzero(np.abs(delta) > legacy_threshold)),
        "policy_execution": True, "device_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-reference", type=Path, required=True)
    parser.add_argument("--fp32-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = compare(args.checkpoint_reference, args.fp32_reference)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"SmolVLA BF16→FP32: max {result['max_abs_error']:.6g}, "
          f"outside tolerance {result['elements_outside_device_tolerance']}/1600")


if __name__ == "__main__":
    main()
