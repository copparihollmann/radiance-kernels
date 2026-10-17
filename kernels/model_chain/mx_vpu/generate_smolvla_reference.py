#!/usr/bin/env python3
"""Run pinned upstream SmolVLA with an independent MX/VPU action projection.

Only the selected projection changes: E4M3 MX matmul and BF16 VPU bias addition.
All camera, VLM, expert and denoising dependencies remain in the upstream policy.
An original FP32 golden is mandatory so action drift is reported separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "evaluation/llm"))
from mx_vpu import reference
from verify_smolvla_policy import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--original-golden", type=Path, required=True)
    parser.add_argument("--projection-module", choices=("action_in_proj", "state_proj"), default="action_in_proj")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch

    def projection(value, weight, bias):
        if value.device.type != "cpu" or value.dtype != torch.float32:
            raise ValueError("format golden requires the pinned FP32 CPU policy")
        result = reference.linear(value.detach().numpy(), weight.detach().numpy().T)
        if bias is not None:
            result = reference.add(result, bias.detach().numpy())
        return torch.from_numpy(result)

    original = json.loads(args.original_golden.read_text())
    result = run(args.checkpoint_dir, True, action_projection=projection,
                 projection_module=args.projection_module)
    if original["checkpoint_weight_sha256"] != result["checkpoint_weight_sha256"] or original["input_sha256"] != result["input_sha256"]:
        raise ValueError("original golden uses different checkpoint or inputs")
    actual = np.asarray(result["output_values"], np.float32)
    baseline = np.asarray(original["output_values"], np.float32)
    delta = np.abs(actual-baseline)
    failures = int(np.count_nonzero(delta > 1e-3 + 1e-3*np.abs(baseline)))
    result.update(execution_backend="rakanic_mx_vpu",
                  projection_module=args.projection_module,
                  format_reference_sha256=hashlib.sha256((HERE/"reference.py").read_bytes()).hexdigest(),
                  format_reference_produced=True,
                  original_action_comparison=dict(passed=failures == 0, failed_elements=failures,
                      rtol=1e-3, atol=1e-3, max_abs_error=float(delta.max()),
                      original_golden_sha256=hashlib.sha256(args.original_golden.read_bytes()).hexdigest()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["original_action_comparison"]))


if __name__ == "__main__":
    main()
