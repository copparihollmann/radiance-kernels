#!/usr/bin/env python3
"""Run pinned upstream SmolVLA with an independent MX/VPU action projection.

Only the selected projection changes: E4M3 MX matmul and BF16 VPU bias addition.
All camera, VLM, expert and denoising dependencies remain in the upstream policy.
An original FP32 golden is mandatory so action drift is reported separately.
"""
import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "evaluation/llm"))
from mx_vpu.smolvla_reference import projection_function, qualify_result
from verify_smolvla_policy import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--original-golden", type=Path, required=True)
    parser.add_argument("--projection-module", default="action_in_proj",
                        help="linear module path relative to policy.model; checked against the pinned graph")
    parser.add_argument("--column-start", type=int, default=0)
    parser.add_argument("--column-count", type=int, help="MX/VPU output-column partition; other columns retain FP32")
    parser.add_argument("--fp32-vision-reference", action="store_true",
                        help="use the independently defined deployment FP32 vision arithmetic")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.column_start < 0 or (args.column_count is not None and args.column_count <= 0):
        parser.error('column partition must have a nonnegative start and positive count')
    projection = projection_function(args.column_start, args.column_count, args.fp32_vision_reference)

    result = run(args.checkpoint_dir, True, action_projection=projection,
                 projection_module=args.projection_module,
                 fp32_vision_reference=args.fp32_vision_reference)
    qualify_result(result,args.original_golden,args.projection_module,args.column_start,
                   args.column_count,args.fp32_vision_reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["original_action_comparison"]))


if __name__ == "__main__":
    main()
