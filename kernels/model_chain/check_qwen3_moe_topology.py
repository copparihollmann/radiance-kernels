#!/usr/bin/env python3
"""Compare a reduced Qwen3 MoE decoder schedule with the pinned 48-layer graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "evaluation/llm"))
import reference  # noqa: E402
import stitch  # noqa: E402


def check() -> dict:
    pinned = stitch.model_specs()["qwen3_30b_a3b"]
    reduced = reference.reduced_spec(pinned)
    reduced.update(num_hidden_layers=48, num_experts=16,
                   num_experts_per_tok=8, moe_intermediate_size=16)
    full = stitch.build("qwen3_30b_a3b", prefill=3, decode_steps=2,
                        generation="greedy", specs={"qwen3_30b_a3b": pinned})
    small = stitch.build("qwen3_30b_a3b", prefill=3, decode_steps=2,
                         generation="greedy", specs={"qwen3_30b_a3b": reduced})
    fields = ("id", "op", "reads", "writes")
    if (len(full.stages) != 3467 or len(small.stages) != 3467
            or any(any(a[field] != b[field] for field in fields)
                   for a, b in zip(full.stages, small.stages))
            or full.outputs != small.outputs
            or [step["token_input"] for step in full.execution_schedule["decode_loop"]]
            != [step["token_input"] for step in
                small.execution_schedule["decode_loop"]]):
        raise AssertionError("reduced graph changes the pinned stage dependencies")
    return {
        "schema": "radiance.qwen3-moe-full-depth-topology-check.v1",
        "status": "passed",
        "scope": "48-layer reduced-dimension synthetic graph against pinned full-dimension dependency graph",
        "ordered_stages_compared": len(full.stages),
        "ordered_stage_ids_ops_reads_writes_equal": True,
        "outputs_equal": True,
        "greedy_decode_token_dependencies_equal": True,
        "reduced_dimensions": {key: reduced[key] for key in (
            "hidden_size", "num_attention_heads", "num_key_value_heads",
            "head_dim", "num_experts", "num_experts_per_tok",
            "moe_intermediate_size", "vocab_size")},
        "pinned_config_sha256": pinned["source_sha256"],
        "graph_builder_sha256": hashlib.sha256(
            Path(stitch.__file__).read_bytes()).hexdigest(),
        "device_execution": False,
        "performance_measurement": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"matched {result['ordered_stages_compared']} ordered stages")


if __name__ == "__main__":
    main()
