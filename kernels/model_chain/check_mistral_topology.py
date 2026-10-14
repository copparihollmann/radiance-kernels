#!/usr/bin/env python3
"""Check the reduced 32-layer Mistral schedule against the pinned model graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
LLM = HERE.parent / "evaluation/llm"
sys.path.insert(0, str(LLM))
import reference  # noqa: E402
import stitch  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check() -> dict:
    pinned_path = LLM / "mistral-7b-v03-graph.json"
    pinned_graph = json.loads(pinned_path.read_text())
    spec = stitch.model_specs()["mistral_7b_v03"]
    reduced = reference.reduced_spec(spec)
    reduced["num_hidden_layers"] = 32
    generated = stitch.build("mistral_7b_v03", prefill=3, decode_steps=2,
                             generation="greedy", specs={"mistral_7b_v03": reduced})
    fields = ("id", "op", "reads", "writes")
    a, b = pinned_graph["stages"], generated.stages
    if (pinned_graph["stage_count"] != 1643 or len(a) != 1643
            or len(b) != 1643 or pinned_graph["outputs"] != generated.outputs
            or any(any(left[field] != right[field] for field in fields)
                   for left, right in zip(a, b))):
        raise AssertionError("reduced schedule changes pinned stage dependencies")
    expected_loop = pinned_graph["execution_schedule"]["decode_loop"]
    actual_loop = generated.execution_schedule["decode_loop"]
    for field in ("token_input", "output_logits"):
        if [step[field] for step in expected_loop] != [step[field] for step in actual_loop]:
            raise AssertionError(f"reduced schedule changes greedy {field}")
    return {
        "schema": "radiance.mistral-full-depth-topology-check.v1",
        "status": "passed",
        "scope": "32-layer reduced-dimension graph against pinned full-dimension dependencies",
        "ordered_stages_compared": len(a),
        "ordered_stage_ids_ops_reads_writes_equal": True,
        "outputs_equal": True,
        "greedy_decode_token_dependencies_equal": True,
        "reduced_dimensions": {key: reduced[key] for key in (
            "hidden_size", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "head_dim", "vocab_size")},
        "pinned_graph_sha256": sha256(pinned_path),
        "pinned_config_sha256": spec["source_sha256"],
        "graph_builder_sha256": sha256(Path(stitch.__file__)),
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
