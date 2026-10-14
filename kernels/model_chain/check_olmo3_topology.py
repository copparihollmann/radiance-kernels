#!/usr/bin/env python3
"""Check reduced OLMo 3 dependencies against the pinned 32-layer graph."""

from __future__ import annotations

import argparse
from collections import Counter
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
    pinned_path = LLM / "olmo3-7b-graph.json"
    pinned_graph = json.loads(pinned_path.read_text())
    spec = stitch.model_specs()["olmo3_7b"]
    reduced = reference.reduced_spec(spec)
    reduced["num_hidden_layers"] = 32
    reduced["layer_types"] = spec["layer_types"][:32]
    generated = stitch.build("olmo3_7b", prefill=3, decode_steps=2,
                             generation="greedy", specs={"olmo3_7b": reduced})
    pinned, small = pinned_graph["stages"], generated.stages
    fields = ("id", "op", "reads", "writes")
    if (pinned_graph["stage_count"] != 2123 or len(pinned) != 2123
            or len(small) != 2123 or pinned_graph["outputs"] != generated.outputs
            or any(any(a[field] != b[field] for field in fields)
                   for a, b in zip(pinned, small))):
        raise AssertionError("reduced OLMo graph changes pinned dependencies")
    for a, b in zip(pinned, small):
        if a["op"] == "causal_gqa" and any(
                a["attrs"][key] != b["attrs"][key]
                for key in ("attention_type", "window")):
            raise AssertionError(f"{a['id']}: attention policy differs")
        if a["op"] == "rmsnorm" and a["attrs"]["norm_axis"] != b["attrs"]["norm_axis"]:
            raise AssertionError(f"{a['id']}: norm axis differs")
        if a["op"] == "rope_yarn" and any(
                a["attrs"][key] != b["attrs"][key]
                for key in ("theta", "scaling")):
            raise AssertionError(f"{a['id']}: YaRN parameters differ")
    for key in ("token_input", "output_logits"):
        if ([phase[key] for phase in pinned_graph["execution_schedule"]["decode_loop"]]
                != [phase[key] for phase in generated.execution_schedule["decode_loop"]]):
            raise AssertionError(f"greedy {key} dependency differs")
    counts = Counter(stage["attrs"]["attention_type"] for stage in small
                     if stage["op"] == "causal_gqa")
    if counts != {"sliding_attention": 72, "full_attention": 24}:
        raise AssertionError(f"unexpected attention order/counts: {counts}")
    return {
        "schema": "radiance.olmo3-full-depth-topology-check.v1",
        "status": "passed",
        "scope": "32-layer reduced-dimension graph against pinned full-dimension dependencies",
        "ordered_stages_compared": len(small),
        "ordered_stage_ids_ops_reads_writes_equal": True,
        "attention_policy_norm_axes_yarn_equal": True,
        "attention_stages_by_type": dict(counts),
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
