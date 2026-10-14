#!/usr/bin/env python3
"""Check all pinned Nemotron blocks with cached Mamba and attention state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource

import numpy as np

import nemotron_graph_reference as nemotron
import stitch


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check() -> dict:
    pinned = stitch.model_specs()["nemotron3_nano_4b"]
    spec = nemotron.control_spec(pinned, len(pinned["layer_kinds"]))
    if (spec["layer_kinds"] != pinned["layer_kinds"] or
            {kind: spec["layer_kinds"].count(kind) for kind in set(spec["layer_kinds"])}
            != {"mamba2": 21, "full_attention": 4, "mlp_only": 17}):
        raise AssertionError("reduced block order differs from pinned model")
    graph = stitch.build("nemotron3_nano_4b", prefill=3, decode_steps=2,
                         generation="teacher_forced",
                         specs={"nemotron3_nano_4b": spec})
    full = stitch.build("nemotron3_nano_4b", prefill=5, decode_steps=0,
                        specs={"nemotron3_nano_4b": spec})
    ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    inputs = {"prefill.token_ids": ids[:, :3],
              "decode0.token_ids": ids[:, 3:4],
              "decode1.token_ids": ids[:, 4:5]}
    incremental = nemotron.execute(graph, inputs)
    complete = nemotron.execute(full, {"prefill.token_ids": ids})
    final_error = float(np.max(np.abs(
        incremental[graph.outputs[-1]] - complete[full.outputs[-1]])))
    if not np.allclose(incremental[graph.outputs[-1]], complete[full.outputs[-1]],
                       rtol=1e-4, atol=1e-5):
        raise AssertionError("cached decoder differs from full causal pass")

    greedy_graph = stitch.build("nemotron3_nano_4b", prefill=3, decode_steps=2,
                                generation="greedy",
                                specs={"nemotron3_nano_4b": spec})
    pinned_graph = json.loads((HERE / "nemotron3-nano-4b-graph.json").read_text())
    fields = ("id", "op", "reads", "writes")
    if (len(greedy_graph.stages) != len(pinned_graph["stages"]) or
            greedy_graph.outputs != pinned_graph["outputs"] or
            any(any(reduced[field] != original[field] for field in fields)
                for reduced, original in zip(greedy_graph.stages,
                                             pinned_graph["stages"]))):
        raise AssertionError("reduced graph differs from pinned stage topology")
    for reduced, original in zip(greedy_graph.stages, pinned_graph["stages"]):
        keys = ("window", "rotary") if reduced["op"] == "causal_gqa" else (
            ("mode", "causal", "activation") if reduced["op"] == "mamba_depthwise_conv"
            else ("mode", "dt_preprocessing", "A_preprocessing")
            if reduced["op"] == "mamba_selective_scan" else ())
        if any(reduced["attrs"].get(key) != original["attrs"].get(key)
               for key in keys):
            raise AssertionError(f"{reduced['id']}: pinned operation policy differs")
    greedy = nemotron.execute(greedy_graph, {"prefill.token_ids": ids[:, :3]})
    generated = [int(greedy[f"decode{step}.token_ids"][0, 0])
                 for step in range(2)]
    greedy_ids = np.array([[1, 2, 3, *generated]], dtype=np.int32)
    greedy_full = nemotron.execute(full, {"prefill.token_ids": greedy_ids})
    greedy_error = float(np.max(np.abs(
        greedy[greedy_graph.outputs[-1]] - greedy_full[full.outputs[-1]])))
    if not np.allclose(greedy[greedy_graph.outputs[-1]],
                       greedy_full[full.outputs[-1]], rtol=1e-4, atol=1e-5):
        raise AssertionError("greedy decoder differs from its full causal pass")

    changed_inputs = dict(inputs)
    changed_inputs["decode1.token_ids"] = np.array([[6]], dtype=np.int32)
    changed = nemotron.execute(graph, changed_inputs)
    token_delta = float(np.max(np.abs(
        incremental[graph.outputs[-1]] - changed[graph.outputs[-1]])))
    if token_delta <= 0:
        raise AssertionError("last decode token does not affect the output")

    def break_reads(stage_id: str, reads: list[str], output: str) -> float:
        stage = next(stage for stage in graph.stages if stage["id"] == stage_id)
        original = stage["reads"]
        try:
            stage["reads"] = reads
            broken = nemotron.execute(graph, inputs)
        finally:
            stage["reads"] = original
        delta = float(np.max(np.abs(incremental[output] - broken[output])))
        if delta <= 1e-7:
            raise AssertionError(f"{stage_id}: broken state is not detected")
        return delta

    conv_id = "decode0.layer00.depthwise_conv"
    attention_id = "decode0.layer12.attention"
    conv_delta = break_reads(conv_id, ["decode0.layer00.conv_input"], conv_id)
    attention_delta = break_reads(attention_id, [
        "decode0.layer12.q_proj", "decode0.layer12.k_proj",
        "decode0.layer12.v_proj"], attention_id)
    return {
        "schema": "radiance.nemotron-reduced-hybrid-chain.v1",
        "status": "passed",
        "scope": "all 42 pinned blocks in order, reduced widths and synthetic FP32 weights",
        "block_kinds": spec["layer_kinds"],
        "stage_count": len(graph.stages),
        "greedy_stage_count": len(greedy_graph.stages),
        "pinned_ordered_stage_dependencies_equal": True,
        "pinned_attention_and_mamba_policy_equal": True,
        "prefill_tokens": 3,
        "decode_steps": 2,
        "input_token_ids": ids.ravel().tolist(),
        "cached_vs_full_max_abs_logit_error": final_error,
        "generated_token_ids": generated,
        "greedy_cached_vs_full_max_abs_logit_error": greedy_error,
        "changed_decode_token_max_abs_logit_delta": token_delta,
        "missing_convolution_state_max_abs_stage_delta": conv_delta,
        "missing_attention_cache_max_abs_stage_delta": attention_delta,
        "pinned_config_sha256": pinned["source_sha256"],
        "pinned_graph_sha256": sha256(HERE / "nemotron3-nano-4b-graph.json"),
        "graph_builder_sha256": sha256(HERE / "stitch.py"),
        "graph_reference_sha256": sha256(HERE / "nemotron_graph_reference.py"),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "checkpoint_weights": False,
        "device_execution": False,
        "full_network_equivalence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"checked {result['stage_count']} ordered stages")


if __name__ == "__main__":
    main()
