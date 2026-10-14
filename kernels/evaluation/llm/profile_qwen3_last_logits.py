#!/usr/bin/env python3
"""Compare full versus last-token prefill logits for greedy Qwen3 graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reference import execute, parameter, reduced_spec
from stitch import build, model_specs

HERE = Path(__file__).resolve().parent
MODEL = "qwen3_0_6b"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(batch: int, prefill: int, decode_steps: int) -> dict:
    kwargs = {"batch": batch, "prefill": prefill,
              "decode_steps": decode_steps, "generation": "greedy"}
    full = build(MODEL, **kwargs)
    last = build(MODEL, prefill_logits="last", **kwargs)
    names_full = [s["id"] for s in full.stages]
    names_last = [s["id"] for s in last.stages]
    if (names_last != names_full[:names_full.index("prefill.lm_head")]
            + ["prefill.last_hidden"]
            + names_full[names_full.index("prefill.lm_head"):]):
        raise AssertionError("last-token graph changes a non-output stage")
    if full.execution_schedule["decode_loop"] != last.execution_schedule["decode_loop"]:
        raise AssertionError("last-token graph changes decode or KV dependencies")
    vocab = model_specs()[MODEL]["vocab_size"]
    if (full.tensors["prefill.lm_head"]["shape"] != [batch, prefill, vocab]
            or last.tensors["prefill.lm_head"]["shape"] != [batch, 1, vocab]):
        raise AssertionError("unexpected prefill logit shape")
    macs = lambda graph: next(s["attrs"]["macs"] for s in graph.stages
                              if s["id"] == "prefill.lm_head")
    full_macs, last_macs = macs(full), macs(last)
    if full_macs != prefill * last_macs:
        raise AssertionError("LM head MAC accounting changed unexpectedly")
    return {
        "batch": batch, "prefill_tokens": prefill,
        "decode_steps": decode_steps,
        "full_graph_stages": len(full.stages),
        "last_graph_stages": len(last.stages),
        "prefill_logits_full_shape": [batch, prefill, vocab],
        "prefill_logits_last_shape": [batch, 1, vocab],
        "prefill_logits_full_fp32_bytes": batch * prefill * vocab * 4,
        "prefill_logits_last_fp32_bytes": batch * vocab * 4,
        "prefill_logits_tensor_fp32_bytes_saved": batch * (prefill - 1) * vocab * 4,
        "prefill_lm_head_full_macs": full_macs,
        "prefill_lm_head_last_macs": last_macs,
        "decode_and_kv_dependencies_unchanged": True,
    }


def check_last_row_projection() -> dict:
    spec = reduced_spec(model_specs()[MODEL])
    graph = build(MODEL, prefill=3, decode_steps=1, generation="greedy",
                  specs={MODEL: spec})
    values = execute(graph, {"prefill.token_ids": np.array([[1, 2, 3]],
                                                            dtype=np.int32)})
    h, vocab = spec["hidden_size"], spec["vocab_size"]
    weight = parameter("embed_tokens", (vocab, h), 0.05)
    selected = values["prefill.final_norm"][:, -1:, :] @ weight.T
    expected = values["prefill.lm_head"][:, -1:, :]
    error = float(np.max(np.abs(selected - expected)))
    if not np.allclose(selected, expected, rtol=1e-6, atol=1e-7):
        raise AssertionError("last-row LM projection differs from full logits")
    return {
        "scope": "reduced synthetic Qwen3 NumPy projection identity",
        "input_ids": [1, 2, 3], "max_abs_error": error,
        "prefill_argmax_equal": bool(np.array_equal(
            np.argmax(selected, axis=-1), np.argmax(expected, axis=-1))),
        "full_greedy_decode_token": int(values["decode0.token_ids"][0, 0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "radiance.qwen3-prefill-logit-shape-profile.v1",
        "model": MODEL, "source_config_sha256": sha(HERE / "inputs/qwen3-0.6b.json"),
        "graph_source_sha256": sha(HERE / "stitch.py"),
        "scope": "static greedy-generation graph; no device execution",
        "cases": [profile(batch, prefill, 2)
                  for batch, prefill in ((1, 3), (1, 128), (8, 128), (1, 1024))],
        "last_row_projection_check": check_last_row_projection(),
        "optimized_graph_numerically_executed": False,
        "compiled_device_program": False,
        "hardware_performance_measurement": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
