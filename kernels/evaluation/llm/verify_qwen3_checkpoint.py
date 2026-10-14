#!/usr/bin/env python3
"""Compare the pinned full Qwen3 checkpoint against upstream Transformers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from checkpoint import SafeTensorWeights
from reference import execute
from stitch import build, model_specs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(actual: np.ndarray, upstream: np.ndarray) -> dict:
    delta = actual.astype(np.float64) - upstream.astype(np.float64)
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(np.max(np.abs(delta))),
        "rms_error": float(np.sqrt(np.mean(delta * delta))),
        "allclose_atol_1e-4_rtol_1e-4": bool(np.allclose(
            actual, upstream, atol=1e-4, rtol=1e-4)),
        "top1_equal": bool(np.argmax(actual[0, -1]) == np.argmax(upstream[0, -1])),
        "graph_sha256": hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest(),
        "upstream_sha256": hashlib.sha256(
            np.ascontiguousarray(upstream).tobytes()).hexdigest(),
    }


def check(checkpoint_dir: Path, input_tokens: list[int], decode_steps: int) -> dict:
    torch.set_num_threads(1)
    spec = model_specs()["qwen3_0_6b"]
    config_sha = sha256_file(checkpoint_dir / "config.json")
    weight_sha = sha256_file(checkpoint_dir / "model.safetensors")
    if config_sha != spec["source_sha256"] or weight_sha != spec[
            "checkpoint_weight_sha256"]:
        raise ValueError("Qwen3 checkpoint differs from the pinned config or weights")
    if (not input_tokens or decode_steps <= 0 or
            len(input_tokens) * spec["vocab_size"] > 1_000_000):
        raise ValueError("use one to six prefill tokens and positive decode steps")
    if any(not 0 <= token < spec["vocab_size"] for token in input_tokens):
        raise ValueError("input token is outside the vocabulary")
    weights = SafeTensorWeights(checkpoint_dir / "model.safetensors", "qwen3")
    tied = weights.verify_tied_lm_head()
    graph = build("qwen3_0_6b", prefill=len(input_tokens),
                  decode_steps=decode_steps,
                  generation="greedy")
    mapped = execute(graph, {"prefill.token_ids": np.array(
        [input_tokens], dtype=np.int32)}, weights=weights)

    config = transformers.Qwen3Config.from_pretrained(
        checkpoint_dir, local_files_only=True)
    config._attn_implementation = "eager"
    model = transformers.Qwen3ForCausalLM.from_pretrained(
        checkpoint_dir, config=config, dtype=torch.float32,
        low_cpu_mem_usage=True, local_files_only=True).eval()
    with torch.inference_mode():
        prefix = model(input_ids=torch.tensor([input_tokens], dtype=torch.long),
                       use_cache=True)
        outputs = [prefix.logits.detach().cpu().numpy()]
        selected = []
        previous = prefix
        for _ in range(decode_steps):
            token = int(torch.argmax(previous.logits[0, -1]))
            selected.append(token)
            previous = model(input_ids=torch.tensor([[token]], dtype=torch.long),
                             past_key_values=previous.past_key_values, use_cache=True)
            outputs.append(previous.logits.detach().cpu().numpy())
        full = model(input_ids=torch.tensor([input_tokens + selected],
                                                 dtype=torch.long), use_cache=False)
    upstream_full_last = full.logits[:, -1:, :].detach().cpu().numpy()
    comparisons = {
        phase: compare(mapped[name], upstream)
        for phase, name, upstream in zip(
            ["prefill", *(f"decode{step}" for step in range(decode_steps))],
            graph.outputs, outputs)}
    comparisons["upstream_cached_vs_full"] = compare(outputs[-1],
                                                       upstream_full_last)
    graph_tokens = [int(mapped[f"decode{step}.token_ids"][0, 0])
                    for step in range(decode_steps)]
    passed = (all(row["allclose_atol_1e-4_rtol_1e-4"] and row["top1_equal"]
                  for row in comparisons.values()) and graph_tokens == selected)
    token_record = ({"input_token": input_tokens[0],
                     "graph_selected_token": graph_tokens[0],
                     "upstream_selected_token": selected[0]}
                    if len(input_tokens) == decode_steps == 1 else
                    {"input_tokens": input_tokens,
                     "graph_selected_tokens": graph_tokens,
                     "upstream_selected_tokens": selected})
    return {
        "status": "passed" if passed else "failed",
        "scope": "full_checkpoint_fp32_software_reference",
        "model": spec["name"], "layers": spec["num_hidden_layers"],
        "prefill_tokens": len(input_tokens), "decode_steps": decode_steps,
        "generation": "greedy", **token_record,
        "checkpoint_config_sha256": config_sha,
        "checkpoint_weight_sha256": weight_sha,
        "saved_tied_head_check": tied,
        "upstream_storage_tied": (model.lm_head.weight.data_ptr() ==
                                  model.model.embed_tokens.weight.data_ptr()),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "graph_stages": len(graph.stages),
        "comparisons": comparisons,
        "radiance_execution": False, "hardware_timing": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--input-token", type=int, default=5)
    parser.add_argument("--prefill-ids", nargs="+", type=int)
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = check(args.checkpoint_dir, args.prefill_ids or [args.input_token],
                   args.decode_steps)
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
