#!/usr/bin/env python3
"""Check Qwen3 MoE top-k routing and expert accumulation against Transformers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock


CONFIG = Path(__file__).parent / "inputs/qwen3-30b-a3b.json"


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def route_and_experts(
        hidden: np.ndarray, weights: dict[str, np.ndarray], top_k: int,
        *, normalize_selected: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = flat @ weights["gate.weight"].T
    probabilities = softmax(logits)
    selected = np.argsort(-probabilities, axis=-1)[:, :top_k]
    scores = np.take_along_axis(probabilities, selected, axis=-1)
    if normalize_selected:
        scores = scores / np.sum(scores, axis=-1, keepdims=True)
    expert_gate_up = weights["experts.gate_up_proj"]
    expert_down = weights["experts.down_proj"]
    result = np.zeros_like(flat)
    for token in range(flat.shape[0]):
        for slot, expert in enumerate(selected[token]):
            gate_up = expert_gate_up[expert] @ flat[token]
            gate, up = np.split(gate_up, 2)
            activated = (gate / (1.0 + np.exp(-gate))) * up
            result[token] += scores[token, slot] * (expert_down[expert] @ activated)
    return result.reshape(hidden.shape), selected, scores


def check() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(29)
    raw = CONFIG.read_bytes()
    pinned = json.loads(raw)
    with (CONFIG.parent / "sources.csv").open(newline="") as stream:
        source = next(row for row in csv.DictReader(stream)
                      if row["case"] == "qwen3-30b-a3b")
    config_sha256 = hashlib.sha256(raw).hexdigest()
    if config_sha256 != source["config_sha256"]:
        raise ValueError("pinned Qwen3 MoE config changed")
    if (pinned["model_type"] != "qwen3_moe" or pinned["num_experts_per_tok"] != 8
            or not pinned["norm_topk_prob"] or pinned["decoder_sparse_step"] != 1
            or pinned["mlp_only_layers"]):
        raise ValueError("pinned MoE routing fields differ")
    small = {"hidden_size": 32, "intermediate_size": 64,
             "moe_intermediate_size": 16, "num_experts": 16,
             "num_experts_per_tok": pinned["num_experts_per_tok"],
             "norm_topk_prob": pinned["norm_topk_prob"],
             "hidden_act": pinned["hidden_act"],
             "num_hidden_layers": 1, "num_attention_heads": 2,
             "num_key_value_heads": 1, "head_dim": 16, "vocab_size": 64}
    config = transformers.Qwen3MoeConfig(**small)
    block = Qwen3MoeSparseMoeBlock(config).eval()
    with torch.no_grad():
        block.gate.weight.copy_(torch.randn_like(block.gate.weight) * 0.15)
        block.experts.gate_up_proj.copy_(
            torch.randn_like(block.experts.gate_up_proj) * 0.15)
        block.experts.down_proj.copy_(
            torch.randn_like(block.experts.down_proj) * 0.15)
    inputs = torch.randn(1, 7, small["hidden_size"])
    with torch.no_grad():
        upstream = block(inputs).detach().cpu().numpy()
        _, upstream_scores, upstream_selected = block.gate(inputs)
    weights = {name: tensor.detach().cpu().numpy()
               for name, tensor in block.state_dict().items()}
    mapped, selected, scores = route_and_experts(inputs.numpy(), weights,
                                                 small["num_experts_per_tok"])
    without_normalization, _, _ = route_and_experts(
        inputs.numpy(), weights, small["num_experts_per_tok"],
        normalize_selected=False)
    selection_equal = bool(np.array_equal(
        selected, upstream_selected.detach().cpu().numpy()))
    score_error = float(np.max(np.abs(
        scores - upstream_scores.detach().cpu().numpy())))
    output_error = float(np.max(np.abs(mapped - upstream)))
    negative_delta = float(np.max(np.abs(without_normalization - upstream)))
    passed = (selection_equal and score_error < 1e-6 and output_error < 1e-5
              and negative_delta > 1e-4 and np.isfinite(mapped).all())
    return {
        "status": "passed" if passed else "failed",
        "scope": "single reduced random-weight sparse expert block, not a decoder, checkpoint, or Radiance execution",
        "pinned_config_sha256": config_sha256,
        "source_model": source["model"],
        "source_revision": source["revision"],
        "small_config": small, "seed": 29,
        "input_shape": list(inputs.shape),
        "selected_experts_shape": list(selected.shape),
        "selected_experts": selected.tolist(),
        "selection_matches_upstream": selection_equal,
        "selected_score_max_abs_error": score_error,
        "output_max_abs_error": output_error,
        "without_topk_renormalization_max_abs_delta": negative_delta,
        "input_sha256": digest(inputs.numpy()),
        "mapped_output_sha256": digest(mapped),
        "upstream_output_sha256": digest(upstream),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = check()
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
