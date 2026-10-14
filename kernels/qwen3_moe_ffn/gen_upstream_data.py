#!/usr/bin/env python3
"""Extract a reduced Qwen3 MoE FFN fixture from the Transformers CPU block."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import transformers
from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/llm"))
from verify_qwen3_moe_router import CONFIG, digest, route_and_experts

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib/golden"))
import golden as G

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parents[0] / "evaluation/llm/qwen3-moe-router-reference.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(out: Path) -> dict:
    pinned = json.loads(CONFIG.read_text())
    saved = json.loads(REFERENCE.read_text())
    with (CONFIG.parent / "sources.csv").open(newline="") as stream:
        source = next(row for row in csv.DictReader(stream)
                      if row["case"] == "qwen3-30b-a3b")
    if (sha(CONFIG) != source["config_sha256"]
            or sha(CONFIG) != saved["pinned_config_sha256"]
            or saved["status"] != "passed"
            or saved["small_config"]["num_experts"] != 16
            or saved["small_config"]["num_experts_per_tok"] != 8
            or not pinned["norm_topk_prob"]):
        raise ValueError("pinned or reduced MoE reference changed")
    torch.set_num_threads(1)
    torch.manual_seed(saved["seed"])
    config = transformers.Qwen3MoeConfig(**saved["small_config"])
    block = Qwen3MoeSparseMoeBlock(config).eval()
    with torch.no_grad():
        block.gate.weight.copy_(torch.randn_like(block.gate.weight) * 0.15)
        block.experts.gate_up_proj.copy_(
            torch.randn_like(block.experts.gate_up_proj) * 0.15)
        block.experts.down_proj.copy_(
            torch.randn_like(block.experts.down_proj) * 0.15)
    inputs = torch.randn(1, 7, saved["small_config"]["hidden_size"])
    with torch.no_grad():
        upstream_output = block(inputs).detach().cpu().numpy()
        _, upstream_scores, upstream_selected = block.gate(inputs)
    weights = {name: tensor.detach().cpu().numpy()
               for name, tensor in block.state_dict().items()}
    mapped, selected, scores = route_and_experts(
        inputs.numpy(), weights, saved["small_config"]["num_experts_per_tok"])
    selected = np.ascontiguousarray(selected, dtype=np.uint32)
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    output = np.ascontiguousarray(mapped[0], dtype=np.float32)
    if (digest(inputs.numpy()) != saved["input_sha256"]
            or digest(mapped) != saved["mapped_output_sha256"]
            or digest(upstream_output) != saved["upstream_output_sha256"]
            or selected.tolist() != saved["selected_experts"]
            or not np.array_equal(selected, upstream_selected.numpy())
            or not np.allclose(scores, upstream_scores.numpy(),
                               rtol=3e-5, atol=3e-6)
            or not np.allclose(output, upstream_output[0],
                               rtol=3e-5, atol=3e-6)):
        raise AssertionError("recreated reduced fixture differs from upstream check")
    arrays = {
        "input_raw": np.ascontiguousarray(inputs.numpy()[0], dtype=np.float32),
        "router_weight_raw": np.ascontiguousarray(weights["gate.weight"],
                                                    dtype=np.float32),
        "expert_gate_up_raw": np.ascontiguousarray(
            weights["experts.gate_up_proj"], dtype=np.float32),
        "expert_down_raw": np.ascontiguousarray(
            weights["experts.down_proj"], dtype=np.float32),
        "gold_scores_raw": scores,
        "gold_output_raw": output,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as stream:
        for name, value in (("TOKENS", 7), ("HIDDEN", 32), ("EXPERTS", 16),
                            ("SELECTED", 8), ("INTERMEDIATE", 16)):
            stream.write(f"#define {name} {value}\n")
        for name, array in arrays.items():
            G.emit_f32(stream, name, array.ravel(), dims=f"[{array.size}]",
                       trailing_comma=True)
        G.emit_u32(stream, "gold_indices_raw", selected.ravel(),
                   dims=f"[{selected.size}]")
    return {
        "schema": "radiance.qwen3-moe-ffn-fixture.v1",
        "source": "transformers_reduced",
        "scope": "single reduced random-weight expert block; no checkpoint",
        "tokens": 7, "hidden": 32, "experts": 16,
        "selected": 8, "intermediate": 16,
        "pinned_config_sha256": sha(CONFIG),
        "reference_receipt_sha256": sha(REFERENCE),
        "modeling_source_sha256": sha(Path(modeling_qwen3_moe.__file__)),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "array_sha256": {name: digest(array) for name, array in arrays.items()},
        "gold_indices_sha256": digest(selected),
        "generated_header_sha256": sha(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    report = generate(args.out)
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
