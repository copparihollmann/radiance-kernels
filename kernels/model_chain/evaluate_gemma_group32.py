#!/usr/bin/env python3
"""Check group-32 INT4 Gemma greedy tokens against the pinned checkpoint.

This is a numerical feasibility check, not a Radiance executable. It reads the
checkpoint in place and keeps no weight image on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "evaluation/llm"))

from checkpoint import (SafeTensorWeights, TiedINT8GroupINT4GemmaWeights)  # noqa: E402
from export_decoder_weights import sha256_file  # noqa: E402
import reference  # noqa: E402
import stitch  # noqa: E402


def evaluate(checkpoint_dir: Path) -> dict:
    spec = stitch.model_specs()["gemma_2_2b_it"]
    if sha256_file(checkpoint_dir / "config.json") != spec["source_sha256"]:
        raise ValueError("checkpoint config differs from pinned Gemma")
    files = {name: sha256_file(checkpoint_dir / name)
             for name in spec["checkpoint_files_sha256"]}
    if files != spec["checkpoint_files_sha256"]:
        raise ValueError("checkpoint shards differ from pinned Gemma")
    fixture = json.loads((HERE / "evaluation/gemma-token-greedy-upstream.json").read_text())
    graph = stitch.build("gemma_2_2b_it", prefill=2, decode_steps=2,
                         generation="greedy")
    provider = TiedINT8GroupINT4GemmaWeights(
        SafeTensorWeights(checkpoint_dir, "gemma2"))
    values = reference.execute(
        graph, {"prefill.token_ids": np.asarray([[2, 100]], dtype=np.int32)},
        weights=provider)
    generated = [int(values[f"decode{step}.token_ids"][0, 0]) for step in range(2)]
    upstream = fixture["upstream_generated_token_ids"]
    return {
        "model": "gemma_2_2b_it", "layers": 26, "stages": len(graph.stages),
        "weight_format": "tied_INT8_embedding_attention_group32_INT4_FFN",
        "input_token_ids": [2, 100], "generated_token_ids": generated,
        "upstream_generated_token_ids": upstream,
        "tokens_match_upstream": generated == upstream,
        "checkpoint_config_sha256": spec["source_sha256"],
        "checkpoint_files_sha256": files,
        "radiance_device_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint_dir), indent=2))


if __name__ == "__main__":
    main()
