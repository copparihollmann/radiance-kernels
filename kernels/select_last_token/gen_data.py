#!/usr/bin/env python3
"""Generate a final-token selection fixture from NumPy or saved Qwen3 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_chain"))
from verify_qwen3_last_logit import UPSTREAM_REFERENCE, saved_array, sha

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib/golden"))
import golden as G


def generate(out: Path, source: str, build_dir: Path | None) -> dict:
    if source == "synthetic":
        if build_dir is not None:
            raise ValueError("synthetic fixture does not use a saved build")
        rng = np.random.default_rng(37)
        data = rng.normal(size=(2, 5, 33)).astype(np.float32)
        provenance = {"seed": 37}
    elif source == "qwen3":
        if build_dir is None:
            raise ValueError("Qwen3 fixture requires --build-dir")
        manifest_path = build_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        upstream = json.loads(UPSTREAM_REFERENCE.read_text())
        if (manifest["model"] != "qwen3_0_6b" or manifest["layers"] != 28
                or manifest["prefill_tokens"] != 3
                or not manifest["full_model_dimensions"]
                or not manifest["checkpoint_weights"]
                or upstream["status"] != "passed"
                or upstream["input_tokens"]
                != manifest["input_token_ids"]["prefill.token_ids"]
                or upstream["checkpoint_weight_sha256"]
                != manifest["checkpoint_sha256"]
                or upstream["checkpoint_config_sha256"]
                != manifest["source_config_sha256"]
                or "prefill.final_norm" not in manifest["verified_tensors"]):
            raise ValueError("saved build lacks the full Qwen3 prefill state")
        data_path = build_dir / "model_data.cpp"
        with data_path.open("rb") as stream:
            memory = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            data = saved_array(memory, "v_gold_prefill_final_norm", 3 * 1024)
            data = data.reshape(1, 3, 1024).copy()
            memory.close()
        provenance = {
            "build_manifest_sha256": sha(manifest_path),
            "generated_model_data_sha256": sha(data_path),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "source_config_sha256": manifest["source_config_sha256"],
            "upstream_reference_sha256": sha(UPSTREAM_REFERENCE),
        }
    else:
        raise ValueError(f"unsupported fixture source: {source}")
    batch, tokens, width = data.shape
    gold = np.ascontiguousarray(data[:, -1, :])
    wrong_delta = float(np.max(np.abs(data[:, 0, :] - gold)))
    if wrong_delta < 1e-3:
        raise ValueError("first-token negative control is too weak")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as stream:
        for name, value in (("BATCH", batch), ("TOKENS", tokens), ("WIDTH", width)):
            stream.write(f"#define {name} {value}\n")
        for name, array in (("input_raw", data.ravel()), ("gold_raw", gold.ravel())):
            G.emit_f32(stream, name, array, dims=f"[{array.size}]",
                       trailing_comma=True)
    digest = lambda array: hashlib.sha256(array.tobytes()).hexdigest()
    return {
        "schema": "radiance.select-last-token-fixture.v1",
        "source": source, "provenance": provenance,
        "batch": batch, "tokens": tokens, "width": width,
        "input_sha256": digest(data), "gold_sha256": digest(gold),
        "wrong_first_token_max_abs_delta": wrong_delta,
        "generated_header_sha256": sha(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source", choices=("synthetic", "qwen3"),
                        default="synthetic")
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    report = generate(args.out, args.source, args.build_dir)
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
