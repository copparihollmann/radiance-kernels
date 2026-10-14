#!/usr/bin/env python3
"""Check last-prefill-token projection against a saved Qwen3 checkpoint build."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
UPSTREAM_REFERENCE = (HERE.parent / "evaluation/llm/"
                      "qwen3-checkpoint-multitoken-reference.json")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def saved_array(memory: mmap.mmap, name: str, count: int) -> np.ndarray:
    marker = f"__global float {name}[{(count + 15) // 16 * 16}] = {{".encode()
    start = memory.find(marker)
    if start < 0:
        raise ValueError(f"generated data lacks {name}")
    start += len(marker)
    end = memory.find(b"};", start)
    if end < 0:
        raise ValueError(f"unterminated generated array {name}")
    values = np.fromstring(memory[start:end].replace(b"f", b"").decode(),
                           sep=",", dtype="<f4")
    if values.size != count:
        raise ValueError(f"wrong element count for {name}: {values.size}")
    return values


def verify(build_dir: Path, image_manifest: Path) -> dict:
    manifest = json.loads((build_dir / "manifest.json").read_text())
    image = json.loads(image_manifest.read_text())
    if (manifest["model"] != "qwen3_0_6b" or manifest["generation"] != "greedy"
            or manifest["prefill_tokens"] < 2 or manifest["layers"] != 28
            or not manifest["full_model_dimensions"]
            or manifest["weight_image_sha256"] != image["image_sha256"]
            or manifest["checkpoint_sha256"] != image["checkpoint_weight_sha256"]
            or manifest["source_config_sha256"] != image["source_config_sha256"]):
        raise ValueError("build is not the expected full Qwen3 checkpoint mapping")
    if any(name not in manifest["verified_tensors"]
           for name in ("prefill.final_norm", "prefill.lm_head")):
        raise ValueError("prefill hidden state or logits were not checked")
    upstream = json.loads(UPSTREAM_REFERENCE.read_text())
    if (upstream["status"] != "passed" or upstream["layers"] != 28
            or upstream["prefill_tokens"] != manifest["prefill_tokens"]
            or upstream["decode_steps"] != manifest["decode_steps"]
            or upstream["input_tokens"]
            != manifest["input_token_ids"]["prefill.token_ids"]
            or upstream["graph_selected_tokens"]
            != manifest["reference_generated_token_ids"]
            or upstream["graph_selected_tokens"]
            != upstream["upstream_selected_tokens"]
            or upstream["checkpoint_config_sha256"]
            != manifest["source_config_sha256"]
            or upstream["checkpoint_weight_sha256"]
            != manifest["checkpoint_sha256"]):
        raise ValueError("saved full-width build differs from upstream control")
    table = next((entry for entry in image["parameters"]
                  if entry["logical_name"] == "embed_tokens"), None)
    if table is None:
        raise ValueError("tied embedding table is absent")
    vocab, width = table["shape"]
    if (table is None or table["storage_dtype"] != "fp16"
            or tuple(table["shape"]) != (vocab, width)
            or table["segment_index"] != 0
            or table["size_bytes"] != vocab * width * 2):
        raise ValueError("unexpected tied FP16 embedding layout")
    data_path = build_dir / "model_data.cpp"
    prefill = manifest["prefill_tokens"]
    with data_path.open("rb") as stream:
        memory = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        hidden = saved_array(memory, "v_gold_prefill_final_norm",
                             prefill * width).reshape(prefill, width).copy()
        logits = saved_array(memory, "v_gold_prefill_lm_head",
                             prefill * vocab).reshape(prefill, vocab)[-1].copy()
        memory.close()
    segment = image_manifest.parent / image["segments"][0]["image_file"]
    if segment.stat().st_size < table["offset_bytes"] + table["size_bytes"]:
        raise ValueError("weight segment does not contain the tied table")
    selected = np.empty(vocab, dtype=np.float32)
    wrong_position = np.empty(vocab, dtype=np.float32)
    packed_hash = hashlib.sha256()
    with segment.open("rb") as stream:
        stream.seek(table["offset_bytes"])
        for start in range(0, vocab, 1024):
            end = min(start + 1024, vocab)
            raw = stream.read((end - start) * width * 2)
            if len(raw) != (end - start) * width * 2:
                raise ValueError("truncated tied table")
            packed_hash.update(raw)
            block = np.frombuffer(raw, dtype="<f2").reshape(end - start, width)
            fp32 = block.astype(np.float32)
            selected[start:end] = fp32 @ hidden[-1]
            wrong_position[start:end] = fp32 @ hidden[0]
    if packed_hash.hexdigest() != table["packed_sha256"]:
        raise ValueError("packed tied table differs from image manifest")
    max_error = float(np.max(np.abs(selected - logits)))
    wrong_error = float(np.max(np.abs(wrong_position - logits)))
    if (not np.allclose(selected, logits, rtol=3e-5, atol=3e-5)
            or int(np.argmax(selected)) != int(np.argmax(logits))
            or wrong_error <= 1e-3):
        raise AssertionError("last-position projection or control failed")
    return {
        "schema": "radiance.qwen3-last-prefill-logit-check.v1",
        "status": "passed", "scope": "full-width checkpoint-backed prefill",
        "optimized_device_executed": False, "rtl_execution": False,
        "performance_measurement": False,
        "layers": manifest["layers"], "prefill_tokens": prefill,
        "vocab_size": vocab, "hidden_size": width,
        "selected_position": prefill - 1,
        "selected_last_row_max_abs_error": max_error,
        "wrong_first_row_max_abs_error": wrong_error,
        "selected_top1": int(np.argmax(selected)),
        "saved_full_logit_top1": int(np.argmax(logits)),
        "saved_reference_output_sha256": manifest["reference_output_sha256"],
        "upstream_reference_sha256": sha(UPSTREAM_REFERENCE),
        "upstream_selected_tokens": upstream["upstream_selected_tokens"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "weight_image_sha256": image["image_sha256"],
        "tied_table_packed_sha256": table["packed_sha256"],
        "build_manifest_sha256": sha(build_dir / "manifest.json"),
        "weight_image_manifest_sha256": sha(image_manifest),
        "generated_model_data_sha256": sha(data_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--weight-image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.build_dir.resolve(), args.weight_image.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"],
                      "max_abs_error": report["selected_last_row_max_abs_error"],
                      "top1": report["selected_top1"]}))


if __name__ == "__main__":
    main()
