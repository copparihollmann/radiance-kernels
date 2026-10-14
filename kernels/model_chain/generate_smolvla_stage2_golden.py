#!/usr/bin/env python3
"""Generate a pinned PyTorch golden for SmolVLA's first vision layer norm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from split_decoder_weights import image_segments, verify_image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernels/evaluation/llm"))
from stitch import build, model_specs  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(stage1_image: Path, weight_image: Path,
             out_dir: Path, gpu_base: int) -> dict:
    import torch

    graph = build("smolvla_base")
    first, second = graph.stages[:2]
    if (first["op"] != "patch_embed" or second["op"] != "layernorm"
            or second["reads"] != [first["writes"]]
            or second["shape"] != [1, 1024, 768]):
        raise ValueError("pinned SmolVLA vision prefix changed")
    spec = model_specs()["smolvla_base"]
    reference = json.loads((HERE / "evaluation/smolvla-stage1-upstream-reference.json").read_text())
    source, source_paths = verify_image(stage1_image)
    if (source["image_sha256"] != reference["reference_output_sha256"]
            or source["reference_stage"] != reference["stage"]
            or reference["stage"].removeprefix("v.") != first["id"]
            or source["checkpoint_weight_sha256"] != spec["checkpoint_weight_sha256"]
            or len(source_paths) != 1 or source["image_size_bytes"] != 786432 * 4):
        raise ValueError("first-stage upstream golden differs from pinned source")
    weights, weight_paths = verify_image(weight_image)
    if (weights["model"] != "smolvla_base"
            or weights["dtype"] != "fp32"
            or weights["checkpoint_weight_sha256"] != spec["checkpoint_weight_sha256"]):
        raise ValueError("second-stage weights differ from pinned checkpoint")
    params = {item["logical_name"]: item for item in weights["parameters"]}
    segments = image_segments(weights)

    def parameter(name: str) -> np.ndarray:
        item = params[name]
        index = item["segment_index"]
        offset = item["gpu_address"] - segments[index]["gpu_base_address"]
        value = np.fromfile(weight_paths[index], dtype="<f4", count=768,
                            offset=offset)
        if value.size != 768 or item["shape"] != [768]:
            raise ValueError(f"invalid layer-norm parameter: {name}")
        return value

    gamma = parameter(next(name for name in second["attrs"]["parameters"]
                           if name.endswith("layer_norm1.weight")))
    bias = parameter(next(name for name in second["attrs"]["parameters"]
                          if name.endswith("layer_norm1.bias")))
    stage1 = np.fromfile(source_paths[0], dtype="<f4").reshape(1, 1024, 768)
    torch.set_num_threads(1)
    with torch.no_grad():
        output = torch.nn.functional.layer_norm(
            torch.from_numpy(stage1), (768,), torch.from_numpy(gamma),
            torch.from_numpy(bias), eps=second["attrs"]["epsilon"])
    payload = np.ascontiguousarray(output.numpy().astype("<f4", copy=False)).tobytes()
    if gpu_base < 0 or gpu_base % 64 or gpu_base + len(payload) > 1 << 32:
        raise ValueError("stage golden GPU address must be aligned and in range")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage2.bin").write_bytes(payload)
    manifest = {
        "model": "smolvla_base", "dtype": "fp32",
        "checkpoint_weight_sha256": spec["checkpoint_weight_sha256"],
        "gpu_base_address": gpu_base, "image_size_bytes": len(payload),
        "image_file": "stage2.bin", "image_sha256": hashlib.sha256(payload).hexdigest(),
        "parameters": [], "reference_stage": second["id"],
        "source_stage1_image_sha256": source["image_sha256"],
        "source_weight_image_sha256": weights["image_sha256"],
        "source_spec_sha256": sha(ROOT / "kernels/evaluation/llm/pr1-models.json"),
        "epsilon": second["attrs"]["epsilon"],
        "reference_operation": "torch.nn.functional.layer_norm",
    }
    path = out_dir / "stage-golden-image.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"manifest": str(path), "image_sha256": manifest["image_sha256"],
            "image_size_bytes": len(payload)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-image", type=Path, required=True)
    parser.add_argument("--weight-image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu-base", type=lambda value: int(value, 0),
                        default=0x02000000)
    args = parser.parse_args()
    print(json.dumps(generate(args.stage1_image, args.weight_image,
                              args.out_dir, args.gpu_base), indent=2))


if __name__ == "__main__":
    main()
