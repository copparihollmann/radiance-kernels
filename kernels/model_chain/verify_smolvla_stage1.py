#!/usr/bin/env python3
"""Compare the generated first vision stage with the pinned upstream module."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from split_decoder_weights import verify_image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernels/evaluation/llm"))
from stitch import model_specs  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(checkpoint_dir: Path, input_image: Path, stage_dump: Path) -> dict:
    import torch
    from safetensors import safe_open

    spec = model_specs()["smolvla_base"]
    checkpoint = checkpoint_dir / "model.safetensors"
    if (sha256(checkpoint_dir / "config.json") != spec["source_sha256"] or
            sha256(checkpoint) != spec["checkpoint_weight_sha256"]):
        raise ValueError("SmolVLA checkpoint differs from pinned revision")
    image_manifest, image_paths = verify_image(input_image)
    if (image_manifest["model"] != "smolvla_base" or
            image_manifest["checkpoint_weight_sha256"] !=
            spec["checkpoint_weight_sha256"]):
        raise ValueError("SmolVLA native input differs from pinned fixture")
    camera = next(item for item in image_manifest["parameters"]
                  if item["logical_name"] == "camera0.image")
    source = np.fromfile(image_paths[0], dtype="<f4", count=3 * 512 * 512,
                         offset=camera["offset_bytes"]).reshape(1, 3, 512, 512)
    actual = np.fromfile(stage_dump, dtype="<f4")
    if actual.size != 1024 * 768:
        raise ValueError("first native vision stage has the wrong output size")
    actual = actual.reshape(1, 1024, 768)
    prefix = "model.vlm_with_expert.vlm.model.vision_model.embeddings."
    with safe_open(str(checkpoint), framework="pt", device="cpu") as file:
        weight = file.get_tensor(prefix + "patch_embedding.weight").float()
        bias = file.get_tensor(prefix + "patch_embedding.bias").float()
        position = file.get_tensor(prefix + "position_embedding.weight").float()
    torch.set_num_threads(16)
    with torch.no_grad():
        expected = (torch.nn.functional.conv2d(
            torch.from_numpy(source), weight, bias, stride=16)
            .flatten(2).transpose(1, 2) + position).numpy()
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    passed = bool(np.allclose(actual, expected, rtol=1e-3, atol=1e-3))
    return {
        "model": "smolvla_base", "stage": "v.camera0.patch_embed",
        "checkpoint_weight_sha256": spec["checkpoint_weight_sha256"],
        "input_image_sha256": image_manifest["image_sha256"],
        "native_output_sha256": sha256(stage_dump),
        "reference_output_sha256": hashlib.sha256(
            np.ascontiguousarray(expected).tobytes()).hexdigest(),
        "output_shape": list(actual.shape),
        "max_abs_error": float(np.max(np.abs(delta))),
        "rms_error": float(np.sqrt(np.mean(delta * delta))),
        "rtol": 1e-3, "atol": 1e-3, "passed": passed,
        "device_execution": False, "rtl_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--stage-dump", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args.checkpoint_dir, args.input_image, args.stage_dump)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"SmolVLA first vision stage: {'passed' if result['passed'] else 'failed'}, "
          f"maximum error {result['max_abs_error']:.3g}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
