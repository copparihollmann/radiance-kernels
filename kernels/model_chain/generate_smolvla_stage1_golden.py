#!/usr/bin/env python3
"""Generate the pinned PyTorch golden for SmolVLA's first vision stage."""

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
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(checkpoint_dir: Path, input_image: Path,
             out_dir: Path, gpu_base: int) -> dict:
    from safetensors import safe_open
    import torch

    spec = model_specs()["smolvla_base"]
    checkpoint = checkpoint_dir / "model.safetensors"
    if (sha256(checkpoint_dir / "config.json") != spec["source_sha256"] or
            sha256(checkpoint) != spec["checkpoint_weight_sha256"]):
        raise ValueError("SmolVLA checkpoint differs from pinned revision")
    image, paths = verify_image(input_image)
    reference_path = HERE / "evaluation/smolvla-stage1-upstream-reference.json"
    reference = json.loads(reference_path.read_text())
    if (image["model"] != "smolvla_base" or
            image["checkpoint_weight_sha256"] !=
            spec["checkpoint_weight_sha256"] or
            image["image_sha256"] != reference["input_image_sha256"]):
        raise ValueError("SmolVLA input differs from recorded upstream control")
    camera = next(item for item in image["parameters"]
                  if item["logical_name"] == "camera0.image")
    segment = image.get("segments", [{"gpu_base_address": image["gpu_base_address"]}])[
        camera.get("segment_index", 0)]
    offset = camera["gpu_address"] - segment["gpu_base_address"]
    source = np.fromfile(paths[camera.get("segment_index", 0)], dtype="<f4",
                         count=3 * 512 * 512, offset=offset).reshape(1, 3, 512, 512)
    prefix = "model.vlm_with_expert.vlm.model.vision_model.embeddings."
    with safe_open(str(checkpoint), framework="pt", device="cpu") as weights:
        patch = weights.get_tensor(prefix + "patch_embedding.weight").float()
        bias = weights.get_tensor(prefix + "patch_embedding.bias").float()
        position = weights.get_tensor(prefix + "position_embedding.weight").float()
    torch.set_num_threads(16)
    with torch.no_grad():
        output = (torch.nn.functional.conv2d(
            torch.from_numpy(source), patch, bias, stride=16)
            .flatten(2).transpose(1, 2) + position).numpy()
    if list(output.shape) != reference["output_shape"]:
        raise ValueError("upstream stage output has an unexpected shape")
    payload = np.ascontiguousarray(output.astype("<f4", copy=False)).tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != reference["reference_output_sha256"]:
        raise ValueError("upstream stage output differs from recorded reference")
    if gpu_base < 0 or gpu_base % 64 or gpu_base + len(payload) > (1 << 32):
        raise ValueError("stage golden GPU address must be aligned and in range")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage1.bin").write_bytes(payload)
    manifest = {
        "model": "smolvla_base", "dtype": "fp32",
        "checkpoint_weight_sha256": spec["checkpoint_weight_sha256"],
        "gpu_base_address": gpu_base, "image_size_bytes": len(payload),
        "image_file": "stage1.bin", "image_sha256": digest,
        "parameters": [], "reference_stage": reference["stage"],
        "source_reference_sha256": sha256(reference_path),
        "source_input_sha256": image["image_sha256"],
    }
    path = out_dir / "stage-golden-image.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"manifest": str(path), "image_sha256": digest,
            "image_size_bytes": len(payload)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu-base", type=lambda value: int(value, 0),
                        default=0x02000000)
    args = parser.parse_args()
    print(json.dumps(generate(args.checkpoint_dir.resolve(),
                              args.input_image.resolve(), args.out_dir.resolve(),
                              args.gpu_base), indent=2))


if __name__ == "__main__":
    main()
