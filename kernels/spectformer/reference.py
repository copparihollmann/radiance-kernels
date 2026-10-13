#!/usr/bin/env python3
"""Run the pinned upstream SpectFormer on a normalized NCHW input tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prepare import UPSTREAM_SOURCE_SHA256, load_upstream, sha256


def load_input(path: Path, image_size: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        pixels = np.load(path, allow_pickle=False)
    else:
        from PIL import Image
        from torchvision import transforms
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        with Image.open(path) as image:
            pixels = transform(image.convert("RGB")).unsqueeze(0).numpy()
    if pixels.shape != (1, 3, image_size, image_size):
        raise ValueError("expected NCHW image [1,3,H,W]")
    if pixels.dtype != np.float32 or not np.isfinite(pixels).all():
        raise ValueError("image must contain finite FP32 values")
    return np.ascontiguousarray(pixels, dtype="<f4")


def make_model(manifest: dict):
    import torch

    source = Path(manifest["upstream_source"])
    if sha256(source) != UPSTREAM_SOURCE_SHA256:
        raise ValueError("upstream source hash differs from the pinned revision")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(manifest["seed"])
    upstream = load_upstream(source)
    model = upstream.SpectFormer(
        img_size=manifest["image_size"], patch_size=16,
        embed_dim=manifest["embed_dim"], depth=manifest["depth"],
        num_classes=manifest["num_classes"], drop_rate=0.0,
        drop_path_rate=0.0)
    checkpoint_path = manifest.get("checkpoint")
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
            raise ValueError("checkpoint differs from export")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.eval()
    return model


def logits(manifest: dict, pixels: np.ndarray) -> np.ndarray:
    import torch

    model = make_model(manifest)
    with torch.no_grad():
        result = model(torch.from_numpy(pixels.copy())).detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("upstream produced nonfinite logits")
    return np.ascontiguousarray(result, dtype="<f4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.image_manifest.read_text())
    result = logits(manifest, load_input(args.input, manifest["image_size"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, result)
    print(args.out)


if __name__ == "__main__":
    main()
