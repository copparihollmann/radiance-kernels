#!/usr/bin/env python3
"""Export an upstream vanilla SpectFormer and one input/golden to FP32 GMEM.

The export accepts the original source file rather than reimplementing the
network in Python. A random-initialized export is a structural correctness
fixture; pass --checkpoint for a pretrained-weight inference experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np


BASE_ADDRESS = 0x30000000
ALIGNMENT = 64
UPSTREAM_REVISION = "b6cf487a0ea71eeb252d5ba0fc7410b3f5dbd256"
UPSTREAM_SOURCE_SHA256 = "611aee409308e898d23fcbb1c75de3a0c5fafc205df74c6250e49bfe2ac53250"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_upstream(path: Path):
    # The pinned source imports an unused NumPy 1.x private module. NumPy 2.x
    # removed that import path; supply only the public pad alias it requests.
    if "numpy.lib.arraypad" not in sys.modules:
        try:
            import numpy.lib.arraypad  # noqa: F401
        except ModuleNotFoundError:
            compatibility = types.ModuleType("numpy.lib.arraypad")
            compatibility.pad = np.pad
            sys.modules["numpy.lib.arraypad"] = compatibility
    spec = importlib.util.spec_from_file_location("spectformer_upstream", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import upstream source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export(args: argparse.Namespace) -> Path:
    import torch

    if sha256(args.upstream_source) != UPSTREAM_SOURCE_SHA256:
        raise ValueError("upstream source differs from the pinned SpectFormers revision")
    if args.embed_dim % 6:
        raise ValueError("upstream vanilla attention hardcodes six heads")
    if args.depth < 5 or args.image_size % 16:
        raise ValueError("need at least one attention block and a 16-pixel patch grid")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(args.seed)
    upstream = load_upstream(args.upstream_source)
    model = upstream.SpectFormer(img_size=args.image_size, patch_size=16,
                                 embed_dim=args.embed_dim, depth=args.depth,
                                 num_classes=args.num_classes,
                                 drop_rate=0.0, drop_path_rate=0.0)
    checkpoint_digest = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state, strict=True)
        checkpoint_digest = sha256(args.checkpoint)
    model.eval()
    if args.input_npy:
        pixels = np.load(args.input_npy, allow_pickle=False)
        if pixels.shape != (1, 3, args.image_size, args.image_size):
            raise ValueError("input NPY must have shape [1,3,image_size,image_size]")
        if pixels.dtype != np.float32 or not np.isfinite(pixels).all():
            raise ValueError("input NPY must contain finite float32 values")
        image = torch.from_numpy(np.ascontiguousarray(pixels))
    else:
        image = torch.linspace(-1.0, 1.0, 3 * args.image_size * args.image_size,
                               dtype=torch.float32).reshape(1, 3, args.image_size,
                                                            args.image_size)
    with torch.no_grad():
        golden = model(image).detach().cpu().numpy().astype("<f4")
    if not np.isfinite(golden).all():
        raise ValueError("upstream source produced nonfinite logits")
    linear_names = {name for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear)}
    path = args.out / "image.bin"
    entries: dict[str, dict] = {}
    offset = 0

    def add(stream, name: str, values: np.ndarray) -> None:
        nonlocal offset
        padded = (-offset) % ALIGNMENT
        if padded:
            stream.write(bytes(padded))
            offset += padded
        values = np.ascontiguousarray(values, dtype="<f4")
        payload = values.tobytes()
        stream.write(payload)
        entries[name] = {"address": BASE_ADDRESS + offset,
                         "offset_bytes": offset, "size_bytes": len(payload),
                         "shape": list(values.shape),
                         "sha256": hashlib.sha256(payload).hexdigest()}
        offset += len(payload)

    with path.open("wb") as stream:
        for name, tensor in model.state_dict().items():
            values = tensor.detach().cpu().numpy()
            if name.endswith(".weight") and name[:-len(".weight")] in linear_names:
                values = values.T  # Radiance linear consumes row-major W[K,N].
            add(stream, name, values)
        add(stream, "input.image", image.numpy())
        add(stream, "output.logits", golden)
        padded = (-offset) % ALIGNMENT
        if padded:
            stream.write(bytes(padded))
            offset += padded
    if BASE_ADDRESS + offset >= 0x7EE00000:
        raise ValueError("weight image overlaps Muon warp stacks")
    manifest = {
        "format": "spectformer.vanilla.fp32.v1", "upstream_source": str(args.upstream_source),
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_source_sha256": sha256(args.upstream_source),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": checkpoint_digest,
        "weight_provenance": "pretrained_checkpoint" if args.checkpoint else "upstream_random_initialization",
        "input_provenance": str(args.input_npy) if args.input_npy else "torch.linspace(-1,1)",
        "input_npy_sha256": sha256(args.input_npy) if args.input_npy else None,
        "seed": args.seed, "image_size": args.image_size, "patch_size": 16,
        "embed_dim": args.embed_dim, "depth": args.depth,
        "spectral_blocks": 4, "attention_blocks": args.depth - 4,
        "heads": 6, "num_classes": args.num_classes,
        "base_address": BASE_ADDRESS, "image_size_bytes": offset,
        "image_sha256": sha256(path), "entries": entries,
        "reference": "upstream SpectFormer.eval() FP32 CPU",
    }
    result = args.out / "manifest.json"
    result.write_text(json.dumps(manifest, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input-npy", type=Path,
                        help="float32 NCHW input [1,3,H,W]; default is a fixed linspace control")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1827)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(export(args))


if __name__ == "__main__":
    main()
