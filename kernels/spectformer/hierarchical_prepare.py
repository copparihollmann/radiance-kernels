#!/usr/bin/env python3
"""Export the pinned upstream hierarchical SpectFormer and its FP32 golden."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

from prepare import BASE_ADDRESS, ALIGNMENT, UPSTREAM_REVISION, sha256
from reference import load_input


SOURCE_SHA256 = "38f0097924ecb270d6b7b1e90daf2423c1ffc6c6cea70c570e11f89bb73e68c6"
VARIANTS = {"h-s": "spectformer_s", "h-b": "spectformer_b", "h-l": "spectformer_l"}


def load_source(path: Path):
    if sha256(path) != SOURCE_SHA256:
        raise ValueError("hierarchical source differs from the pinned upstream revision")
    if "spectformer_hierarchical_upstream" in sys.modules:
        return sys.modules["spectformer_hierarchical_upstream"]
    # pytorch_wavelets is imported by the source but never used by this graph.
    if ("pytorch_wavelets" not in sys.modules and
            importlib.util.find_spec("pytorch_wavelets") is None):
        shim = types.ModuleType("pytorch_wavelets")
        shim.DWTForward = None
        shim.DWTInverse = None
        sys.modules["pytorch_wavelets"] = shim
    spec = importlib.util.spec_from_file_location("spectformer_hierarchical_upstream", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_model(manifest: dict):
    import torch

    torch.manual_seed(manifest["seed"])
    upstream = load_source(Path(manifest["upstream_source"]))
    model = getattr(upstream, VARIANTS[manifest["variant"]])(
        pretrained=False, num_classes=manifest["num_classes"],
        token_label=True, drop_path_rate=0.0)
    checkpoint_path = manifest.get("checkpoint")
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
            raise ValueError("checkpoint differs from export")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return model.eval()


def reference_logits(manifest: dict, pixels: np.ndarray) -> np.ndarray:
    import torch

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = make_model(manifest)
    with torch.no_grad():
        result = model(torch.from_numpy(pixels.copy())).detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("upstream produced nonfinite logits")
    return np.ascontiguousarray(result, dtype="<f4")


def export(args: argparse.Namespace) -> Path:
    import torch

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "spectformer.hierarchical.fp32.v1",
                "variant": args.variant, "image_size": 224,
                "num_classes": 1000, "seed": args.seed,
                "upstream_revision": UPSTREAM_REVISION,
                "upstream_source": str(args.upstream_source.resolve()),
                "upstream_source_sha256": SOURCE_SHA256,
                "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
                "weight_provenance": "trained_checkpoint" if args.checkpoint else
                                     "upstream_random_initialization"}
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = make_model(manifest)
    if args.input:
        pixels = load_input(args.input, 224)
    else:
        pixels = torch.linspace(-1, 1, 3 * 224 * 224,
                                dtype=torch.float32).reshape(1, 3, 224, 224).numpy()
    golden = reference_logits(manifest, pixels)
    linear_names = {name for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear)}
    entries: dict[str, dict] = {}
    offset = 0
    path = args.out / "image.bin"

    def add(stream, name: str, values: np.ndarray) -> None:
        nonlocal offset
        padding = (-offset) % ALIGNMENT
        stream.write(bytes(padding))
        offset += padding
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
            if name.endswith(".weight") and name[:-7] in linear_names:
                values = values.T
            add(stream, name, values)
        add(stream, "input.image", pixels)
        add(stream, "output.logits", golden)
        padding = (-offset) % ALIGNMENT
        stream.write(bytes(padding))
        offset += padding
    if BASE_ADDRESS + offset >= 0x7EE00000:
        raise ValueError("weights overlap reserved Muon stacks")
    with path.open("rb") as stream:
        weights = stream.read(entries["input.image"]["offset_bytes"])
        input_bytes = stream.read(entries["input.image"]["size_bytes"])
        stream.seek(entries["output.logits"]["offset_bytes"])
        golden_bytes = stream.read(entries["output.logits"]["size_bytes"])
    (args.out / "weights.bin").write_bytes(weights)
    (args.out / "input.bin").write_bytes(input_bytes)
    (args.out / "golden.bin").write_bytes(golden_bytes)
    manifest.update({
        "base_address": BASE_ADDRESS, "image_size_bytes": offset,
        "image_sha256": sha256(path), "weights_sha256": sha256(args.out / "weights.bin"),
        "weights_size_bytes": len(weights),
        "input_sha256": sha256(args.out / "input.bin"),
        "golden_sha256": sha256(args.out / "golden.bin"),
        "entries": entries, "input_provenance": str(args.input) if args.input else
                                                "torch.linspace(-1,1)",
        "reference": "upstream hierarchical SpectFormer.eval() FP32 CPU"})
    result = args.out / "manifest.json"
    result.write_text(json.dumps(manifest, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--seed", type=int, default=1827)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(export(args))


if __name__ == "__main__":
    main()
