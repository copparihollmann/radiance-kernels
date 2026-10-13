#!/usr/bin/env python3
"""Compare connected block outputs with hooks on the pinned upstream graph."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from reference import make_model as make_vanilla
from hierarchical_prepare import make_model as make_hierarchical


def compare(image: dict, build: dict, pixels: np.ndarray,
            trace_dir: Path) -> dict:
    import torch

    model = (make_hierarchical(image) if image["format"] ==
             "spectformer.hierarchical.fp32.v1" else make_vanilla(image))
    points = build["trace_points"]
    captured: dict[str, np.ndarray] = {}
    hooks = []
    modules = dict(model.named_modules())
    for point in points:
        name = point["name"]
        if name not in modules:
            raise ValueError(f"upstream graph lacks block {name}")
        hooks.append(modules[name].register_forward_hook(
            lambda _module, _inputs, output, label=name:
                captured.__setitem__(label, output.detach().cpu().numpy().reshape(-1))))
    with torch.no_grad():
        model(torch.from_numpy(pixels.copy()))
    for hook in hooks:
        hook.remove()
    results = []
    for index, point in enumerate(points):
        path = trace_dir / f"trace-{index:03d}.bin"
        actual = np.fromfile(path, dtype="<f4")
        expected = captured[point["name"]]
        if actual.size != point["count"] or expected.size != point["count"]:
            raise ValueError(f"{point['name']}: wrong trace length")
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        close = bool(np.isfinite(actual).all() and np.allclose(
            actual, expected, rtol=build["logit_rtol"],
            atol=build["logit_atol"]))
        results.append({"name": point["name"], "all_close": close,
                        "max_abs_error": float(np.max(np.abs(delta))),
                        "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest()})
    return {"status": "passed" if all(item["all_close"] for item in results) else "failed",
            "checked_blocks": len(results), "blocks": results,
            "upstream_source_sha256": image["upstream_source_sha256"]}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--input-npy", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    args = parser.parse_args()
    image = json.loads(args.image_manifest.read_text())
    build = json.loads(args.build_manifest.read_text())
    pixels = np.load(args.input_npy, allow_pickle=False)
    result = compare(image, build, pixels, args.trace_dir)
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
