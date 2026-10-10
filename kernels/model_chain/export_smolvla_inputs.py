#!/usr/bin/env python3
"""Pack the exact upstream SmolVLA golden inputs into GMEM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from export_decoder_weights import ALIGNMENT
from split_decoder_weights import verify_image


HERE = Path(__file__).resolve().parent
BASE = 0x20000000  # ELF starts at 0x10000000; weights begin at 0x30000000.


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export(golden_path: Path, output_dir: Path) -> dict:
    golden = json.loads(golden_path.read_text())
    if (golden["model"] != "smolvla_base" or not golden["passed"] or
            golden["output_shape"] != [1, 50, 32]):
        raise ValueError("golden SmolVLA action chunk is missing or invalid")
    image = torch.linspace(-1.0, 1.0, 3 * 512 * 512,
                           dtype=torch.float32).reshape(1, 3, 512, 512)
    fields = {}
    for camera in range(3):
        fields[f"camera{camera}.image"] = torch.roll(
            image, shifts=camera * 17, dims=-1).numpy()
        fields[f"camera{camera}.valid"] = np.ones(1, dtype=np.uint32)
    language = torch.arange(1, 49, dtype=torch.int64)[None].numpy()
    fields["language.ids"] = language.astype(np.int32)
    fields["language.valid"] = np.ones((1, 48), dtype=np.uint32)
    fields["robot.state"] = torch.linspace(-0.5, 0.5, 32,
                                             dtype=torch.float32)[None].numpy()
    fields["action.noise"] = torch.linspace(-0.25, 0.25, 50 * 32,
                                              dtype=torch.float32).reshape(1, 50, 32).numpy()
    source_hashes = {
        **{f"camera{camera}": sha256(fields[f"camera{camera}.image"].tobytes())
           for camera in range(3)},
        "language": sha256(language.tobytes()),
        "state": sha256(fields["robot.state"].tobytes()),
        "noise": sha256(fields["action.noise"].tobytes()),
    }
    if source_hashes != golden["input_sha256"]:
        raise ValueError("generated input tensors differ from upstream golden fixture")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "inputs.bin"
    temporary = output_dir / "inputs.bin.tmp"
    parameters = []
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as stream:
            for name, value in fields.items():
                offset = (stream.tell() + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
                pad = bytes(offset - stream.tell())
                stream.write(pad)
                digest.update(pad)
                array = np.ascontiguousarray(value)
                raw = memoryview(array).cast("B")
                stream.write(raw)
                digest.update(raw)
                parameters.append({
                    "logical_name": name, "shape": list(array.shape),
                    "dtype": str(array.dtype), "offset_bytes": offset,
                    "gpu_address": BASE + offset, "size_bytes": len(raw),
                    "packed_sha256": sha256(raw),
                })
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    size = destination.stat().st_size
    if BASE + size > 0x30000000:
        raise ValueError("SmolVLA input image exceeds its reserved GMEM region")
    manifest = {
        "model": "smolvla_base", "scope": "exact_upstream_policy_input_fixture",
        "image_file": destination.name, "gpu_base_address": BASE,
        "image_size_bytes": size, "gpu_end_address_exclusive": BASE + size,
        "image_sha256": digest.hexdigest(), "parameters": parameters,
        "source_input_sha256": source_hashes,
        "source_golden_output_sha256": golden["output_sha256"],
        "checkpoint_weight_sha256": golden["checkpoint_weight_sha256"],
        "device_execution": False,
    }
    path = output_dir / "inputs-image.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    verify_image(path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path,
                        default=ROOT / "kernels/evaluation/llm/smolvla-policy-results.json")
    parser.add_argument("--out-dir", type=Path,
                        default=HERE / "generated/smolvla-inputs")
    args = parser.parse_args()
    result = export(args.golden, args.out_dir)
    print(f"SmolVLA input image: {result['image_size_bytes']} bytes, "
          f"SHA-256 {result['image_sha256']}")


ROOT = HERE.parents[1]


if __name__ == "__main__":
    main()
