#!/usr/bin/env python3
"""Relocate verified checkpoint images without changing their bytes.

The output manifests refer to read-only source files through symlinks. Package
the resulting images into a FireSim ELF before transferring the workload.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from split_decoder_weights import (ALIGNMENT, image_segments, verify_image,
                                   verify_image_placement)


def relocate(manifests: list[Path], bases: list[int], device_elf: Path,
             output_dir: Path, gpu_size: int) -> list[Path]:
    originals = [verify_image(path) for path in manifests]
    if len(bases) != sum(len(image_segments(image)) for image, _ in originals):
        raise ValueError("provide one new base for every image segment")
    if gpu_size <= 0 or gpu_size > 1 << 32:
        raise ValueError("invalid GPU region size")
    relocated = []
    sources = []
    next_base = iter(bases)
    for image_index, (image, paths) in enumerate(originals):
        result = copy.deepcopy(image)
        segments = image_segments(result)
        for segment, source in zip(segments, paths):
            base = next(next_base)
            if base < 0 or base % 4096 or base + segment["image_size_bytes"] > gpu_size:
                raise ValueError("image exceeds the GPU region or is not page aligned")
            delta = base - segment["gpu_base_address"]
            segment["gpu_base_address"] = base
            segment["image_file"] = f"image-{image_index}-{segment['index']}.bin"
            sources.append((segment["image_file"], source.resolve()))
            for parameter in result["parameters"]:
                if parameter.get("segment_index", 0) == segment["index"]:
                    parameter["gpu_address"] += delta
                    if "scale_gpu_address" in parameter:
                        parameter["scale_gpu_address"] += delta
        if "segments" not in result:
            result["gpu_base_address"] = segments[0]["gpu_base_address"]
            result["image_file"] = segments[0]["image_file"]
        result["gpu_base_address"] = min(s["gpu_base_address"] for s in segments)
        result["gpu_end_address_exclusive"] = max(
            s["gpu_base_address"] + s["image_size_bytes"] for s in segments)
        result["relocated_from"] = str(manifests[image_index].resolve())
        relocated.append(result)
    verify_image_placement(device_elf, relocated)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, image in enumerate(relocated):
        name = f"image-{index}.json"
        destination = output_dir / name
        for filename, source in sources:
            link = output_dir / filename
            if link.exists() or link.is_symlink():
                if not link.is_symlink() or link.resolve() != source:
                    raise ValueError(f"{link}: existing image link differs")
            else:
                link.symlink_to(source)
        destination.write_text(json.dumps(image, indent=2) + "\n")
        verify_image(destination)
        outputs.append(destination)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, action="append", required=True,
                        help="source image manifest; repeat for each image")
    parser.add_argument("--base", type=lambda value: int(value, 0),
                        action="append", required=True,
                        help="new base for each segment in image order")
    parser.add_argument("--device-elf", type=Path, required=True)
    parser.add_argument("--gpu-size", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in relocate(args.image, args.base, args.device_elf,
                         args.out_dir, args.gpu_size):
        print(path)


if __name__ == "__main__":
    main()
