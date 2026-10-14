#!/usr/bin/env python3
"""Move one checkpoint weight segment by changing addresses, without copying bytes."""

import argparse
import json
from pathlib import Path

from split_decoder_weights import (ALIGNMENT, CONSOLE_MMIO, STACK_BOTTOM,
                                   STACK_TOP, image_segments, verify_image)


def relocate(source: Path, output: Path, index: int, base: int) -> dict:
    if output.parent.resolve() != source.parent.resolve() or output == source:
        raise ValueError("new manifest must have a different name in the image directory")
    original, _ = verify_image(source)
    segments = [dict(item) for item in image_segments(original)]
    if not 0 <= index < len(segments) or base % ALIGNMENT or base < 0:
        raise ValueError("invalid segment index or unaligned base address")
    segment = segments[index]
    delta = base - segment["gpu_base_address"]
    end = base + segment["image_size_bytes"]
    if (end > 1 << 32 or base < STACK_TOP and end > STACK_BOTTOM or
            base < CONSOLE_MMIO and end > CONSOLE_MMIO):
        raise ValueError("relocated segment overlaps reserved device memory")
    segment["gpu_base_address"] = base
    parameters = []
    for original_item in original["parameters"]:
        item = dict(original_item)
        if item.get("segment_index", 0) == index:
            item["gpu_address"] += delta
            if item.get("scale_gpu_address") is not None:
                item["scale_gpu_address"] += delta
        parameters.append(item)
    result = {**original, "segments": segments, "parameters": parameters,
              "gpu_base_address": min(item["gpu_base_address"] for item in segments),
              "gpu_end_address_exclusive": max(
                  item["gpu_base_address"] + item["image_size_bytes"]
                  for item in segments),
              "relocation": {"source_manifest": source.name,
                             "segment_index": index, "old_base": base - delta,
                             "new_base": base, "delta_bytes": delta}}
    output.write_text(json.dumps(result, indent=2) + "\n")
    verify_image(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--gpu-base", type=lambda value: int(value, 0), required=True)
    args = parser.parse_args()
    result = relocate(args.source.resolve(), args.output.resolve(),
                      args.segment_index, args.gpu_base)
    print(json.dumps({"image_sha256": result["image_sha256"],
                      "relocation": result["relocation"]}, indent=2))


if __name__ == "__main__":
    main()
