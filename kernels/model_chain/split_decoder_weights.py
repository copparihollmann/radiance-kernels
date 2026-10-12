#!/usr/bin/env python3
"""Place a verified checkpoint image in two disjoint Radiance GMEM regions.

Parameters stay FP32 and byte-identical. The split is at a parameter boundary;
neither blob may enter Muon's warp stacks or the console MMIO window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct


ALIGNMENT = 64
STACK_BOTTOM = 0x7EE00000  # lib/src/mu_start.S, two cores × eight warps
STACK_TOP = 0x7F000000
CONSOLE_MMIO = 0xFF080000  # Cyclotron config.toml


def image_segments(manifest: dict) -> list[dict]:
    if "segments" in manifest:
        return manifest["segments"]
    return [{"index": 0, "image_file": manifest["image_file"],
             "gpu_base_address": manifest["gpu_base_address"],
             "image_size_bytes": manifest["image_size_bytes"],
             "image_sha256": manifest["image_sha256"]}]


def verify_image_placement(elf: Path, images: list[dict]) -> None:
    """Reject preloads that overwrite ELF segments, stacks, or one another."""
    with elf.open("rb") as stream:
        header = stream.read(52)
        if (len(header) != 52 or header[:7] != b"\x7fELF\x01\x01\x01"):
            raise ValueError("expected a little-endian ELF32 device executable")
        phoff = struct.unpack_from("<I", header, 28)[0]
        phentsize, phnum = struct.unpack_from("<HH", header, 42)
        if phentsize < 32 or phnum == 0:
            raise ValueError("device ELF has no valid program headers")
        occupied = [(STACK_BOTTOM, STACK_TOP, "warp stacks"),
                    (CONSOLE_MMIO, 1 << 32, "console MMIO")]
        for index in range(phnum):
            stream.seek(phoff + index * phentsize)
            program = stream.read(32)
            if len(program) != 32:
                raise ValueError("truncated device ELF program header")
            kind, _, vaddr, _, _, memsz, _, _ = struct.unpack("<8I", program)
            if kind == 1 and memsz:
                occupied.append((vaddr, vaddr + memsz, "ELF LOAD segment"))
    for image in images:
        for segment in image_segments(image):
            start = segment["gpu_base_address"]
            end = start + segment["image_size_bytes"]
            for low, high, label in occupied:
                if start < high and end > low:
                    raise ValueError(
                        f"preload [{start:#x},{end:#x}) overlaps {label} "
                        f"[{low:#x},{high:#x})")
            occupied.append((start, end, "another preload"))


def verify_image(manifest_path: Path) -> tuple[dict, list[Path]]:
    manifest = json.loads(manifest_path.read_text())
    segments = image_segments(manifest)
    if not segments or sum(item["image_size_bytes"] for item in segments) != manifest[
            "image_size_bytes"]:
        raise ValueError("weight segment sizes disagree with image manifest")
    overall = hashlib.sha256()
    paths = []
    ranges = []
    for index, item in enumerate(segments):
        if item["index"] != index:
            raise ValueError("weight segments must be ordered by index")
        base, size = item["gpu_base_address"], item["image_size_bytes"]
        if base < 0 or base % ALIGNMENT or size < 0 or base + size > 1 << 32:
            raise ValueError("weight segment has invalid GPU address range")
        if any(base < high and base + size > low for low, high in ranges):
            raise ValueError("weight segments overlap")
        ranges.append((base, base + size))
        path = manifest_path.parent / item["image_file"]
        if path.stat().st_size != size:
            raise ValueError(f"{path}: size differs from image manifest")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                overall.update(block)
        if digest.hexdigest() != item["image_sha256"]:
            raise ValueError(f"{path}: SHA-256 differs from image manifest")
        paths.append(path)
    if overall.hexdigest() != manifest["image_sha256"]:
        raise ValueError("combined weight-image SHA-256 differs from manifest")
    for item in manifest["parameters"]:
        index = item.get("segment_index", 0)
        if not 0 <= index < len(segments):
            raise ValueError(f"{item['logical_name']}: invalid segment index")
        segment = segments[index]
        offset, size = item["offset_bytes"], item["size_bytes"]
        if (offset < 0 or offset % ALIGNMENT or offset + size >
                segment["image_size_bytes"] or item["gpu_address"] !=
                segment["gpu_base_address"] + offset):
            raise ValueError(f"{item['logical_name']}: address or size differs from segment")
        if item.get("storage_dtype") == "int8_scaled":
            scale_offset = item["scale_offset_bytes"]
            if (scale_offset % ALIGNMENT or
                    scale_offset < offset + item["weight_size_bytes"] or
                    scale_offset + 4 * item["scale_count"] != offset + size or
                    item["scale_gpu_address"] != segment["gpu_base_address"] +
                    scale_offset):
                raise ValueError(f"{item['logical_name']}: INT8 scale address differs from segment")
    return manifest, paths


def plan_split(source: dict, first_base: int, second_base: int,
               first_limit: int = STACK_BOTTOM,
               second_limit: int = 0xFF000000) -> dict:
    if "segments" in source:
        raise ValueError("split input must be one contiguous image")
    if (first_base < 0x20000000 or first_base % ALIGNMENT or
            first_limit > STACK_BOTTOM or first_base >= first_limit or
            second_base < STACK_TOP or second_base % ALIGNMENT or
            second_limit > CONSOLE_MMIO or second_base >= second_limit):
        raise ValueError("segment bases or limits intersect reserved device memory")
    segments = [
        {"index": 0, "image_file": "weights-0.bin",
         "gpu_base_address": first_base, "image_size_bytes": 0},
        {"index": 1, "image_file": "weights-1.bin",
         "gpu_base_address": second_base, "image_size_bytes": 0},
    ]
    limits = [first_limit, second_limit]
    chosen = 0
    parameters = []
    for original in source["parameters"]:
        item = dict(original)
        offset = (segments[chosen]["image_size_bytes"] + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        if segments[chosen]["gpu_base_address"] + offset + item["size_bytes"] > limits[chosen]:
            chosen += 1
            if chosen == len(segments):
                raise ValueError(f"{item['logical_name']}: FP32 parameter does not fit either region")
            offset = 0
        item["source_offset_bytes"] = original["offset_bytes"]
        item["segment_index"] = chosen
        item["offset_bytes"] = offset
        item["gpu_address"] = segments[chosen]["gpu_base_address"] + offset
        item["overlaps_warp_stacks"] = False
        segments[chosen]["image_size_bytes"] = offset + item["size_bytes"]
        parameters.append(item)
    segments = [item for item in segments if item["image_size_bytes"]]
    result = dict(source)
    result.pop("image_file", None)
    result.update(
        format="radiance-checkpoint-segments-v1",
        source_image_sha256=source["image_sha256"],
        image_sha256=None,
        image_size_bytes=sum(item["image_size_bytes"] for item in segments),
        gpu_base_address=first_base,
        gpu_end_address_exclusive=max(item["gpu_base_address"] + item["image_size_bytes"]
                                      for item in segments),
        segments=segments,
        parameters=parameters,
        stack_overlap_parameters=[],
        device_execution=False,
    )
    return result


def write_split(source_path: Path, output_dir: Path, first_base: int,
                second_base: int, first_limit: int = STACK_BOTTOM,
                second_limit: int = 0xFF000000) -> dict:
    source, paths = verify_image(source_path)
    if len(paths) != 1:
        raise ValueError("split input must contain one raw image file")
    result = plan_split(source, first_base, second_base,
                        first_limit, second_limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = hashlib.sha256()
    with paths[0].open("rb") as original:
        for segment in result["segments"]:
            target = output_dir / segment["image_file"]
            temporary = target.with_suffix(target.suffix + ".tmp")
            digest = hashlib.sha256()
            with temporary.open("wb") as stream:
                for item in result["parameters"]:
                    if item["segment_index"] != segment["index"]:
                        continue
                    gap = item["offset_bytes"] - stream.tell()
                    if gap < 0 or gap >= ALIGNMENT:
                        raise ValueError("invalid segment parameter alignment")
                    pad = bytes(gap)
                    stream.write(pad)
                    digest.update(pad)
                    overall.update(pad)
                    original.seek(item["source_offset_bytes"])
                    item_hash = hashlib.sha256()
                    remaining = item["size_bytes"]
                    while remaining:
                        block = original.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ValueError("source image ended within a parameter")
                        stream.write(block)
                        item_hash.update(block)
                        digest.update(block)
                        overall.update(block)
                        remaining -= len(block)
                    if item_hash.hexdigest() != item["packed_sha256"]:
                        raise ValueError(f"{item['logical_name']}: source tensor hash differs")
                if stream.tell() != segment["image_size_bytes"]:
                    raise ValueError("segment size differs from placement plan")
            os.replace(temporary, target)
            segment["image_sha256"] = digest.hexdigest()
    result["image_sha256"] = overall.hexdigest()
    target_manifest = output_dir / "weights-image.json"
    target_manifest.write_text(json.dumps(result, indent=2) + "\n")
    verify_image(target_manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--first-base", type=lambda value: int(value, 0),
                        default=0x30000000)
    parser.add_argument("--second-base", type=lambda value: int(value, 0),
                        default=0x80000000)
    args = parser.parse_args()
    result = write_split(args.source_image, args.out_dir,
                         args.first_base, args.second_base)
    print(f"{result['model']}: {len(result['segments'])} segments, "
          f"{result['image_size_bytes']} bytes, SHA-256 {result['image_sha256']}")


if __name__ == "__main__":
    main()
