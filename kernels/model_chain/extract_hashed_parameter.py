#!/usr/bin/env python3
"""Recover a verified parameter range from a compact-hash handoff ZIP.

The model ELF must be stored without ZIP compression. This reads physical
PT_LOAD bytes directly from the ZIP and writes only the requested parameter,
without unpacking a second multi-gigabyte ELF. The output manifest can drive
a checkpoint-backed stage-prefix build.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
from zipfile import ZIP_STORED, ZipFile

from package_firesim import (digest_region, file_regions_from_layout,
                             hash_folds, read_physical)
from split_decoder_weights import image_segments


def zip_data_offset(path: Path, header_offset: int) -> int:
    with path.open("rb") as stream:
        stream.seek(header_offset)
        local = stream.read(30)
    if len(local) != 30 or local[:4] != b"PK\x03\x04":
        raise ValueError("invalid ZIP local file header")
    name_size, extra_size = struct.unpack_from("<HH", local, 26)
    return header_offset + 30 + name_size + extra_size


def digest_stream(source) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def extract(path: Path, *, elf_member: str, parameters: list[str],
            out_dir: Path, image_manifest_path: Path | None = None,
            all_parameters: bool = False) -> dict:
    with ZipFile(path) as archive:
        elf_info = archive.getinfo(elf_member)
        if elf_info.compress_type != ZIP_STORED:
            raise ValueError("model ELF must be ZIP_STORED for bounded-space extraction")
        prefix = elf_member.rsplit("/", 1)[0] + "/"
        receipt = json.loads(archive.read(elf_member + ".json"))
        original = json.loads(image_manifest_path.read_text()
                              if image_manifest_path is not None else
                              archive.read(prefix + "weights-image.json"))
        with archive.open(elf_info) as source:
            if digest_stream(source) != receipt["firesim_elf_sha256"]:
                raise ValueError("archived model ELF differs from its receipt")
    if all_parameters:
        if parameters:
            raise ValueError("choose either named parameters or all parameters")
        parameters = [item["logical_name"] for item in original["parameters"]]
    if not parameters or len(set(parameters)) != len(parameters):
        raise ValueError("request distinct parameter names")
    items = [item for item in original["parameters"]
             if item["logical_name"] in parameters]
    if len(items) != len(parameters):
        raise ValueError("weight image lacks a requested parameter")
    if len({item.get("segment_index", 0) for item in items}) != 1:
        raise ValueError("requested parameters must share an image segment")
    layout = receipt.get("hash_layout")
    if (not layout or not layout.get("compact") or
            layout["cpu_base"] != 0x100000000):
        raise ValueError("handoff is not a compact hashed GPU ELF")
    image = image_segments(original)[items[0].get("segment_index", 0)]
    begin = min(item["gpu_address"] for item in items)
    size = max(item["gpu_address"] + item["size_bytes"]
               for item in items) - begin
    if (begin < image["gpu_base_address"] or
            begin + size > image["gpu_base_address"] + image["image_size_bytes"]):
        raise ValueError("parameter lies outside its source image segment")
    if not any(entry.get("layout") == "hashed" and
               begin >= entry["gpu_address"] and
               begin + size <= entry["gpu_address"] + entry["size_bytes"]
               for entry in receipt["images"]):
        raise ValueError("receipt does not cover requested parameter")
    base = zip_data_offset(path, elf_info.header_offset)
    file_regions = [(low, high, base + offset)
                    for low, high, offset in file_regions_from_layout(layout)]
    folds = hash_folds(layout["size"], layout["unit"], layout["slices"])
    unit, slices = layout["unit"], layout["slices"]
    group_bytes = unit * slices
    slice_size = layout["size"] // slices
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = "weights-parameter.bin"
    output = out_dir / filename
    digest = hashlib.sha256()
    consumed = 0
    with path.open("rb") as source, output.open("wb") as target:
        while consumed < size:
            current = begin + consumed
            remaining = size - consumed
            if current % group_bytes == 0 and remaining >= group_bytes:
                groups = min(8192, remaining // group_bytes)
                first = current // group_bytes
                banks = [read_physical(source.fileno(), groups * unit,
                                       lane * slice_size + first * unit,
                                       file_regions)
                         for lane in range(slices)]
                restored = bytearray(groups * group_bytes)
                for index in range(groups):
                    fold = folds[first + index]
                    for lane in range(slices):
                        start = index * group_bytes + lane * unit
                        bank_start = index * unit
                        restored[start:start + unit] = banks[lane ^ fold][
                            bank_start:bank_start + unit]
                block = restored
            else:
                amount = min(unit - current % unit, remaining)
                group = current // group_bytes
                lane = (current // unit) % slices
                physical = ((lane ^ folds[group]) * slice_size +
                            group * unit + current % unit)
                block = read_physical(source.fileno(), amount, physical,
                                      file_regions)
            target.write(block)
            digest.update(block)
            consumed += len(block)
    if output.stat().st_size != size:
        raise ValueError("recovered parameter size differs from manifest")
    image_digest = digest.hexdigest()
    if (len(items) == len(original["parameters"]) and
            begin == original["gpu_base_address"] and
            size == original["image_size_bytes"] and
            image_digest != original["image_sha256"]):
        raise ValueError("recovered full image differs from exporter")
    recovered_items = []
    for item in items:
        offset = item["gpu_address"] - begin
        if item.get("packed_sha256") and digest_region(
                output, offset, item["size_bytes"]) != item["packed_sha256"]:
            raise ValueError(f"{item['logical_name']}: recovered hash differs from exporter")
        recovered = dict(item)
        recovered["segment_index"] = 0
        recovered["offset_bytes"] = offset
        if recovered.get("scale_offset_bytes") is not None:
            recovered["scale_offset_bytes"] = offset + item[
                "scale_offset_bytes"] - item["offset_bytes"]
        recovered_items.append(recovered)
    subset = dict(original)
    subset.update({
        "gpu_base_address": begin,
        "gpu_end_address_exclusive": begin + size,
        "image_size_bytes": size,
        "image_sha256": image_digest,
        "segments": [{"index": 0, "image_file": filename,
                      "gpu_base_address": begin, "image_size_bytes": size,
                      "image_sha256": image_digest}],
        "parameters": recovered_items,
        "distinct_parameters": len(recovered_items),
        "physical_parameters": len(recovered_items),
        "parameter_uses": len(recovered_items),
        "probe_subset": {"parameters": parameters,
                         "source_zip": path.name,
                         "source_elf_sha256": receipt["firesim_elf_sha256"]},
    })
    manifest = out_dir / "weights-image.json"
    manifest.write_text(json.dumps(subset, indent=2) + "\n")
    return {"parameters": parameters, "bytes": size,
            "image_sha256": image_digest,
            "parameter_sha256": {
                item["logical_name"]: item.get("packed_sha256") for item in items},
            "manifest": str(manifest)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--elf-member", required=True)
    parser.add_argument("--image-manifest", type=Path,
                        help="source image manifest when the ZIP has none")
    parser.add_argument("--parameter", action="append", default=[])
    parser.add_argument("--all-parameters", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.zip.resolve(), elf_member=args.elf_member,
                             parameters=args.parameter,
                             out_dir=args.out_dir.resolve(),
                             image_manifest_path=(args.image_manifest.resolve()
                                                  if args.image_manifest else None),
                             all_parameters=args.all_parameters), indent=2))


if __name__ == "__main__":
    main()
