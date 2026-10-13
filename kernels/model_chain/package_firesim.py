#!/usr/bin/env python3
"""Attach checkpoint memory images to a Radiance SoC ELF for FireSim staging.

The standard FireSim workload stages one ELF.  Model-chain kernels keep their
checkpoint weights in separate GPU memory images, so loading the bare ELF
would execute against uninitialised memory.  This tool adds each image as an
ELF PT_LOAD segment at the CPU alias of its GPU address.  It changes neither
the host nor device instructions.  The output still needs target validation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct


PT_LOAD = 1
GPU_HOST_ALIAS = 0x100000000
ALIGN = 4096
ELF_HEADER_SIZE = 64
PHDR_SIZE = 56


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def digest_region(path, offset, size):
    result = hashlib.sha256()
    with path.open("rb") as source:
        source.seek(offset)
        remaining = size
        while remaining:
            block = source.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"{path}: truncated embedded image")
            result.update(block)
            remaining -= len(block)
    return result.hexdigest()


def align(value):
    return (value + ALIGN - 1) & ~(ALIGN - 1)


def read_elf(path):
    with path.open("rb") as source:
        header = source.read(ELF_HEADER_SIZE)
        if header[:6] != b"\x7fELF\x02\x01":
            raise ValueError(f"{path}: expected a little-endian ELF64")
        phoff = struct.unpack_from("<Q", header, 0x20)[0]
        phentsize, phnum = struct.unpack_from("<HH", header, 0x36)
        if phentsize != PHDR_SIZE or not phnum:
            raise ValueError(f"{path}: unsupported program header layout")
        source.seek(phoff)
        phdrs = source.read(phnum * PHDR_SIZE)
        if len(phdrs) != phnum * PHDR_SIZE:
            raise ValueError(f"{path}: truncated program header table")
    segments = [struct.unpack_from("<IIQQQQQQ", phdrs, i * PHDR_SIZE)
                for i in range(phnum)]
    file_size = path.stat().st_size
    for kind, _, offset, _, _, size, _, _ in segments:
        if kind == PT_LOAD and offset + size > file_size:
            raise ValueError(f"{path}: PT_LOAD extends past end of file")
    return header, phdrs, segments


def image_entries(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    segments = manifest.get("segments") or [manifest]
    entries = []
    for segment in segments:
        path = manifest_path.parent / segment["image_file"]
        size = int(segment["image_size_bytes"])
        if path.stat().st_size != size:
            raise ValueError(f"{path}: size differs from manifest")
        if digest(path) != segment["image_sha256"]:
            raise ValueError(f"{path}: SHA-256 differs from manifest")
        address = GPU_HOST_ALIAS + int(segment["gpu_base_address"])
        if address % ALIGN or size % 64:
            raise ValueError(f"{path}: expected page-aligned address and cache-line sized image")
        entries.append((address, size, path, segment["image_sha256"], 0))
    return entries


def receipt_entries(receipt_path):
    """Reuse verified image segments from a portable logical handoff ELF."""
    receipt = json.loads(receipt_path.read_text())
    donor = receipt_path.with_suffix("")
    if digest(donor) != receipt["firesim_elf_sha256"]:
        raise ValueError(f"{donor}: handoff ELF differs from its receipt")
    _, _, segments = read_elf(donor)
    entries = []
    for image in receipt["images"]:
        address = int(image["cpu_address"])
        size = int(image["size_bytes"])
        offset = int(image["file_offset"])
        if not any(kind == PT_LOAD and vaddr == address and
                   file_offset == offset and filesz == size
                   for kind, _, file_offset, vaddr, _, filesz, _, _ in segments):
            raise ValueError(f"{donor}: image segment differs from receipt")
        if digest_region(donor, offset, size) != image["source_sha256"]:
            raise ValueError(f"{donor}: embedded image differs from receipt")
        entries.append((address, size, donor, image["source_sha256"], offset))
    return entries


def collect_entries(image_manifests, receipts):
    return ([entry for manifest in image_manifests
             for entry in image_entries(manifest)] +
            [entry for receipt in receipts
             for entry in receipt_entries(receipt)])


def check_placement(existing, entries):
    ranges = [(vaddr, vaddr + memsz, "ELF")
              for kind, _, _, vaddr, _, _, memsz, _ in existing
              if kind == PT_LOAD and memsz]
    ranges.extend((address, address + size, str(path))
                  for address, size, path, _, _ in entries)
    for left, right in zip(sorted(ranges), sorted(ranges)[1:]):
        if left[1] > right[0]:
            raise ValueError(f"overlapping load regions: {left} and {right}")
    for address, size, path, _, _ in entries:
        if address < GPU_HOST_ALIAS or address + size > GPU_HOST_ALIAS + (1 << 32):
            raise ValueError(f"{path}: image lies outside 32-bit GPU address space")


def hash_folds(size, unit, slices):
    if (size <= 0 or size > (1 << 32) or unit <= 0 or slices < 2
            or slices > 256
            or any(value & (value - 1) for value in (size, unit, slices))
            or size % (unit * slices) or size // slices < ALIGN):
        raise ValueError("hash size, unit, and slices must fit the GPU address space and be compatible powers of two")
    bits = slices.bit_length() - 1
    folds = bytearray(size // (unit * slices))
    for group in range(1, len(folds)):
        folds[group] = (group & (slices - 1)) ^ folds[group >> bits]
    return folds


def occupied_hash_ranges(logical_ranges, size, unit, slices):
    """Page-aligned physical spans that contain bytes from logical GPU ranges."""
    group_bytes = unit * slices
    slice_size = size // slices
    ranges = []
    for logical_start, length in logical_ranges:
        if logical_start < 0 or logical_start + length > size:
            raise ValueError("logical GPU range exceeds hash size")
        first = logical_start // group_bytes
        end = (logical_start + length + group_bytes - 1) // group_bytes
        for lane in range(slices):
            start = lane * slice_size + (first * unit // ALIGN) * ALIGN
            stop = lane * slice_size + align(end * unit)
            if start < stop:
                ranges.append((start, min(stop, (lane + 1) * slice_size)))
    merged = []
    for start, stop in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def write_at(fd, data, offset):
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, offset)
        if not written:
            raise OSError("short write while building hashed GPU image")
        view = view[written:]
        offset += written


def scatter_hashed(source, source_offset, address, size, output, region_offset,
                   hash_size, unit, slices, folds):
    """Place a logical GPU byte range into the direct-load physical hash image."""
    logical = address - GPU_HOST_ALIAS
    if logical < 0 or logical + size > hash_size:
        raise ValueError(f"load range {address:#x}+{size:#x} crosses hash region")
    group_bytes = unit * slices
    slice_size = hash_size // slices
    source.seek(source_offset)
    consumed = 0
    while consumed < size:
        current = logical + consumed
        remaining = size - consumed
        if current % group_bytes == 0 and remaining >= group_bytes:
            groups = min(8192, remaining // group_bytes)
            data = source.read(groups * group_bytes)
            if len(data) != groups * group_bytes:
                raise ValueError("source image changed during hash packaging")
            banks = [bytearray(groups * unit) for _ in range(slices)]
            group_start = current // group_bytes
            for index in range(groups):
                fold = folds[group_start + index]
                src_at = index * group_bytes
                dst_at = index * unit
                for lane in range(slices):
                    banks[lane ^ fold][dst_at:dst_at + unit] = data[
                        src_at + lane * unit:src_at + (lane + 1) * unit]
            for lane, bank in enumerate(banks):
                write_at(output.fileno(), bank,
                         region_offset + lane * slice_size + group_start * unit)
            consumed += len(data)
        else:
            amount = min(unit - current % unit, remaining)
            data = source.read(amount)
            if len(data) != amount:
                raise ValueError("source image changed during hash packaging")
            group = current // group_bytes
            lane = (current // unit) % slices
            physical = ((lane ^ folds[group]) * slice_size
                        + group * unit + current % unit)
            write_at(output.fileno(), data, region_offset + physical)
            consumed += amount


def digest_hashed_image(output, address, size, layout, folds):
    """Reconstruct logical bytes from a packed physical hash image and hash them."""
    unit, slices = layout["unit"], layout["slices"]
    group_bytes = unit * slices
    slice_size = layout["size"] // slices
    logical = address - layout["cpu_base"]
    if logical < 0 or logical + size > layout["size"]:
        raise ValueError("image lies outside recorded hash region")
    result = hashlib.sha256()
    with output.open("rb") as source:
        consumed = 0
        while consumed < size:
            current = logical + consumed
            remaining = size - consumed
            if current % group_bytes == 0 and remaining >= group_bytes:
                groups = min(8192, remaining // group_bytes)
                group_start = current // group_bytes
                banks = [os.pread(source.fileno(), groups * unit,
                                  layout["file_offset"] + lane * slice_size
                                  + group_start * unit)
                         for lane in range(slices)]
                if any(len(bank) != groups * unit for bank in banks):
                    raise ValueError("truncated hashed GPU image")
                restored = bytearray(groups * group_bytes)
                for index in range(groups):
                    fold = folds[group_start + index]
                    src_at = index * unit
                    dst_at = index * group_bytes
                    for lane in range(slices):
                        restored[dst_at + lane * unit:
                                 dst_at + (lane + 1) * unit] = banks[
                                     lane ^ fold][src_at:src_at + unit]
                result.update(restored)
                consumed += len(restored)
            else:
                amount = min(unit - current % unit, remaining)
                group = current // group_bytes
                lane = (current // unit) % slices
                physical = ((lane ^ folds[group]) * slice_size
                            + group * unit + current % unit)
                data = os.pread(source.fileno(), amount,
                                layout["file_offset"] + physical)
                if len(data) != amount:
                    raise ValueError("truncated hashed GPU image")
                result.update(data)
                consumed += amount
    return result.hexdigest()


def package_hashed(elf, image_manifests, output, hash_size, unit, slices,
                   receipts=()):
    """Build one LoadMem ELF for the specified Radiance address hash."""
    folds = hash_folds(hash_size, unit, slices)
    header, _, existing = read_elf(elf)
    entries = collect_entries(image_manifests, receipts)
    if not entries:
        raise ValueError("at least one image is required")
    check_placement(existing, entries)
    hash_end = GPU_HOST_ALIAS + hash_size
    hashed_source = []
    kept = []
    for segment in existing:
        kind, _, offset, vaddr, _, filesz, memsz, _ = segment
        if kind == PT_LOAD and memsz and GPU_HOST_ALIAS <= vaddr < hash_end:
            if vaddr + memsz > hash_end:
                raise ValueError(f"ELF segment at {vaddr:#x} crosses hash region")
            if filesz:
                hashed_source.append((offset, vaddr, filesz))
        else:
            kept.append(segment)
    lower = [entry for entry in entries if entry[0] < hash_end]
    upper = [entry for entry in entries if entry[0] >= hash_end]
    if upper:
        raise ValueError(
            "checkpoint images exceed the configured hashed GPU memory; "
            "confirm the target capacity and hash range before packaging")
    for kind, _, _, vaddr, _, _, memsz, _ in kept:
        if (kind == PT_LOAD and memsz and
                GPU_HOST_ALIAS <= vaddr < GPU_HOST_ALIAS + (1 << 32)):
            raise ValueError(
                "device ELF has GPU load bytes outside the configured hash range")
    if not hashed_source and not lower:
        raise ValueError("no GPU load bytes lie in the specified hash region")
    for address, size, path, _, _ in lower:
        if address + size > hash_end:
            raise ValueError(f"{path}: image crosses hash region")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    result_images = []
    source_gpu_segments = [
        {"gpu_address": address - GPU_HOST_ALIAS,
         "cpu_address": address,
         "size_bytes": size,
         "sha256": digest_region(elf, offset, size)}
        for offset, address, size in hashed_source]
    load_ranges = occupied_hash_ranges(
        [(address - GPU_HOST_ALIAS, size)
         for _, address, size in hashed_source] +
        [(address - GPU_HOST_ALIAS, size)
         for address, size, _, _, _ in lower],
        hash_size, unit, slices)
    try:
        with elf.open("rb") as source, partial.open("wb+") as target:
            shutil.copyfileobj(source, target, 8 * 1024 * 1024)
            region_offset = align(target.tell())
            target.seek(region_offset + hash_size - 1)
            target.write(b"\0")
            target.flush()
            for offset, address, size in hashed_source:
                scatter_hashed(source, offset, address, size, target,
                               region_offset, hash_size, unit, slices, folds)
            for address, size, path, expected_hash, source_offset in lower:
                with path.open("rb") as image:
                    scatter_hashed(image, source_offset, address, size, target,
                                   region_offset, hash_size, unit, slices, folds)
                result_images.append({"source": str(path.resolve()),
                                      "source_sha256": expected_hash,
                                      "gpu_address": address - GPU_HOST_ALIAS,
                                      "cpu_address": address,
                                      "size_bytes": size,
                                      "layout": "hashed"})
            for first, end in load_ranges:
                kept.append((PT_LOAD, 6, region_offset + first,
                             GPU_HOST_ALIAS + first, GPU_HOST_ALIAS + first,
                             end - first, end - first, ALIGN))
            target.seek(region_offset + hash_size)
            phoff = align(target.tell())
            target.write(b"\0" * (phoff - target.tell()))
            for segment in kept:
                target.write(struct.pack("<IIQQQQQQ", *segment))
            patched = bytearray(header)
            struct.pack_into("<Q", patched, 0x20, phoff)
            struct.pack_into("<H", patched, 0x38, len(kept))
            target.seek(0)
            target.write(patched)
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    result = {"source_elf": str(elf.resolve()),
              "source_elf_sha256": digest(elf),
              "firesim_elf": str(output.resolve()),
              "firesim_elf_sha256": digest(output),
              "size_bytes": output.stat().st_size,
              "images": result_images,
              "source_gpu_segments": source_gpu_segments,
              "hash_layout": {"gpu_base": 0, "cpu_base": GPU_HOST_ALIAS,
                              "size": hash_size, "unit": unit,
                              "slices": slices, "file_offset": region_offset,
                              "load_regions": [
                                  {"physical_offset": first,
                                   "file_offset": region_offset + first,
                                   "cpu_address": GPU_HOST_ALIAS + first,
                                   "size_bytes": end - first}
                                  for first, end in load_ranges]},
              "load_method": "direct LoadMem, physically hashed GPU region",
              "validation": "ELF structure and source image hashes; target execution pending"}
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result


def package(elf, image_manifests, output, receipts=()):
    header, old_phdrs, existing = read_elf(elf)
    entries = collect_entries(image_manifests, receipts)
    if not entries:
        raise ValueError("at least one image is required")
    check_placement(existing, entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    added_phdrs = bytearray()
    manifest_entries = []
    try:
        with elf.open("rb") as source, partial.open("wb+") as target:
            shutil.copyfileobj(source, target, 8 * 1024 * 1024)
            for address, size, path, expected_hash, source_offset in entries:
                offset = align(target.tell())
                target.write(b"\0" * (offset - target.tell()))
                with path.open("rb") as image:
                    image.seek(source_offset)
                    remaining = size
                    while remaining:
                        data = image.read(min(8 * 1024 * 1024, remaining))
                        if not data:
                            raise ValueError(f"{path}: changed during packaging")
                        target.write(data)
                        remaining -= len(data)
                if target.tell() - offset != size:
                    raise ValueError(f"{path}: changed during packaging")
                added_phdrs.extend(struct.pack("<IIQQQQQQ", PT_LOAD, 6,
                                               offset, address, address, size,
                                               size, ALIGN))
                manifest_entries.append({"source": str(path.resolve()),
                                         "source_file_offset": source_offset,
                                         "source_sha256": expected_hash,
                                         "gpu_address": address - GPU_HOST_ALIAS,
                                         "cpu_address": address,
                                         "file_offset": offset,
                                         "size_bytes": size})
            phoff = align(target.tell())
            target.write(b"\0" * (phoff - target.tell()))
            target.write(old_phdrs + added_phdrs)
            patched = bytearray(header)
            struct.pack_into("<Q", patched, 0x20, phoff)
            struct.pack_into("<H", patched, 0x38,
                             len(existing) + len(entries))
            target.seek(0)
            target.write(patched)
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    result = {"source_elf": str(elf.resolve()),
              "source_elf_sha256": digest(elf),
              "firesim_elf": str(output.resolve()),
              "firesim_elf_sha256": digest(output),
              "size_bytes": output.stat().st_size,
              "images": manifest_entries,
              "load_method": "ELF PT_LOAD at CPU GPU-DRAM aliases",
              "validation": "ELF structure and image hashes; target execution pending"}
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result


def verify_package(manifest_path):
    result = json.loads(manifest_path.read_text())
    output = manifest_path.with_suffix("")
    if digest(output) != result["firesim_elf_sha256"]:
        raise ValueError(f"{output}: packaged ELF hash differs")
    _, _, phdrs = read_elf(output)
    if "hash_layout" in result:
        layout = result["hash_layout"]
        regions = layout.get("load_regions", [
            {"physical_offset": 0, "file_offset": layout["file_offset"],
             "cpu_address": layout["cpu_base"], "size_bytes": layout["size"]}])
        for region in regions:
            expected = (PT_LOAD, region["file_offset"],
                        region["cpu_address"], region["size_bytes"])
            if not any((kind, offset, vaddr, size) == expected
                       for kind, _, offset, vaddr, _, size, _, _ in phdrs):
                raise ValueError(f"{output}: missing hashed GPU load region {expected}")
        if "load_regions" in layout:
            source_ranges = [(item["gpu_address"], item["size_bytes"])
                             for item in result.get("source_gpu_segments", [])]
            image_ranges = [(item["gpu_address"], item["size_bytes"])
                            for item in result["images"]
                            if item.get("layout") == "hashed"]
            required = occupied_hash_ranges(source_ranges + image_ranges,
                                            layout["size"], layout["unit"],
                                            layout["slices"])
            recorded = [(item["physical_offset"],
                         item["physical_offset"] + item["size_bytes"])
                        for item in regions]
            if required != recorded:
                raise ValueError(f"{output}: hashed load regions omit required bytes")
        folds = hash_folds(layout["size"], layout["unit"], layout["slices"])
        for item in result.get("source_gpu_segments", []):
            actual = digest_hashed_image(output, item["cpu_address"],
                                         item["size_bytes"], layout, folds)
            if actual != item["sha256"]:
                raise ValueError(f"{output}: hashed device program differs at "
                                 f"{item['cpu_address']:#x}")
    for image in result["images"]:
        if image.get("layout") == "hashed":
            actual = digest_hashed_image(output, image["cpu_address"],
                                         image["size_bytes"], layout, folds)
            if actual != image["source_sha256"]:
                raise ValueError(f"{output}: hashed image differs at "
                                 f"{image['cpu_address']:#x}")
            continue
        expected = (PT_LOAD, image["file_offset"], image["cpu_address"],
                    image["size_bytes"])
        if not any((kind, offset, vaddr, size) == expected
                   for kind, _, offset, vaddr, _, size, _, _ in phdrs):
            raise ValueError(f"{output}: missing image load segment {expected}")
        actual = digest_region(output, image["file_offset"], image["size_bytes"])
        if actual != image["source_sha256"]:
            raise ValueError(f"{output}: embedded image hash differs at "
                             f"{image['cpu_address']:#x}")
    return {"elf": str(output), "sha256": result["firesim_elf_sha256"],
            "embedded_images_verified": len(result["images"]),
            "hashed_images_present": sum(
                image.get("layout") == "hashed" for image in result["images"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", type=Path)
    parser.add_argument("--image", type=Path, action="append",
                        help="weights-image.json or inputs-image.json; repeatable")
    parser.add_argument("--receipt", type=Path, action="append",
                        help="portable logical handoff .elf.json; repeatable")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hash-size", type=lambda value: int(value, 0),
                        help="GPU hash range size for direct FireSim LoadMem")
    parser.add_argument("--hash-unit", type=int, default=32)
    parser.add_argument("--hash-slices", type=int, default=4)
    parser.add_argument("--verify", type=Path,
                        help="verify a packaged ELF using its .elf.json receipt")
    args = parser.parse_args()
    if args.verify:
        if args.elf or args.image or args.receipt or args.output or args.hash_size:
            parser.error("--verify cannot be combined with packaging arguments")
        result = verify_package(args.verify)
    else:
        if not args.elf or not (args.image or args.receipt) or not args.output:
            parser.error("--elf, --output, and --image or --receipt are required")
        if args.hash_size:
            result = package_hashed(args.elf, args.image or (), args.output,
                                    args.hash_size, args.hash_unit,
                                    args.hash_slices, args.receipt or ())
        else:
            result = package(args.elf, args.image or (), args.output,
                             args.receipt or ())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
