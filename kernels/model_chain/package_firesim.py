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
        entries.append((address, size, path, segment["image_sha256"]))
    return entries


def check_placement(existing, entries):
    ranges = [(vaddr, vaddr + memsz, "ELF")
              for kind, _, _, vaddr, _, _, memsz, _ in existing
              if kind == PT_LOAD and memsz]
    ranges.extend((address, address + size, str(path))
                  for address, size, path, _ in entries)
    for left, right in zip(sorted(ranges), sorted(ranges)[1:]):
        if left[1] > right[0]:
            raise ValueError(f"overlapping load regions: {left} and {right}")
    for address, size, path, _ in entries:
        if address < GPU_HOST_ALIAS or address + size > GPU_HOST_ALIAS + (1 << 32):
            raise ValueError(f"{path}: image lies outside 32-bit GPU address space")


def package(elf, image_manifests, output):
    header, old_phdrs, existing = read_elf(elf)
    entries = [entry for manifest in image_manifests
               for entry in image_entries(manifest)]
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
            for address, size, path, expected_hash in entries:
                offset = align(target.tell())
                target.write(b"\0" * (offset - target.tell()))
                with path.open("rb") as image:
                    shutil.copyfileobj(image, target, 8 * 1024 * 1024)
                if target.tell() - offset != size:
                    raise ValueError(f"{path}: changed during packaging")
                added_phdrs.extend(struct.pack("<IIQQQQQQ", PT_LOAD, 6,
                                               offset, address, address, size,
                                               size, ALIGN))
                manifest_entries.append({"source": str(path.resolve()),
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
    output = Path(result["firesim_elf"])
    if digest(output) != result["firesim_elf_sha256"]:
        raise ValueError(f"{output}: packaged ELF hash differs")
    _, _, phdrs = read_elf(output)
    for image in result["images"]:
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
            "embedded_images_verified": len(result["images"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", type=Path)
    parser.add_argument("--image", type=Path, action="append",
                        help="weights-image.json or inputs-image.json; repeatable")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path,
                        help="verify a packaged ELF using its .elf.json receipt")
    args = parser.parse_args()
    if args.verify:
        if args.elf or args.image or args.output:
            parser.error("--verify cannot be combined with packaging arguments")
        result = verify_package(args.verify)
    else:
        if not args.elf or not args.image or not args.output:
            parser.error("--elf, --image, and --output are required for packaging")
        result = package(args.elf, args.image, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
