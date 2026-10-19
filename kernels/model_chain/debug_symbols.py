#!/usr/bin/env python3
"""Verify that a GPU debug ELF preserves a qualified ELF's loaded memory."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_signature(path):
    data = path.read_bytes()
    if len(data) < 52 or data[:6] != b'\x7fELF\x01\x01':
        raise ValueError('expected a little-endian ELF32 device image')
    entry, phoff = struct.unpack_from('<II', data, 24)
    phsize, phnum = struct.unpack_from('<HH', data, 42)
    if phsize != 32 or not phnum or phoff + phsize * phnum > len(data):
        raise ValueError('invalid ELF32 program headers')
    segments = []
    for index in range(phnum):
        kind, offset, virtual, physical, size, memory, flags, alignment = struct.unpack_from(
            '<8I', data, phoff + index * phsize)
        if kind != 1:
            continue
        if size > memory or offset + size > len(data):
            raise ValueError('invalid loaded segment size')
        segments.append(dict(virtual_address=virtual, physical_address=physical,
            file_bytes=size, memory_bytes=memory, flags=flags, alignment=alignment,
            sha256=hashlib.sha256(data[offset:offset + size]).hexdigest()))
    if not segments:
        raise ValueError('device image has no loaded segments')
    return dict(entry=entry, segments=segments)


def verify_pair(qualified, debug):
    original = load_signature(qualified)
    return dict(passed=original == load_signature(debug),
        qualified_sha256=digest(qualified), debug_sha256=digest(debug),
        scope='entry point and every PT_LOAD address, size, BSS extent, flag, alignment and byte',
        loaded_memory=original)


def local(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('path escapes bundle')
    return path


def verify_bundle(base, companion):
    manifest = json.loads((base / 'manifest.json').read_text())
    debug = json.loads((companion / 'manifest.json').read_text())
    if debug['base_manifest_sha256'] != digest(base / 'manifest.json'):
        raise ValueError('debug symbols target a different workload manifest')
    workloads = {row['name']: row for row in manifest['workloads']}
    rows = []
    for item in debug['workloads']:
        row = workloads[item['workload']]
        original = local(base, row['device_elf']['path'])
        symbols = local(companion, item['debug_elf'])
        result = verify_pair(original, symbols)
        result['passed'] &= (result['qualified_sha256'] == row['device_elf']['sha256'] ==
            item['qualified_device_elf_sha256'] and result['debug_sha256'] == item['debug_elf_sha256'])
        rows.append(dict(workload=item['workload'], passed=result['passed']))
    names = [row['workload'] for row in rows]
    complete = len(names) == len(set(names)) == len(workloads) and set(names) == set(workloads)
    return dict(passed=complete and all(row['passed'] for row in rows), workloads=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    pair = sub.add_parser('pair')
    pair.add_argument('--qualified', type=Path, required=True)
    pair.add_argument('--debug', type=Path, required=True)
    bundle = sub.add_parser('bundle')
    bundle.add_argument('--base-bundle', type=Path, required=True)
    bundle.add_argument('--debug-bundle', type=Path, required=True)
    args = parser.parse_args()
    result = (verify_pair(args.qualified, args.debug) if args.command == 'pair' else
              verify_bundle(args.base_bundle, args.debug_bundle))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
