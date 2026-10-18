#!/usr/bin/env python3
"""Build and use a relocatable Radiance workload handoff (Python 3.11+).

The recipe is local build input. Only relative, content-addressed paths enter
the runnable manifest. Historical qualification receipts remain unmodified.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import tomllib


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def local(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('path escapes bundle: ' + name)
    return path


class Store:
    def __init__(self, root):
        self.root = root
        self.cached = {}
        self.patched = {}

    def put(self, source):
        source = Path(source).resolve()
        key = str(source)
        if key in self.cached:
            return self.cached[key]
        sha = digest(source)
        target = self.root / 'blobs' / sha
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            # Copies isolate the handoff from later source writes and symlinks.
            shutil.copyfile(source, target)
            if digest(target) != sha:
                raise ValueError('source changed during copy: ' + key)
        record = dict(path=str(target.relative_to(self.root)), sha256=sha,
                      bytes=target.stat().st_size)
        self.cached[key] = record
        return record

    def adopt_temporary(self, source):
        """Keep a completed private temporary image without copying it again."""
        sha = digest(source)
        target = self.root / 'blobs' / sha
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            # Only called for our private, completed patch temporary. Closing
            # NamedTemporaryFile removes its name; the blob owns the inode.
            os.link(source, target)
        return dict(path=str(target.relative_to(self.root)), sha256=sha,
                    bytes=target.stat().st_size)


def patched_load(store, load, patches):
    """Materialize checked initialization overlays before exporting an image."""
    relevant = [p for p in patches if load['address'] <= p['address'] and
                p['address'] + p['bytes'] <= load['address'] + load['bytes']]
    if not relevant:
        return store.put(load['path'])
    key = (load['path'], load['address'], load['bytes'],
           tuple((p['address'], p['bytes'], p['expected_old_sha256'],
                  p['expected_new_sha256']) for p in relevant))
    if key in store.patched:
        return store.patched[key]
    with tempfile.NamedTemporaryFile(dir=os.environ['TMPDIR']) as tmp:
        shutil.copyfile(load['path'], tmp.name)
        metadata = []
        with open(tmp.name, 'r+b') as target:
            for patch in relevant:
                offset = patch['address'] - load['address']
                target.seek(offset)
                previous = target.read(patch['bytes'])
                if hashlib.sha256(previous).hexdigest() != patch['expected_old_sha256']:
                    raise ValueError('initialization overlay old bytes changed')
                data = Path(patch['path']).read_bytes()
                if len(data) != patch['bytes'] or hashlib.sha256(data).hexdigest() != patch['expected_new_sha256']:
                    raise ValueError('initialization overlay new bytes changed')
                target.seek(offset); target.write(data)
                metadata.append(dict(address=patch['address'], bytes=patch['bytes'],
                                     old_sha256=patch['expected_old_sha256'],
                                     new_sha256=patch['expected_new_sha256']))
        blob = store.adopt_temporary(tmp.name)
    result = dict(blob, original_image_sha256=digest(load['path']), applied_initialization_patches=metadata)
    store.patched[key] = result
    return result


def build(recipe_path, output):
    recipe = json.loads(recipe_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    store = Store(output)
    manifest = dict(schema='radiance-portable-workloads-v1',
                    target=recipe['target'], provenance=recipe['provenance'],
                    workloads=[], evidence={})
    for item in recipe['workloads']:
        cfg = tomllib.loads(Path(item['config']).read_text())
        sim = cfg['sim']
        device = store.put(item['device_elf'])
        row = dict(name=item['name'], family=item['family'],
                   num_clusters=sim['num_clusters'],
                   logical_partitions=item.get('logical_partitions'),
                   device_elf=device, soc_elf=store.put(item['soc_elf']),
                   software_qualification=item['software_qualification'],
                   fpga_execution_verified=False, loads=[], checks=[],
                   smem_init=sim.get('smem_init', []),
                   nightly_host_handshake=sim.get('nightly_host_handshake', False),
                   smem_reports=item.get('smem_reports', []),
                   sources={})
        patches = sim.get('gmem_patches', [])
        for patch in patches:
            owners = [x for x in sim.get('gmem_load', []) if x['address'] <= patch['address'] and
                      patch['address'] + patch['bytes'] <= x['address'] + x['bytes']]
            if len(owners) != 1:
                raise ValueError('initialization overlay does not belong to exactly one load')
        for load in sim.get('gmem_load', []):
            blob = patched_load(store, load, patches)
            if blob['bytes'] != load['bytes']:
                raise ValueError('load size changed: ' + load['path'])
            row['loads'].append(dict(address=load['address'], **blob))
        for check in item.get('checks', []):
            record = {k: v for k, v in check.items() if k != 'reference'}
            if 'reference' in check:
                record['reference'] = store.put(check['reference'])
            row['checks'].append(record)
        for label, source in item.get('sources', {}).items():
            row['sources'][label] = store.put(source)
        row['sources']['qualified_simulator_configuration'] = store.put(item['config'])
        manifest['workloads'].append(row)
        print('staged', row['name'], flush=True)
    for label, source in recipe.get('evidence', {}).items():
        manifest['evidence'][label] = store.put(source)
    for name in ('workload_bundle.py', 'package_firesim.py'):
        shutil.copyfile(Path(__file__).with_name(name), output / name)
    write_json(output / 'manifest.json', manifest)
    (output / 'README.md').write_text(recipe['readme'])
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output / 'SHA256SUMS').write_text(''.join(
        f'{digest(p)}  {p.relative_to(output)}\n' for p in files))
    return manifest


def load_manifest(root):
    return json.loads((root / 'manifest.json').read_text())


def workload(root, name):
    rows = [x for x in load_manifest(root)['workloads'] if x['name'] == name]
    if len(rows) != 1:
        raise ValueError('unknown or duplicate workload: ' + name)
    return rows[0]


def verify(root):
    count = size = 0
    verified = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        p = local(root, name)
        if not p.is_file() or p.is_symlink() or digest(p) != expected:
            raise ValueError('missing or changed bundle file: ' + name)
        count += 1
        size += p.stat().st_size
        verified[name] = expected
    manifest = load_manifest(root)
    names = [x['name'] for x in manifest['workloads']]
    if len(names) != len(set(names)):
        raise ValueError('duplicate workload names')
    def refs(value):
        if isinstance(value, dict):
            if {'path', 'sha256', 'bytes'} <= value.keys():
                p = local(root, value['path'])
                if (p.stat().st_size != value['bytes'] or p.name != value['sha256'] or
                        verified.get(value['path']) != value['sha256']):
                    raise ValueError('invalid blob metadata: ' + str(value))
            for child in value.values():
                refs(child)
        elif isinstance(value, list):
            for child in value:
                refs(child)
    refs(manifest)
    return dict(passed=True, files=count, bytes=size, workloads=len(names))


def prepare(root, name, output, loader, hash_size, hash_unit, hash_slices):
    """Produce a self-contained RV64 boot ELF for one chosen loader path."""
    from package_firesim import package, package_hashed, verify_package, read_elf
    row = workload(root, name)
    for blob in [row['soc_elf'], *row['loads']]:
        if digest(local(root, blob['path'])) != blob['sha256']:
            raise ValueError('input changed: ' + blob['path'])
    output.mkdir(parents=True, exist_ok=False)
    images = []
    for i, load in enumerate(row['loads']):
        # Paths are relative to this staging directory; no source-machine paths.
        p = output / f'image-{i}.json'
        write_json(p, dict(image_file=os.path.relpath(local(root, load['path']), output),
                           gpu_base_address=load['address'],
                           image_size_bytes=load['bytes'],
                           image_sha256=load['sha256']))
        images.append(p)
    source = local(root, row['soc_elf']['path'])
    boot = output / 'boot.elf'
    if loader == 'direct-hashed':
        result = package_hashed(source, images, boot, hash_size,
                                hash_unit, hash_slices, compact=True)
    elif images:
        result = package(source, images, boot)
    else:
        shutil.copyfile(source, boot)
        result = dict(firesim_elf_sha256=digest(boot), images=[],
                      validation='host/device ELF structure only; FPGA unverified')
        read_elf(boot)
        write_json(output / 'boot.elf.json', result)
    verified = verify_package(output / 'boot.elf.json')
    dump_plan = []
    for check in row['checks']:
        entry = {k: check[k] for k in ('file', 'address', 'bytes')}
        if entry not in dump_plan:
            dump_plan.append(entry)
    write_json(output / 'dump-plan.json', dict(
        workload=name, address_space='logical GPU; CPU alias = 0x100000000 + address',
        reads=dump_plan, smem_reports=row.get('smem_reports', []),
        instruction='Read after completion and required cache visibility/drain; save each file under results/.'))
    write_json(output / 'firesim-workload.json', dict(
        benchmark_name=name, common_bootbinary='boot.elf', common_rootfs=None,
        common_simulation_outputs=['uartlog']))
    write_json(output / 'preparation.json', dict(
        workload=name, loader=loader, hash_size=hash_size if loader == 'direct-hashed' else None,
        hash_unit=hash_unit if loader == 'direct-hashed' else None,
        hash_slices=hash_slices if loader == 'direct-hashed' else None,
        boot_sha256=result['firesim_elf_sha256'], structural_verification=verified,
        fpga_execution_verified=False,
        acceptance='Host wrapper forwards device termination only. Complete-output checking is required separately.'))
    return dict(boot=str(boot), **verified)


def check_outputs(root, name, results):
    """Check readbacks; never infer a workload pass from boot/exit alone."""
    import numpy as np
    row = workload(root, name)
    outcomes = []
    for check in row['checks']:
        p = local(results, check['file'])
        ok = p.is_file() and p.stat().st_size == check['bytes']
        record = dict(file=check['file'], passed=False)
        if not ok:
            record['reason'] = 'missing readback or wrong byte count'
            outcomes.append(record)
            continue
        mode = check['mode']
        if mode == 'sha256':
            record['actual_sha256'] = digest(p)
            ok = record['actual_sha256'] == check['expected_sha256']
        else:
            ref = check['reference']
            ref_path = local(root, ref['path'])
            if digest(ref_path) != ref['sha256']:
                raise ValueError('reference changed: ' + ref['path'])
            dtype = {'float32': '<f4', 'int32': '<i4', 'bf16': '<u2'}[check['dtype']]
            got = np.fromfile(p, dtype=dtype)
            expected = np.fromfile(ref_path, dtype=dtype)
            ok = got.shape == expected.shape
            if ok and mode == 'exact':
                ok = np.array_equal(got, expected)
            elif ok and mode == 'close':
                # Same FP32 operations/tolerances as the qualification checker.
                delta = np.abs(got - expected)
                good = np.isfinite(got) & (delta <= np.float32(check['atol']) +
                        np.float32(check['rtol']) * np.abs(expected))
                ok = bool(np.all(good))
                record.update(failed_elements=int(np.count_nonzero(~good)),
                              max_abs_error=float(np.max(delta)))
            elif ok and mode == 'gemm':
                # Unchanged complete-output nightly GEMM acceptance.
                g = got.astype(np.uint32) << 16
                e = expected.astype(np.uint32) << 16
                gf = g.view(np.float32).astype(np.float64)
                ef = e.view(np.float32).astype(np.float64)
                same = got == expected
                one = ((got >> 15) == (expected >> 15)) & (
                    np.abs((got & 0x7fff).astype(np.int32) - (expected & 0x7fff).astype(np.int32)) == 1)
                tiny = ((got | expected) & 0x7fff) <= 0x80
                bad = int(np.count_nonzero(~(same | one | tiny)))
                finite = bool(np.all(np.isfinite(gf)))
                poison = int(np.count_nonzero(got == 0xeeee))
                se, sr = float(np.sum((gf - ef) ** 2)), float(np.sum(ef ** 2))
                error = math.sqrt(se / sr) if sr else (math.inf if se else 0.)
                ok = finite and not poison and error <= .01 and bad <= len(got) // 100
                record.update(bad=bad, poison=poison, relative_frobenius_error=error)
        record['passed'] = bool(ok)
        outcomes.append(record)
    reports = []
    for spec in row.get('smem_reports', []):
        p = local(results, spec['file'])
        ok = p.is_file() and p.stat().st_size == 64
        if ok:
            words = struct.unpack('<16I', p.read_bytes())
            v = words[::2]
            ok = (words[::2] == words[1::2] and v[0] == 0x5ef1d0e5 and
                  v[1] == 512 * 512 and v[4] <= v[1] // 100 and
                  v[5] == 0 and v[6] == 0 and v[7] <= 100)
            reports.append(v)
        outcomes.append(dict(file=spec['file'], passed=bool(ok)))
    if reports:
        outcomes.append(dict(check='replica_mailbox_agreement',
                             passed=all(v[2:] == reports[0][2:] for v in reports)))
    return dict(workload=name, passed=bool(outcomes) and all(x['passed'] for x in outcomes),
                scope='complete requested output readbacks; execution provenance/termination must be retained separately',
                checks=outcomes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build'); b.add_argument('--recipe', type=Path, required=True)
    b.add_argument('--output', type=Path, required=True)
    sub.add_parser('verify'); sub.add_parser('list')
    s = sub.add_parser('prepare'); s.add_argument('name'); s.add_argument('--output', type=Path, required=True)
    s.add_argument('--loader', choices=['logical', 'direct-hashed'], required=True)
    s.add_argument('--hash-size', type=lambda x: int(x, 0), default=0x80000000)
    s.add_argument('--hash-unit', type=int, default=32); s.add_argument('--hash-slices', type=int, default=4)
    c = sub.add_parser('check'); c.add_argument('name'); c.add_argument('--results', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'build':
        m = build(args.recipe, args.output); result = dict(workloads=len(m['workloads']))
    elif args.command == 'verify': result = verify(args.root)
    elif args.command == 'list':
        for row in load_manifest(args.root)['workloads']:
            print(row['name'], f"SMs={row['num_clusters']}", f"partitions={row['logical_partitions']}")
        return
    elif args.command == 'prepare':
        result = prepare(args.root, args.name, args.output, args.loader,
                         args.hash_size, args.hash_unit, args.hash_slices)
    else:
        result = check_outputs(args.root, args.name, args.results)
    print(json.dumps(result, indent=2))
    if result.get('passed') is False: sys.exit(1)


if __name__ == '__main__':
    main()
