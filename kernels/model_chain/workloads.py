#!/usr/bin/env python3
"""Select portable workloads by graph scope and physical SM count."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


KINDS = ('full_model', 'layer', 'hpc', 'control')


def describe(row):
    """Use explicit metadata; support the original portable bundle as well."""
    name = row['name']
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', name) or name in ('.', '..'):
        raise ValueError('unsafe workload name: ' + name)
    kind = row.get('kind')
    if kind is None:
        kind = {'model': 'full_model', 'hpc': 'hpc', 'gemm': 'hpc'}.get(row.get('family'))
    if kind not in KINDS:
        raise ValueError('missing or unknown workload kind: ' + name)
    sms = row['num_clusters']
    if type(sms) is not int or sms <= 0 or row.get('sm_count', sms) != sms:
        raise ValueError('invalid or inconsistent physical SM count: ' + name)
    benchmark = row.get('benchmark')
    if benchmark is None and kind == 'hpc':
        benchmark = ('fp8_gemm' if row.get('family') == 'gemm' else
                     'xrage' if name.startswith('xrage-') else name.split('--', 1)[0])
    model = row.get('model')
    if model is None and kind == 'full_model':
        model = re.sub(r'-\d+sm$', '', name)
    return dict(name=name, kind=kind, sm_count=sms,
                logical_partitions=row.get('logical_partitions'),
                model=model, benchmark=benchmark,
                software_status=row.get('software_qualification', {}).get('status', 'pending'),
                timing_status=row.get('timing_qualification', {}).get('status', 'pending'),
                fpga_execution_verified=row.get('fpga_execution_verified', False))


def select(manifest, *, kinds=None, sms=None, models=None, benchmarks=None, names=None,
           status='any'):
    rows = manifest['workloads']
    metadata = [describe(row) for row in rows]
    available = {item['name'] for item in metadata}
    if len(available) != len(rows):
        raise ValueError('duplicate workload names')
    if names and set(names) - available:
        raise ValueError('unknown workloads: ' + ', '.join(sorted(set(names) - available)))
    result = []
    for row, item in zip(rows, metadata):
        if ((kinds and item['kind'] not in kinds) or
                (sms and item['sm_count'] not in sms) or
                (models and item['model'] not in models) or
                (benchmarks and item['benchmark'] not in benchmarks) or
                (names and item['name'] not in names) or
                (status != 'any' and item['software_status'] != status)):
            continue
        result.append((row, item))
    return result


def run_plan(root, selected, run_root, loader, python=sys.executable):
    """Emit argument arrays for preparation/checking; do not claim a board runner."""
    root, run_root = root.resolve(), run_root.resolve()
    helper = str(root / 'workload_bundle.py')
    jobs = []
    for row, item in selected:
        directory = run_root / item['name']
        prepare = [python, helper, '--root', str(root), 'prepare', item['name'],
                   '--loader', loader, '--output', str(directory)]
        if loader == 'direct-hashed':
            prepare += ['--hash-size', '0x80000000', '--hash-unit', '32', '--hash-slices', '4']
        jobs.append(dict(**item,
            device_elf=row['device_elf'], debug_elf=row.get('debug_elf'),
            soc_elf=row['soc_elf'], loads=row['loads'], checks=row['checks'],
            smem_init=row.get('smem_init', []),
            smem_init_per_cluster=row.get('smem_init_per_cluster', []),
            completion_contract=row.get('completion_contract'),
            prepare_argv=prepare, boot_elf=str(directory / 'boot.elf'),
            firesim_workload_json=str(directory / 'firesim-workload.json'),
            dump_plan=str(directory / 'dump-plan.json'),
            check_argv=[python, helper, '--root', str(root), 'check', item['name'],
                        '--results', str(directory / 'readbacks')],
            board_run_argv=None))
    return dict(schema='radiance-workload-plan-v1', bundle_root=str(root),
                manifest=str(root / 'manifest.json'), selected_count=len(jobs), jobs=jobs,
                board_runner='Supply the matching HWDB/runtime and completion/readback adapter.',
                physical_sms_field='sm_count; logical_partitions does not select hardware')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('list', 'plan'):
        p = commands.add_parser(command)
        p.add_argument('--kind', choices=KINDS, action='append')
        p.add_argument('--sms', type=int, nargs='+')
        p.add_argument('--model', action='append')
        p.add_argument('--benchmark', action='append')
        p.add_argument('--name', action='append')
        p.add_argument('--status', choices=('any', 'passed', 'pending', 'failed'),
                       default='passed' if command == 'plan' else 'any')
        if command == 'list':
            p.add_argument('--format', choices=('table', 'json', 'names'), default='table')
        else:
            p.add_argument('--run-root', type=Path, default=Path('runs'))
            p.add_argument('--loader', choices=('logical', 'direct-hashed'), required=True)
    p = commands.add_parser('show')
    p.add_argument('name')
    args = parser.parse_args()
    raw = (args.root / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    try:
        if args.command == 'show':
            row, _ = select(manifest, names=[args.name])[0]
            print(json.dumps(row, indent=2))
            return
        selected = select(manifest, kinds=args.kind, sms=args.sms, models=args.model,
                          benchmarks=args.benchmark, names=args.name, status=args.status)
        if args.command == 'plan':
            if not selected:
                raise ValueError('no workloads match the selection')
            result = run_plan(args.root, selected, args.run_root, args.loader)
            result['manifest_sha256'] = hashlib.sha256(raw).hexdigest()
            print(json.dumps(result, indent=2))
        elif args.format == 'json':
            print(json.dumps([item for _, item in selected], indent=2))
        elif args.format == 'names':
            for _, item in selected:
                print(item['name'])
        else:
            print(f'{"WORKLOAD":45} {"KIND":12} {"SMs":>3} {"PARTITIONS":>10} SOFTWARE')
            for _, item in selected:
                partitions = item['logical_partitions']
                print(f'{item["name"]:45} {item["kind"]:12} {item["sm_count"]:>3} '
                      f'{str(partitions) if partitions is not None else "-":>10} '
                      f'{item["software_status"]}')
    except (KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
