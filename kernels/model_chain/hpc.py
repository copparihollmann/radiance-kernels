#!/usr/bin/env python3
"""List HPC workloads and collect saved logs/readbacks. Does not run FireSim."""
import argparse
import csv
import json
from pathlib import Path
import sys

from workloads import select
from profile_output import PREFIX, summarize
import workload_bundle as bundle


def collect(root, selected, logs, output):
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for workload, meta in selected:
        name = meta['name']
        directory = logs / name
        console = directory / 'console.log'
        context = directory / 'run-context.json'
        report = dict(workload=name, sm_count=meta['sm_count'], benchmark=meta['benchmark'],
            correctness='not_checked', profile_status='unavailable',
            work_cycles=None, ipc=None, simt_issue_utilization=None,
            mx_utilization=None, vpu_utilization=None, lane_efficiency=None,
            software_model_cycles=workload.get('timing_qualification', {}).get('simulated_cycles'),
            reason='No RADIANCE_PROFILE records; existing release ELFs do not emit them.')
        if console.is_file():
            report['console_sha256'] = bundle.digest(console)
            with console.open() as stream:
                records = [line for line in stream if line.startswith(PREFIX)]
            if records:
                try:
                    ctx = json.loads(context.read_text())
                    if (ctx['workload'] != name or ctx['sm_count'] != meta['sm_count'] or
                            ctx['device_elf_sha256'] != workload['device_elf']['sha256']):
                        raise ValueError('run context does not match selected workload/ELF/SM count')
                    result = summarize(records, ctx)
                    bundle.write_json(output / (name + '-profile.json'), result)
                    report.update({key: result['total'][key] for key in
                        ['work_cycles', 'ipc', 'simt_issue_utilization', 'mx_utilization',
                         'vpu_utilization', 'lane_efficiency']})
                    report.update(profile_status='parsed', reason=None)
                except (OSError, KeyError, ValueError) as error:
                    report.update(profile_status='invalid', reason=str(error))
        else:
            report.update(profile_status='missing_log', reason='Missing ' + str(console))
        if (directory / 'readbacks').is_dir():
            try:
                checked = bundle.check_outputs(root, name, directory / 'readbacks')
                report['correctness'] = 'passed' if checked['passed'] else 'failed'
                bundle.write_json(output / (name + '-check.json'), checked)
            except (OSError, ValueError, KeyError) as error:
                report.update(correctness='failed', check_error=str(error))
        reports.append(report)
    bundle.write_json(output / 'results.json', dict(schema='radiance-hpc-results-v1',
        manifest_sha256=bundle.digest(root / 'manifest.json'), rows=reports,
        note='Software model cycles are separate from measured run work cycles. Null means unavailable.'))
    fields = ['workload', 'sm_count', 'benchmark', 'correctness', 'profile_status',
              'work_cycles', 'ipc', 'simt_issue_utilization', 'mx_utilization',
              'vpu_utilization', 'lane_efficiency', 'software_model_cycles', 'reason']
    with (output / 'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(reports)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['list', 'collect']:
        cmd = commands.add_parser(name)
        cmd.add_argument('--sms', type=int, nargs='+', choices=[1, 2, 4])
        cmd.add_argument('--benchmark', action='append', choices=['stream', 'spatter', 'xrage', 'fp8_gemm'])
        cmd.add_argument('--include-controls', action='store_true')
        if name == 'collect':
            cmd.add_argument('--logs', type=Path, required=True)
            cmd.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = bundle.load_manifest(args.root)
    selected = select(manifest, kinds=['hpc'], sms=args.sms, benchmarks=args.benchmark)
    if not args.include_controls:
        selected = [(row, meta) for row, meta in selected
            if not (meta['sm_count'] == 1 and meta['logical_partitions'] == 2)]
    if not selected:
        parser.error('no matching HPC workloads')
    if args.command == 'list':
        print(json.dumps([dict(**meta, run_group=row.get('run_group', 'primary'))
                          for row, meta in selected], indent=2))
    else:
        rows = collect(args.root, selected, args.logs, args.output)
        print(json.dumps(dict(workloads=len(rows), profiles=sum(r['profile_status']=='parsed' for r in rows),
            correctness_passes=sum(r['correctness']=='passed' for r in rows), output=str(args.output))))
        if any(r['profile_status']=='invalid' or r['correctness']=='failed' for r in rows):
            sys.exit(1)


if __name__ == '__main__':
    main()
