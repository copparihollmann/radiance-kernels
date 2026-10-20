#!/usr/bin/env python3
"""List HPC workloads and collect saved logs/readbacks. Does not run FireSim."""
import argparse
import csv
import json
from pathlib import Path
import sys

from workloads import select
from profile_output import PREFIX, parse_log, add_lane_trace
import workload_bundle as bundle


def collect(root, selected, logs, output, *, mx_clock_ratio=None, gpu_clock_hz=None):
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for workload, meta in selected:
        name = meta['name']
        directory = logs / name
        console = directory / 'console.log'
        context = directory / 'run-context.json'
        report = dict(workload=name, kind=meta['kind'], model=meta.get('model'), sm_count=meta['sm_count'], benchmark=meta['benchmark'],
            correctness='not_checked', profile_status='unavailable',
            work_cycles=None, instructions=None, ipc=None, simt_issue_utilization=None,
            mx_utilization=None, vpu_utilization=None, lane_efficiency=None,
            software_model_cycles=workload.get('timing_qualification', {}).get('simulated_cycles'),
            reason='No RADIANCE_PROFILE records in the saved log.')
        if console.is_file():
            report['console_sha256'] = bundle.digest(console)
            with console.open() as stream:
                records = [line for line in stream if line.startswith((PREFIX, 'RADIANCE_CONTEXT ',
                    'RADIANCE_FAILURE ', 'RADIANCE_TRAP '))]
            if records:
                try:
                    ctx = json.loads(context.read_text()) if context.exists() else None
                    with console.open() as stream:
                        result = parse_log(stream, ctx, mx_clock_ratio=mx_clock_ratio, gpu_clock_hz=gpu_clock_hz,
                            useful_payload_bytes=workload.get('profiling', {}).get('useful_payload_bytes'))
                    ctx = result['context']
                    if (ctx['workload'] != name or ctx['sm_count'] != meta['sm_count'] or
                            ctx['device_elf_sha256'] != workload['device_elf']['sha256']):
                        raise ValueError('run context does not match selected workload/ELF/SM count')
                    for trace in [directory / 'issue-trace.csv.gz', directory / 'issue-trace.csv']:
                        if trace.is_file():
                            add_lane_trace(result, trace)
                            break
                    bundle.write_json(output / (name + '-profile.json'), result)
                    report.update({key: result['total'][key] for key in
                        ['work_cycles', 'instructions', 'ipc', 'simt_issue_utilization', 'mx_utilization',
                         'vpu_utilization', 'lane_efficiency']})
                    report.update(profile_status='parsed' if result['complete_run_log'] else 'partial',
                                  reason=None if result['complete_run_log'] else 'Work records present; final RADIANCE_DONE is missing.')
                    report['host'] = result['host']
                    if result['host']:
                        for phase in ['setup', 'wait']:
                            for metric in ['cycles', 'ipc']:
                                report['host_'+phase+'_'+metric] = result['host'][phase+'_'+metric]
                    report['memory'] = result.get('memory')
                    if result.get('memory'):
                        report['memory_read_bytes'] = result['memory']['read_bytes']
                        report['memory_write_bytes'] = result['memory']['write_bytes']
                    report['memory_bandwidth_bytes_per_second'] = result['total'].get('memory_bandwidth_bytes_per_second')
                    report['useful_payload_bytes_per_cycle'] = result['total'].get('useful_payload_bytes_per_cycle')
                    report['useful_bandwidth_bytes_per_second'] = result['total'].get('useful_bandwidth_bytes_per_second')
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
    bundle.write_json(output / 'results.json', dict(schema='radiance-workload-results-v1',
        manifest_sha256=bundle.digest(root / 'manifest.json'), rows=reports,
        note='Software model cycles are separate from measured run work cycles. Null means unavailable.'))
    fields = ['workload', 'kind', 'model', 'sm_count', 'benchmark', 'correctness', 'profile_status',
              'work_cycles', 'instructions', 'ipc', 'simt_issue_utilization', 'mx_utilization',
              'vpu_utilization', 'lane_efficiency', 'memory_read_bytes', 'memory_write_bytes', 'useful_payload_bytes_per_cycle',
              'useful_bandwidth_bytes_per_second', 'memory_bandwidth_bytes_per_second',
              'host_setup_cycles', 'host_wait_cycles', 'host_setup_ipc', 'host_wait_ipc', 'software_model_cycles', 'reason']
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
            cmd.add_argument('--mx-clock-ratio', type=float)
            cmd.add_argument('--gpu-clock-hz', type=float)
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
        rows = collect(args.root, selected, args.logs, args.output,
                       mx_clock_ratio=args.mx_clock_ratio, gpu_clock_hz=args.gpu_clock_hz)
        print(json.dumps(dict(workloads=len(rows), profiles=sum(r['profile_status']=='parsed' for r in rows),
            correctness_passes=sum(r['correctness']=='passed' for r in rows), output=str(args.output))))
        if any(r['profile_status']=='invalid' or r['correctness']=='failed' for r in rows):
            sys.exit(1)


if __name__ == '__main__':
    main()
