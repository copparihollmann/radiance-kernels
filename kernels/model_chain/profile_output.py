#!/usr/bin/env python3
"""Parse committed RADIANCE_PROFILE records without guessing missing counters."""
import argparse
import csv
import json
from pathlib import Path
import re

PREFIX = 'RADIANCE_PROFILE '
UINT64 = 1 << 64


def integer(value, name, limit=UINT64):
    if type(value) is not int or not 0 <= value < limit:
        raise ValueError('invalid ' + name)
    return value


def summarize(lines, context):
    if not isinstance(context, dict) or type(context.get('schema')) is not int or context['schema'] != 1:
        raise ValueError('unsupported context schema')
    sms = integer(context['sm_count'], 'sm_count', 5)
    cores = integer(context['cores_per_sm'], 'cores_per_sm', 3)
    if sms not in (1, 2, 4) or cores != 2:
        raise ValueError('expected 1/2/4 SMs with two cores per SM')
    for key in ('workload', 'backend', 'memory_model', 'clock_domain'):
        if not isinstance(context.get(key), str) or not context[key]:
            raise ValueError('missing context ' + key)
    if not re.fullmatch(r'[0-9a-f]{64}', context.get('device_elf_sha256', '')):
        raise ValueError('missing device ELF hash')
    common_epoch = context.get('common_cycle_epoch')
    if type(common_epoch) is not bool:
        raise ValueError('common_cycle_epoch must be explicit')
    semantics = context.get('instruction_counter_semantics')
    if semantics not in ('issued_warp_instructions', 'retired_warp_instructions', 'unavailable'):
        raise ValueError('instruction counter semantics must be explicit')
    peak = context.get('peak_issue_per_core_per_cycle')
    if peak is not None and (type(peak) not in (int, float) or not 0 < peak < float('inf')):
        raise ValueError('invalid peak issue capacity')
    if peak is not None and semantics != 'issued_warp_instructions':
        raise ValueError('issue utilization requires an issued-instruction counter')

    records = {}
    for line in lines:
        if not line.startswith(PREFIX):
            continue
        try:
            raw = json.loads(line[len(PREFIX):])
        except json.JSONDecodeError as error:
            raise ValueError('truncated or invalid profile record') from error
        if not isinstance(raw, dict) or type(raw.get('schema')) is not int or raw['schema'] != 1:
            raise ValueError('unsupported profile schema')
        sm = integer(raw['sm'], 'sm', sms)
        core = integer(raw['core'], 'core', cores)
        key = sm, core
        if key in records:
            raise ValueError('duplicate profile record (possibly concatenated runs)')
        start = integer(raw['cycle_begin'], 'cycle_begin')
        end = integer(raw['cycle_end'], 'cycle_end')
        if end <= start:
            raise ValueError('nonpositive interval or counter reset/wrap; cannot aggregate epochs')
        ia, ib = raw['instructions_begin'], raw['instructions_end']
        if semantics == 'unavailable':
            if ia is not None or ib is not None:
                raise ValueError('unqualified instruction counter must be null')
            instructions = None
        else:
            ia, ib = integer(ia, 'instructions_begin'), integer(ib, 'instructions_end')
            if ib < ia:
                raise ValueError('instruction counter reset or wrap')
            instructions = ib - ia
        cycles = end - start
        records[key] = dict(sm=sm, core=core, cycle_begin=start, cycle_end=end,
            work_cycles=cycles, instructions=instructions,
            ipc=None if instructions is None else instructions / cycles,
            simt_issue_utilization=None if peak is None else instructions / (cycles * peak))
    if len(records) != sms * cores:
        raise ValueError(f'missing profile records: got {len(records)}, expected {sms * cores}')
    per_core = [records[key] for key in sorted(records)]
    core_cycles = sum(row['work_cycles'] for row in per_core)
    instructions = None if semantics == 'unavailable' else sum(row['instructions'] for row in per_core)
    cycles = max(row['cycle_end'] for row in per_core) - min(row['cycle_begin'] for row in per_core) if common_epoch else None
    return dict(schema='radiance-profile-summary-v1', context=context, per_core=per_core,
        total=dict(work_cycles=cycles, sum_core_cycles=core_cycles,
            max_core_duration_cycles=max(row['work_cycles'] for row in per_core),
            instructions=instructions,
            ipc=None if instructions is None or cycles is None else instructions / cycles,
            mean_core_ipc=None if instructions is None else instructions / core_cycles,
            simt_issue_utilization=None if peak is None or cycles is None else instructions / (cycles * sms * cores * peak),
            mx_utilization=None, vpu_utilization=None, lane_efficiency=None,
            whole_radiance_utilization=None),
        limitations=[
            'CSR sampling is optional instrumentation and has nonzero target instruction overhead.',
            'Cycle and instruction samples are not simultaneous; no overhead has been subtracted.',
            'ELF/backend/memory identity comes from the supplied run context; the console does not attest it.',
            'MX, VPU, lane efficiency and whole-chip FLOP utilization need separately qualified counters and peaks.',
            'Functional or assisted-core cycle counts are not timing results.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    try:
        with args.log.open() as lines:
            result = summarize(lines, json.loads(args.context.read_text()))
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if args.csv:
        with args.csv.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result['per_core'][0]))
            writer.writeheader()
            writer.writerows(result['per_core'])
    print(json.dumps(result['total'], allow_nan=False))


if __name__ == '__main__':
    main()
