#!/usr/bin/env python3
"""Parse committed RADIANCE_PROFILE records without guessing missing counters."""
import argparse
import csv
import gzip
import json
import math
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
            if instructions == 0:
                raise ValueError('instruction counter did not advance; do not report zero IPC for this run')
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
    result = dict(schema='radiance-profile-summary-v1', context=context, per_core=per_core,
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
    macs = context.get('mx_useful_macs')
    if macs is not None:
        integer(macs, 'mx_useful_macs')
        if macs == 0:
            result['total']['mx_utilization'] = 0.0
        elif cycles is not None and context.get('mx_cycles_per_muon_cycle') is not None:
            ratio = context['mx_cycles_per_muon_cycle']
            mx_peak = context.get('mx_peak_macs_per_cycle')
            instances = context.get('mx_instances')
            if (type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0 or
                    type(mx_peak) is not int or mx_peak <= 0 or type(instances) is not int or instances != sms):
                raise ValueError('invalid MX peak, instance count or clock ratio')
            result['total']['mx_utilization'] = macs / (cycles * ratio * mx_peak * instances)
    if macs and result['total']['mx_utilization'] is None:
        result['limitations'].append('Supply the actual MX clock / Muon clock ratio to compute MX utilization.')
    return result


def parse_log(lines, context=None, *, mx_clock_ratio=None, gpu_clock_hz=None, useful_payload_bytes=None):
    """Auto-read host metadata while retaining a separate operator context override."""
    raw, embedded, host, completion = [], [], [], []
    failures, memory, engines = [], [], []
    for line in lines:
        if line.startswith(PREFIX):
            raw.append(line)
        for prefix, destination in [('RADIANCE_CONTEXT ', embedded), ('RADIANCE_HOST ', host),
                                    ('RADIANCE_DONE ', completion), ('RADIANCE_FAILURE ', failures),
                                    ('RADIANCE_TRAP ', failures), ('RADIANCE_MEMORY ', memory),
                                    ('RADIANCE_ENGINE ', engines)]:
            if line.startswith(prefix):
                try:
                    value = json.loads(line[len(prefix):])
                except json.JSONDecodeError as error:
                    raise ValueError('truncated host context/status record') from error
                if not isinstance(value, dict):
                    raise ValueError('host context/status must be an object')
                destination.append(value)
    if failures:
        raise ValueError('device/host reported a failure: ' + str(failures))
    if len(embedded) > 1 or len(host) > 1 or len(completion) > 1:
        raise ValueError('duplicate host metadata; possibly concatenated runs')
    if context is None:
        if len(embedded) != 1:
            raise ValueError('missing RADIANCE_CONTEXT; supply --context for older raw snapshot logs')
        context = embedded[0]
    elif embedded:
        for key in ['workload', 'device_elf_sha256', 'sm_count', 'cores_per_sm']:
            if context[key] != embedded[0][key]:
                raise ValueError('operator context and emitted identity disagree: ' + key)
        for key in ['mx_useful_macs', 'mx_instances', 'mx_peak_macs_per_cycle',
                    'peak_issue_per_core_per_cycle', 'instruction_counter_semantics']:
            if key in context and key in embedded[0] and context[key] != embedded[0][key]:
                raise ValueError('operator context changes a compiled workload/counter property: '+key)
        context = embedded[0] | context
    context = dict(context)
    if mx_clock_ratio is not None:
        if not math.isfinite(mx_clock_ratio) or mx_clock_ratio <= 0:
            raise ValueError('MX clock ratio must be finite and positive')
        context['mx_cycles_per_muon_cycle'] = mx_clock_ratio
    result = summarize(raw, context)
    result['host'] = host[0] if host else None
    if host:
        for phase in ['setup', 'wait']:
            cycles = integer(host[0][phase+'_cycles'], 'host '+phase+' cycles')
            instructions = integer(host[0][phase+'_instructions'], 'host '+phase+' instructions')
            host[0][phase+'_ipc'] = instructions / cycles if cycles else None
    result['device_completion'] = completion[0] if completion else None
    result['complete_run_log'] = bool(completion)
    if len(memory) > 1:
        raise ValueError('expected one aggregate work-window memory record')
    result['memory'] = memory[0] if memory else None
    result['total']['memory_bandwidth_bytes_per_second'] = None
    result['total']['useful_payload_bytes_per_cycle'] = None
    result['total']['useful_bandwidth_bytes_per_second'] = None
    if useful_payload_bytes is not None:
        integer(useful_payload_bytes, 'useful_payload_bytes')
        context['useful_payload_bytes'] = useful_payload_bytes
        if result['total']['work_cycles'] is not None:
            rate = useful_payload_bytes / result['total']['work_cycles']
            result['total']['useful_payload_bytes_per_cycle'] = rate
            if gpu_clock_hz is not None:
                if not math.isfinite(gpu_clock_hz) or gpu_clock_hz <= 0:
                    raise ValueError('GPU clock frequency must be finite and positive')
                result['total']['useful_bandwidth_bytes_per_second'] = rate * gpu_clock_hz
    if memory:
        record = memory[0]
        if record.get('scope') != 'work' or not record.get('source'):
            raise ValueError('memory counters require work-window scope and source')
        for field in ['read_bytes', 'write_bytes']: integer(record[field], field)
        if gpu_clock_hz is not None:
            if not math.isfinite(gpu_clock_hz) or gpu_clock_hz <= 0:
                raise ValueError('GPU clock frequency must be finite and positive')
            if result['total']['work_cycles'] is not None:
                result['total']['memory_bandwidth_bytes_per_second'] = (record['read_bytes']+record['write_bytes'])*gpu_clock_hz/result['total']['work_cycles']
    for record in engines:
        if record.get('engine') != 'vpu' or record.get('scope') != 'work' or not record.get('source'):
            raise ValueError('unsupported engine counter record')
        busy = integer(record['busy_cycles'], 'busy_cycles')
        capacity = integer(record['capacity_cycles'], 'capacity_cycles')
        if not capacity or busy > capacity or result['total']['vpu_utilization'] is not None:
            raise ValueError('invalid or duplicate aggregate VPU counters')
        result['total']['vpu_utilization'] = busy / capacity
    if completion:
        done = completion[0]
        if done.get('workload') != context['workload'] or done.get('sms') != context['sm_count'] or done.get('device_code') != 0:
            raise ValueError('completion record does not match successful workload identity')
    return result


def add_lane_trace(result, path):
    """Read a complete issued-warp CSV trace, restricted to each core's work window.

    Required columns: cycle,sm,core,tmask (decimal integers or 0x hex). One row per
    accepted warp instruction, including stores; this is not a per-lane trace.
    """
    windows = {(r['sm'], r['core']): r for r in result['per_core']}
    counts = {key: [0, 0] for key in windows}
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        reader = csv.DictReader(stream)
        if not {'cycle', 'sm', 'core', 'tmask'} <= set(reader.fieldnames or []):
            raise ValueError('issue trace needs cycle,sm,core,tmask columns')
        for row in reader:
            cycle, sm, core, mask = [int(row[k], 0) for k in ['cycle', 'sm', 'core', 'tmask']]
            integer(cycle, 'trace cycle'); integer(mask, 'trace lane mask', 1 << 16)
            if (sm, core) not in windows:
                raise ValueError('trace topology differs from selected workload')
            window = windows[sm, core]
            if window['cycle_begin'] <= cycle < window['cycle_end']:
                counts[sm, core][0] += 1
                counts[sm, core][1] += mask.bit_count()
    if any(n == 0 for n, active in counts.values()):
        raise ValueError('complete issue trace is missing a participating core')
    for key, window in windows.items():
        n, active = counts[key]
        window.update(trace_warp_instructions=n, trace_active_lanes=active, lane_efficiency=active / (16*n))
    n = sum(v[0] for v in counts.values()); active = sum(v[1] for v in counts.values())
    result['total']['lane_efficiency'] = active / (16*n)
    result['trace_scope'] = 'Operator-supplied complete issued-warp trace, filtered to per-core measured work windows.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--context', type=Path)
    parser.add_argument('--mx-clock-ratio', type=float, help='Actual MX clock frequency / Muon frequency (e.g. 1)')
    parser.add_argument('--issue-trace', type=Path, help='Complete accepted-warp CSV trace, optionally gzip-compressed')
    parser.add_argument('--gpu-clock-hz', type=float, help='Actual Muon frequency for memory bandwidth conversion')
    parser.add_argument('--useful-payload-bytes', type=int, help='Known benchmark logical payload bytes, not physical DRAM traffic')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    try:
        with args.log.open() as lines:
            result = parse_log(lines, json.loads(args.context.read_text()) if args.context else None,
                               mx_clock_ratio=args.mx_clock_ratio, gpu_clock_hz=args.gpu_clock_hz,
                               useful_payload_bytes=args.useful_payload_bytes)
        if args.issue_trace:
            add_lane_trace(result, args.issue_trace)
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
