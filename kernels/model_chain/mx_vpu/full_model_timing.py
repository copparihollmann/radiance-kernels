#!/usr/bin/env python3
"""Prepare, run and summarize unassisted full-model timing from a qualified bundle.

Runs one case at a time. Existing attempts are immutable; passing cases are
rechecked on resume, and an interrupted/failed case gets a new attempt directory.
No helper-assisted ticks or partial run can pass as full-model latency.
"""
import argparse
from collections import defaultdict
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
import tomllib

from run_e2e_queue import atomic_json, digest, validate

CHAIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHAIN))
from workload_bundle import local


def toml_records(name, records):
    return name + ' = [' + ', '.join('{' + ', '.join(
        key + '=' + json.dumps(value) for key, value in row.items()) + '}'
        for row in records) + ']\n'


def prepare(bundle, cyclotron, simulator, output, cycle_limit, source_ticks, memory_ticks):
    bundle, cyclotron, simulator, output = [p.resolve() for p in
        (bundle, cyclotron, simulator, output)]
    if min(cycle_limit, source_ticks, memory_ticks) <= 0:
        raise ValueError('cycle limit and clock periods must be positive')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    rows = [row for row in manifest['workloads'] if row['family'] == 'model']
    if len(rows) != 8 or {row['num_clusters'] for row in rows} != {1, 2}:
        raise ValueError('expected all four models at one and two SMs')
    output.mkdir(parents=True, exist_ok=False)
    configurations = [cyclotron / 'config/timing' / name for name in [
        'radiance/gmem.toml', 'radiance/icache.toml', 'lsu_scheduler.toml',
        'smem.toml', 'writeback_operand_fetch.toml', 'barrier.toml',
        'fence.toml', 'dma_tensor.toml', 'radiance/execute.toml', 'radiance/hazards.toml']]
    hbm = cyclotron / 'configs/hbm/hbm2-4ch.yaml'
    pins = {str(p): digest(p) for p in configurations + [hbm]}
    metadata = {'schema': 'radiance-full-model-timing-v1', 'created': time.time(),
        'bundle': str(bundle), 'manifest_sha256': digest(bundle / 'manifest.json'),
        'simulator': str(simulator), 'simulator_sha256': digest(simulator),
        'cycle_limit': cycle_limit, 'source_ticks': source_ticks,
        'memory_ticks': memory_ticks, 'hardware_calibrated': False, 'cases': []}
    summary_ref = manifest['evidence']['all_full_models']
    summary_path = local(bundle, summary_ref['path'])
    if digest(summary_path) != summary_ref['sha256']:
        raise ValueError('full-model qualification summary changed')
    qualified_models = {model['model']: model for model in
        json.loads(summary_path.read_text())['models']}
    seen = set()
    for row in rows:
        refs = [row['device_elf'], *row['loads'], row['sources']['manifest'],
                row['sources']['qualified_simulator_configuration']]
        refs += [check['reference'] for check in row['checks'] if 'reference' in check]
        for ref in refs:
            if ref['path'] not in seen:
                path = local(bundle, ref['path'])
                if path.stat().st_size != ref['bytes'] or digest(path) != ref['sha256']:
                    raise ValueError('bundle input changed: ' + ref['path'])
                seen.add(ref['path'])
        model = json.loads(local(bundle, row['sources']['manifest']['path']).read_text())
        qualified = tomllib.loads(local(bundle,
            row['sources']['qualified_simulator_configuration']['path']).read_text())
        topology = qualified['muon']
        case = output / row['name']
        case.mkdir()
        # A fresh configuration prevents inheritance of assistance, old output
        # paths, stale patches, tracing, or accidentally disabled core timing.
        prefix = '[muon]\n' + ''.join(key + '=' + json.dumps(value) + '\n'
            for key, value in topology.items())
        prefix += '\n[mem]\n' + ''.join(key + '=' + json.dumps(value) + '\n'
            for key, value in qualified.get('mem', {}).items())
        prefix += '\n[sim]\ntiming=true\ntrace=false\n'
        prefix += f'num_clusters={row["num_clusters"]}\ntimeout={cycle_limit}\n'
        prefix += 'progress_every_cycles=10000000\n'
        prefix += 'smem_init=' + json.dumps(row['smem_init']) + '\n'
        prefix += 'smem_dump=[]\nnightly_host_handshake=false\n'
        prefix += toml_records('gmem_load', [dict(address=load['address'],
            bytes=load['bytes'], path=str(local(bundle, load['path']))) for load in row['loads']])
        prefix += '\n[gemmini]\nenabled=true\nmode="both"\npreset="radiance"\n'
        prefix += '\n[timing]\ninclude=' + json.dumps(list(map(str, configurations))) + '\n'
        prefix += '\n[timing.gmem.hbm]\nconfig=' + json.dumps(str(hbm)) + '\ntrace=""\n'
        prefix += f'source_ticks={source_ticks}\nmemory_ticks={memory_ticks}\n'
        prefix += 'report="hbm.json"\n'
        (case / 'config-template.toml').write_text(prefix)
        atomic_json(case / 'workload.json', row)
        atomic_json(case / 'model.json', model)
        counts = qualified_models[model['model']]
        metadata['cases'].append(dict(name=row['name'], num_clusters=row['num_clusters'],
            expected_stages=model['stages'], stages=model['stage_backends'],
            expected_stage_order=list(range(model['stages'])),
            expected_gemmini=dict(mx_loops=counts['mx_loops_per_sm'],
                mesh_tiles=counts['mesh_tiles_per_sm'], vpu_commands=counts['vpu_commands_per_sm']),
            prepared_timing_configuration_sha256=pins))
    atomic_json(output / 'plan.json', metadata)
    return metadata


def make_job(root, plan, case, attempt):
    source = root / case['name']
    row = json.loads((source / 'workload.json').read_text())
    bundle = Path(plan['bundle'])
    reads = {check['file']: check for check in row['checks']}
    for name in reads:
        local(attempt, name).parent.mkdir(parents=True, exist_ok=True)
    config = (source / 'config-template.toml').read_text()
    config = config.replace('[sim]\n', '[sim]\nprogress="progress.jsonl"\n' +
        toml_records('gmem_dump', [dict(address=check['address'], bytes=check['bytes'],
            path=str(local(attempt, name))) for name, check in reads.items()]))
    config = config.replace('report="hbm.json"', 'report=' + json.dumps(str(attempt / 'hbm.json')))
    path = attempt / 'cyclotron.toml'
    path.write_text(config)
    inputs = [row['device_elf'], *row['loads']]
    inputs += [check['reference'] for check in row['checks'] if 'reference' in check]
    return dict(name=case['name'], kind='isa', execution_mode='coupled_timing_hbm',
        num_clusters=case['num_clusters'], expected_stages=case['expected_stages'],
        expected_stage_order=case['expected_stage_order'], cwd=str(attempt),
        expected_gemmini=case['expected_gemmini'],
        command=[plan['simulator'], str(path), '--binary-path',
            str(local(bundle, row['device_elf']['path'])), '--gen-trace', 'false'],
        progress=str(attempt / 'progress.jsonl'), hbm_report=str(attempt / 'hbm.json'),
        portable_bundle=str(bundle), portable_workload=case['name'], portable_results=str(attempt),
        prepared_timing_configuration_sha256=case['prepared_timing_configuration_sha256'],
        hash_inputs=[plan['simulator'], str(path), str(root / 'plan.json'),
            str(bundle / 'manifest.json'), str(source / 'workload.json'),
            str(source / 'model.json'), *dict.fromkeys(str(local(bundle, ref['path'])) for ref in inputs)],
        env={'CYCLOTRON_PERF_LOG_DIR': str(attempt / 'performance_logs'),
             'CYCLOTRON_GRAPH_LOG': '0'})


def summarize(journal, case):
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    finish = records[-1]
    if finish['event'] != 'finish' or finish.get('passed') is not True:
        raise ValueError('incomplete simulation has no full-model cycle result')
    entries = defaultdict(list)
    for record in records:
        if record['event'] == 'stage_entry':
            entries[record['cluster']].append(record)
    if set(entries) != set(range(case['num_clusters'])):
        raise ValueError('missing cluster timeline')
    timelines = []
    for cluster, rows in sorted(entries.items()):
        if len(rows) != case['expected_stages']:
            raise ValueError('incomplete stage timeline')
        groups = defaultdict(int)
        intervals = []
        for index, row in enumerate(rows):
            indices = {int(n) for name in row['symbols'] for n in re.findall(r'stage_(\d+)(?:Pv|$)', name)}
            if indices != {index}:
                raise ValueError('stage timeline out of order')
            end = rows[index + 1]['cycle'] if index + 1 < len(rows) else finish['cycle']
            if end < row['cycle']:
                raise ValueError('non-monotonic stage timestamps')
            stage = case['stages'][index]['stage']
            group = stage.split('.', 1)[0]
            cycles = end - row['cycle']
            groups[group] += cycles
            intervals.append(dict(index=index, stage=stage, start=row['cycle'], end=end, cycles=cycles))
        active = finish['cycle'] - rows[0]['cycle']
        assert sum(groups.values()) == active
        timelines.append(dict(cluster=cluster, first_stage_cycle=rows[0]['cycle'],
            cycles_from_first_stage_to_global_finish=active, section_cycles=dict(groups), stages=intervals))
    return dict(completed=True, hardware_calibrated=False, global_finish_cycle=finish['cycle'],
        journal_scope='core0 warp0 entry-to-next-entry intervals include launch/barrier gaps; last interval includes final checks and global completion tail; per-cluster timelines overlap and must not be summed',
        clusters=timelines)


def validated_job(queue_path):
    queue = json.loads(queue_path.read_text())
    prior = queue['jobs'][0]
    inputs_ok = all(Path(path).is_file() and digest(Path(path)) == sha
        for path, sha in prior['input_sha256'].items())
    log = (Path(prior['result_dir']) / 'run.log').read_text()
    if queue['status'] != 'passed' or not inputs_ok or not validate(prior, log, prior['returncode'])['passed']:
        raise ValueError('previous passing result changed; investigate before retrying')
    return prior


def save_timing(root, plan, case, queue_path):
    job = validated_job(queue_path)
    journal = queue_path.parent.parent / 'progress.jsonl'
    timing = summarize(journal, case)
    timing.update(name=case['name'], queue_result=str(queue_path),
        source_ticks=plan['source_ticks'], memory_ticks=plan['memory_ticks'],
        simulator_sha256=plan['simulator_sha256'], journal_sha256=digest(journal))
    timing['simulated_cycles'] = job['simulated_cycles']
    if timing['simulated_cycles'] != timing['global_finish_cycle'] + 1:
        raise ValueError('console cycle count and completed journal disagree')
    timing['hbm_source_cycles_after_drain'] = json.loads(Path(job['hbm_report']).read_text())['source_cycles']
    atomic_json(root / case['name'] / 'result.json', timing)


def run(root, tmpdir, names):
    root = root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    if digest(Path(plan['simulator'])) != plan['simulator_sha256']:
        raise ValueError('simulator changed after preparation')
    if digest(Path(plan['bundle']) / 'manifest.json') != plan['manifest_sha256']:
        raise ValueError('bundle changed after preparation')
    lock = (root / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # A killed supervisor can leave its queue or simulator alive. Never start
    # another attempt beside that process just because the parent lock expired.
    for queue_path in root.glob('*/attempt-*/queue/results.json'):
        prior = json.loads(queue_path.read_text())
        if prior['status'] != 'running':
            continue
        processes = [(prior['pid'], str(queue_path.parent))]
        processes += [(job['pid'], job['command'][1]) for job in prior['jobs'] if job.get('pid')]
        for pid, marker in processes:
            proc = Path('/proc') / str(pid)
            try:
                args = (proc / 'cmdline').read_bytes().split(b'\0')
                alive = proc.stat().st_uid == os.getuid() and marker.encode() in args
            except FileNotFoundError:
                alive = False
            if alive:
                raise RuntimeError('previous attempt is still running: ' + str(queue_path))
    state = dict(pid=os.getpid(), status='running', started=time.time(), cases=[])
    atomic_json(root / 'supervisor.json', state)
    try:
        selected = [case for case in plan['cases'] if not names or case['name'] in names]
        if names and set(names) != {case['name'] for case in selected}:
            raise ValueError('unknown case name')
        for case in selected:
            source = root / case['name']
            result = source / 'result.json'
            if result.exists():
                saved = json.loads(result.read_text())
                save_timing(root, plan, case, Path(saved['queue_result']))
                state['cases'].append(dict(name=case['name'], status='passed', reused=True))
                continue
            recovered = next((path for path in sorted(source.glob('attempt-*/queue/results.json'), reverse=True)
                if json.loads(path.read_text())['status'] == 'passed'), None)
            if recovered:
                save_timing(root, plan, case, recovered)
                state['cases'].append(dict(name=case['name'], status='passed', reused=True))
                continue
            if shutil.disk_usage(root).free < 2 * 1024**3:
                raise RuntimeError('less than 2 GiB free; refusing a new heavy simulation')
            number = 1 + len(list(source.glob('attempt-*')))
            attempt = source / f'attempt-{number:03d}'
            attempt.mkdir()
            job = make_job(root, plan, case, attempt)
            atomic_json(attempt / 'jobs.json', [job])
            state['current_case'] = case['name']
            state['current_attempt'] = str(attempt)
            atomic_json(root / 'supervisor.json', state)
            command = [sys.executable, str(CHAIN / 'mx_vpu/run_e2e_queue.py'),
                '--jobs', str(attempt / 'jobs.json'), '--out', str(attempt / 'queue'),
                '--tmpdir', str(tmpdir.resolve())]
            with (attempt / 'supervisor.log').open('x') as log:
                child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'TMPDIR': str(tmpdir.resolve()), 'PYTHONDONTWRITEBYTECODE': '1'})
                state['child_pid'] = child.pid
                atomic_json(root / 'supervisor.json', state)
                try:
                    rc = child.wait()
                except BaseException:
                    child.terminate()
                    child.wait()
                    raise
            qpath = attempt / 'queue/results.json'
            queue = json.loads(qpath.read_text())
            if rc or queue['status'] != 'passed':
                raise RuntimeError('case failed acceptance: ' + case['name'])
            save_timing(root, plan, case, qpath)
            state['cases'].append(dict(name=case['name'], status='passed', reused=False))
            atomic_json(root / 'supervisor.json', state)
        state['status'] = 'passed' if len(state['cases']) == len(plan['cases']) else 'selected_cases_passed'
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        state['finished'] = time.time()
        atomic_json(root / 'supervisor.json', state)
        lock.close()


def report(root):
    plan = json.loads((root / 'plan.json').read_text())
    rows = []
    for case in plan['cases']:
        path = root / case['name'] / 'result.json'
        row = dict(name=case['name'], num_clusters=case['num_clusters'], status='pending',
            simulated_cycles=None, global_finish_cycle=None, hbm_source_cycles_after_drain=None,
            source_ticks=plan['source_ticks'],
            memory_ticks=plan['memory_ticks'], hardware_calibrated=False)
        if path.exists():
            result = json.loads(path.read_text())
            row.update(status='passed', global_finish_cycle=result['global_finish_cycle'],
                       simulated_cycles=result['simulated_cycles'],
                       hbm_source_cycles_after_drain=result['hbm_source_cycles_after_drain'],
                       result_sha256=digest(path))
        rows.append(row)
    summary = dict(passed=len(rows) == 8 and all(row['status'] == 'passed' for row in rows),
        completed_cases=sum(row['status'] == 'passed' for row in rows), expected_cases=8,
        scope='unassisted complete graphs with coupled HBM; configured software clocks; FPGA execution is separate',
        rows=rows)
    atomic_json(root / 'timing-summary.json', summary)
    fields = ['name', 'num_clusters', 'status', 'simulated_cycles', 'global_finish_cycle',
              'hbm_source_cycles_after_drain',
              'source_ticks', 'memory_ticks', 'hardware_calibrated']
    with (root / 'cycles.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('bundle', 'cyclotron', 'simulator', 'output'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--cycle-limit', type=int, default=10**13)
    prep.add_argument('--source-ticks', type=int, default=1)
    prep.add_argument('--memory-ticks', type=int, default=1)
    execute = sub.add_parser('run')
    execute.add_argument('--root', type=Path, required=True)
    execute.add_argument('--tmpdir', type=Path, required=True)
    execute.add_argument('--names', nargs='+')
    collect = sub.add_parser('report')
    collect.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        plan = prepare(args.bundle, args.cyclotron, args.simulator, args.output,
            args.cycle_limit, args.source_ticks, args.memory_ticks)
        print(json.dumps(dict(prepared=len(plan['cases']), output=str(args.output))))
    elif args.command == 'run':
        def stop(signum, frame):
            raise KeyboardInterrupt('received signal ' + str(signum))
        signal.signal(signal.SIGTERM, stop)
        run(args.root, args.tmpdir, args.names)
    else:
        print(json.dumps(report(args.root), indent=2))


if __name__ == '__main__':
    main()
