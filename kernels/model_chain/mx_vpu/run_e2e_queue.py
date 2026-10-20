#!/usr/bin/env python3
"""Run retained full-graph checks serially and keep explicit, atomic receipts.

A native pass covers generated C++ on the host. A device pass additionally needs
successful tohost, full output checks in the ELF, and a finished progress journal.
An original-checkpoint comparison remains separate from the format comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import tomllib


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def stage_identity(record, numeric=False):
    names = {name for symbol in record.get('symbols', [])
             for name in re.findall(r'(stage_\w+)Pvjjj', symbol)}
    if len(names) != 1:
        return None
    name = next(iter(names))
    if numeric:
        return int(name[6:]) if name[6:].isdigit() else None
    return name


def gemmini_counts(job, log, clusters):
    """Read inline counters or the current process's announced per-SM reports."""
    loop_pattern = r'LOOP_WS pending (\d+), retired (\d+)'
    tile_pattern = r'mesh: (\d+) tiles,.*VPU: (\d+) commands'
    loops, tiles = re.findall(loop_pattern, log), re.findall(tile_pattern, log)
    paths = re.findall(r'^Cyclotron: gemmini report in (.+)$', log, re.MULTILINE)
    result = dict(loop_ws=loops, mesh_tiles_and_vpu=tiles, reports=[], report_binding_pass=True)
    if not paths:
        result['report_binding_pass'] = not job.get('gemmini_report_sha256')
        return result
    # Never search arbitrary directories or substitute stale reports. Cyclotron
    # prints exactly the files it wrote under run_<timestamp>_<process ID>.
    result['report_binding_pass'] = False
    if loops or tiles or len(paths) != clusters or not job.get('cwd') or not job.get('pid'):
        return result
    cwd = Path(job['cwd']).resolve()
    root = Path(job.get('env', {}).get('CYCLOTRON_PERF_LOG_DIR', 'performance_logs'))
    root = (root if root.is_absolute() else cwd / root).resolve()
    commands = re.findall(r'^Cyclotron: cluster (\d+) gemmini: (\d+) commands', log, re.MULTILINE)
    if len(commands) != clusters or {int(c) for c, _ in commands} != set(range(clusters)):
        return result
    commands = {int(c): int(count) for c, count in commands}
    reports = {}
    for raw in paths:
        path = Path(raw)
        path = (path if path.is_absolute() else cwd / path).resolve()
        cluster = re.fullmatch(r'gemmini_report_cluster(\d+)\.txt', path.name)
        if (not cluster or path.parent.parent != root or
                not re.fullmatch(r'run_\d+_' + str(job['pid']), path.parent.name)):
            return result
        cluster = int(cluster[1])
        if cluster not in commands or cluster in reports:
            return result
        try:
            text = path.read_text()
            sha = digest(path)
        except OSError:
            return result
        retired = re.findall(loop_pattern, text)
        physical = re.findall(tile_pattern, text)
        total = re.findall(r'^gemmini radiance: (\d+) commands', text, re.MULTILINE)
        if (len(retired) != 1 or len(physical) != 1 or len(total) != 1 or
                int(total[0]) != commands[cluster]):
            return result
        reports[cluster] = dict(cluster=cluster, path=str(path), sha256=sha,
                                loop_ws=retired[0], mesh_tiles_and_vpu=physical[0])
    result['reports'] = [reports[c] for c in range(clusters)]
    pins = {r['path']: r['sha256'] for r in result['reports']}
    result.update(loop_ws=[r['loop_ws'] for r in result['reports']],
                  mesh_tiles_and_vpu=[r['mesh_tiles_and_vpu'] for r in result['reports']],
                  report_binding_pass=job.get('gemmini_report_sha256', pins) == pins)
    return result


def validate(job, log, returncode):
    result = {'process_ok': returncode == 0}
    if job['kind'] == 'native':
        if job['model'] == 'smolvla_base':
            actions = re.search(r'actions=(\d+) failures=(\d+).*max_abs_error=([^\s]+)', log)
            boundaries = re.search(r'BOUNDARY_TOTAL checked=(\d+) failed_stages=(\d+)', log)
            result['format_final_pass'] = bool(actions and int(actions[1]) == 1600 and int(actions[2]) == 0)
            result['module_boundaries_pass'] = bool(boundaries and int(boundaries[1]) == 1930 and int(boundaries[2]) == 0)
            result['final_summary'] = actions[0] if actions else None
            result['boundary_summary'] = boundaries[0] if boundaries else None
        else:
            replicas = job.get('num_replicas', 1)
            summaries = re.findall(r'stages=(\d+) check_chunks=(\d+) tohost=0', log)
            result['format_all_stage_checks_pass'] = (len(summaries) == replicas and
                all(int(stages) == job['expected_stages'] and int(chunks) > 0 for stages,chunks in summaries))
            result['generated_tokens'] = [int(x) for x in re.findall(r'generated_token_\d+=(-?\d+)', log)]
            result['greedy_tokens_pass'] = result['generated_tokens'] == job['reference_generated_token_ids']*replicas
            if replicas != 1:
                result['all_native_replicas_pass'] = (replicas == 2 and
                    re.findall(r'REPLICA_BEGIN (\d+)', log) == ['0','1'] and
                    re.findall(r'REPLICA_DONE (\d+) status=0', log) == ['0','1'])
    elif job['kind'] == 'hpc':
        output = Path(job['output'])
        result['tohost_pass'] = 'isa-test passed with tohost=0' in log
        if job.get('num_clusters') is not None:
            clusters = job['num_clusters']
            completed = [int(c) for c in re.findall(r'cluster (\d+) isa-test passed with tohost=0', log)]
            result['tohost_pass'] = len(completed) == clusters and set(completed) == set(range(clusters))
            result['completed_clusters'] = completed
        result['full_output_pass'] = (output.exists() and output.stat().st_size == job['expected_bytes'] and digest(output) == job['expected_output_sha256'])
        if output.exists():
            result['output_sha256'] = digest(output)
            result['output_bytes'] = output.stat().st_size
    elif job['kind'] == 'gemm':
        from check_fp8_gemm import check
        result.update(check(Path(job['output']), Path(job['golden']), log, job['clusters']))
    elif job['kind'] == 'isa':
        config = (tomllib.loads(Path(job['command'][1]).read_text()) if job.get('command') else {})
        mode = job.get('execution_mode')
        assist_path = config.get('sim',{}).get('functional_assist')
        result['assistance_scope_pass'] = not assist_path or mode == 'assisted_functional_core'
        if mode in ('functional_core','assisted_functional_core'):
            result['functional_mode_config_pass'] = (
                config.get('sim',{}).get('timing') is False and
                config.get('gemmini',{}).get('mode') == 'both' and
                config.get('gemmini',{}).get('preset') == 'radiance' and
                not config.get('gemmini',{}).get('overrides','') and
                not config.get('timing',{}).get('gmem',{}).get('hbm') and
                not job.get('hbm_report'))
            result['execution_scope'] = 'functional core ISA plus libgemmini both; no HBM timing or program-latency claim'
            if mode == 'functional_core':
                result['unassisted_core_pass'] = not assist_path
            else:
                result['execution_scope'] = 'pure helper assistance inside functional core ISA; scheduling/MX/VPU remain ISA; no HBM/program latency'
                result['assistance_completed_pass'] = False
                if assist_path and job.get('assistance_report'):
                    config_dir = Path(job['command'][1]).resolve().parent
                    plan_path = Path(assist_path)
                    if not plan_path.is_absolute(): plan_path = config_dir/plan_path
                    plan = json.loads(plan_path.read_text())
                    report_path = Path(plan['report'])
                    if not report_path.is_absolute(): report_path = plan_path.parent/report_path
                    report = json.loads(report_path.read_text()) if report_path.exists() else {}
                    result['assistance_completed_pass'] = bool(
                        Path(job['assistance_report']).resolve() == report_path.resolve() and
                        report.get('schema') == 'radiance-pure-fallback-result-v1' and report.get('completed') is True and
                        report.get('whole_program_latency_valid') is False and report.get('includes_hbm') is False and
                        report.get('elf_sha256') == plan.get('elf_sha256') == digest(Path(job['command'][3])) and
                        report.get('plugin_sha256') == plan.get('plugin_sha256') and
                        report.get('combined_source_sha256') == plan.get('combined_source_sha256') and
                        report.get('statistics',{}).get('warp_calls',0) > 0 and
                        sum(v.get('work_items',0) for v in report.get('statistics',{}).get('operations',{}).values()) > 0)
                    result['assistance_report'] = report

        clusters = job.get('num_clusters')
        if clusters is None:
            config = tomllib.loads(Path(job['command'][1]).read_text())
            clusters = config.get('sim', {}).get('num_clusters', 1)
        finished = [int(x) for x in re.findall(r'cluster (\d+) isa-test passed with tohost=0', log)]
        result['tohost_pass'] = len(finished) == clusters and set(finished) == set(range(clusters))
        if job.get('expected_gemmini'):
            expected = job['expected_gemmini']
            counts = gemmini_counts(job, log, clusters)
            loops, tiles = counts['loop_ws'], counts['mesh_tiles_and_vpu']
            result['accelerator_counts_pass'] = (counts['report_binding_pass'] and
                len(loops) == clusters and len(tiles) == clusters and
                all(int(pending) == 0 and int(retired) == expected['mx_loops'] for pending, retired in loops) and
                all(int(mesh) == expected['mesh_tiles'] and int(vpu) == expected['vpu_commands'] for mesh,vpu in tiles))
            result['accelerator_counts'] = counts
        journal = Path(job['progress'])
        records = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        entries = [x for x in records if x['event'] == 'stage_entry']
        result['stage_entries'] = len(entries)
        result['journal_finished_pass'] = bool(records and records[-1]['event'] == 'finish' and records[-1]['passed'])
        result['journal_has_stages'] = bool(entries)
        result['every_cluster_reached_stages_pass'] = {x['cluster'] for x in entries} == set(range(clusters))
        if job.get('expected_stages') is not None:
            counts = {cluster: sum(x['cluster'] == cluster for x in entries) for cluster in range(clusters)}
            result['stage_entries_per_cluster'] = counts
            result['complete_stage_count_pass'] = all(count == job['expected_stages'] for count in counts.values())
        if job.get('expected_stage_order') is not None:
            orders = {cluster: [] for cluster in range(clusters)}
            expected = job['expected_stage_order']
            numeric = bool(expected) and isinstance(expected[0], int)
            for entry in entries:
                orders.setdefault(entry['cluster'], []).append(stage_identity(entry, numeric))
            result['stage_order_pass'] = (set(orders) == set(range(clusters)) and
                all(order == expected for order in orders.values()))
        # Entries prove execution reached an address, not that a stage completed.
        result['journal_scope'] = 'core 0 warp 0 retired function entries; acceptance uses final ELF verification'
        if job.get('model_manifest'):
            model_manifest = Path(job['model_manifest']).resolve()
            model = json.loads(model_manifest.read_text())
            if model.get('host_stage_checks'):
                result['host_all_stages_pass'] = False
                if job.get('host_stage_result_dir'):
                    from host_stage_checks import check as check_actual_gpu_stages
                    checks = check_actual_gpu_stages(model_manifest.parent,
                        Path(job['host_stage_result_dir']), clusters)
                    result['host_all_stages_pass'] = checks['passed']
                    result['host_stage_checks'] = checks
    else:
        raise ValueError('Unknown checker: ' + job['kind'])
    if job.get('portable_bundle'):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from workload_bundle import check_outputs
        checks = check_outputs(Path(job['portable_bundle']), job['portable_workload'],
                               Path(job['portable_results']))
        result['portable_outputs_pass'] = checks['passed']
        result['portable_outputs'] = checks
    if job.get('execution_mode') == 'coupled_timing_hbm':
        config = (tomllib.loads(Path(job['command'][1]).read_text())
                  if job['kind'] != 'native' and job.get('command') else {})
        assist_path = config.get('sim',{}).get('functional_assist')
        hbm_config = config.get('timing',{}).get('gmem',{}).get('hbm',{})
        result['coupled_timing_config_pass'] = bool(
            config.get('sim',{}).get('timing') is True and
            config.get('gemmini',{}).get('enabled') is True and
            config.get('gemmini',{}).get('mode') == 'both' and
            config.get('gemmini',{}).get('preset') == 'radiance' and
            not config.get('gemmini',{}).get('overrides','') and
            not assist_path and hbm_config.get('config'))
        report = Path(hbm_config['report']) if hbm_config.get('report') else None
        if report is not None and not report.is_absolute():
            report = Path(job['command'][1]).resolve().parent/report
        result['coupled_report_binding_pass'] = bool(report and job.get('hbm_report') and
            report.resolve() == Path(job['hbm_report']).resolve())
        if job.get('prepared_timing_configuration_sha256'):
            result['coupled_configuration_pins_pass'] = all(
                Path(path).is_file() and digest(path) == expected
                for path,expected in job['prepared_timing_configuration_sha256'].items())
        result['execution_scope'] = 'unassisted core ISA and libgemmini both with shared HBM completion/backpressure; configured software clocks'
    if job.get('hbm_report'):
        report = Path(job['hbm_report'])
        hbm = json.loads(report.read_text()) if report.exists() else None
        result['hbm'] = hbm
        result['hbm_drained_pass'] = bool(hbm and hbm['unconsumed_requests'] == 0 and hbm['pending_dram'] == 0)
        if job.get('expected_hbm_core_sources'):
            sources = hbm.get('sources', {}) if hbm else {}
            result['all_hbm_core_sources_pass'] = all(
                sources.get(str(core), 0) > 0 for core in job['expected_hbm_core_sources'])
        if job.get('execution_mode') == 'coupled_timing_hbm':
            configured = config.get('timing',{}).get('gmem',{}).get('hbm',{})
            source_ticks = configured.get('source_ticks',1)
            memory_ticks = configured.get('memory_ticks',1)
            result['coupled_hbm_clock_ratio_pass'] = bool(hbm and
                source_ticks > 0 and memory_ticks > 0 and
                hbm.get('source_ticks_per_period') == source_ticks and
                hbm.get('memory_ticks_per_period') == memory_ticks)
    checks = [value for key, value in result.items() if key.endswith('_pass') or key == 'process_ok' or key == 'journal_has_stages']
    result['passed'] = all(checks)
    return result


def resume_previous(path):
    receipt = json.loads(path.read_text())
    proc = Path('/proc') / str(receipt['pid'])
    result = {'time': time.time(), 'pid': receipt['pid']}
    try:
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if proc.stat().st_uid != os.getuid() or command != receipt['command']:
            result['status'] = 'identity_changed_not_signalled'
        else:
            os.kill(receipt['pid'], signal.SIGCONT)
            result['status'] = 'resumed'
    except FileNotFoundError:
        result['status'] = 'already_exited'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tmpdir', type=Path, required=True)
    parser.add_argument('--resume-process', type=Path, action='append', default=[])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = args.out / 'results.json'
    if receipt.exists():
        raise SystemExit('Refusing to overwrite a previous queue; use a new output directory.')
    jobs = json.loads(args.jobs.read_text())
    for job in jobs:
        if job.get('kind') not in ('native', 'isa', 'gemm', 'hpc'):
            raise ValueError('Missing or unsupported job kind: ' + str(job.get('name', job.get('model'))))
        job['status'] = 'pending'
    state = {'pid': os.getpid(), 'started': time.time(), 'status': 'running',
             'jobs_sha256': digest(args.jobs), 'jobs': jobs}
    child = None

    def save():
        state['updated'] = time.time()
        atomic_json(receipt, state)

    def stop(signum, frame):
        raise KeyboardInterrupt('received signal ' + str(signum))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    try:
        for index, job in enumerate(jobs):
            dest = args.out / f'{index:02d}-{job["name"]}'
            dest.mkdir()
            job['result_dir'] = str(dest)
            dependency = job.get('wait_for_receipt')
            if dependency:
                job['status'] = 'waiting_for_simulator_slot'
                save()
                while True:
                    previous = json.loads(Path(dependency).read_text())
                    if previous['status'] not in ('running', 'pending'):
                        break
                    # Never start another simulator if the dependency's worker vanished.
                    if not Path('/proc', str(previous['pid'])).exists():
                        raise RuntimeError('Dependency worker exited without a final receipt: ' + dependency)
                    time.sleep(5)
            if job.get('wait_for_build'):
                job['status'] = 'waiting_for_build'
                save()
                build_path = Path(job['wait_for_build'])
                while not build_path.exists():
                    builder = Path('/proc', str(job['builder_pid']))
                    if (not builder.exists() or builder.stat().st_uid != os.getuid() or
                        (builder / 'cmdline').read_bytes().replace(b'\0', b' ').decode() != job['builder_command']):
                        raise RuntimeError('Builder exited without receipt: ' + str(build_path))
                    time.sleep(5)
                build = json.loads(build_path.read_text())
                assert build['image_placement_verified'] and digest(build['elf']) == build['elf_sha256']
                job['build_sha256'] = digest(build_path)
            if job.get('require_receipt_pass'):
                dependency = Path(job['require_receipt_pass'])
                while True:
                    control = json.loads(dependency.read_text())
                    if control['status'] not in ('running', 'pending'):
                        if control['status'] != 'passed':
                            raise RuntimeError('Required control failed: ' + str(dependency))
                        break
                    time.sleep(5)
            job['status'] = 'running'
            job['started'] = time.time()
            job['input_sha256'] = {p: digest(p) for p in job.get('hash_inputs', [])}
            save()
            print('RUN', job['name'], flush=True)
            env = {**os.environ, 'TMPDIR': str(args.tmpdir), 'OMP_NUM_THREADS': '8', 'PYTHONDONTWRITEBYTECODE': '1', **job.get('env', {})}
            with (dest / 'run.log').open('w') as output:
                child = subprocess.Popen(job['command'], cwd=job['cwd'], env=env,
                                         stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                job['pid'] = child.pid
                save()
                while child.poll() is None:
                    job['elapsed_seconds'] = time.time() - job['started']
                    save()
                    time.sleep(5)
                job['returncode'] = child.returncode
                child = None
            job['seconds'] = time.time() - job['started']
            log = (dest / 'run.log').read_text()
            job['validation'] = validate(job, log, job['returncode'])
            reports = job['validation'].get('accelerator_counts', {}).get('reports', [])
            if reports:
                job['gemmini_report_sha256'] = {r['path']: r['sha256'] for r in reports}
            job['log_sha256'] = digest(dest / 'run.log')
            job['native_command_counts'] = re.findall(r'MX_VPU_MODEL commands=(\d+) mx=(\d+) vpu=(\d+) cycles=(\d+)', log)
            job['device_command_counts'] = re.findall(r'cluster (\d+) gemmini: (\d+) commands', log)
            cycles = re.search(r'finished after (\d+) cycles', log)
            if cycles:
                job['simulated_cycles'] = int(cycles[1])
            for field in ('progress', 'hbm_report'):
                if field in job and Path(job[field]).exists():
                    job[field + '_sha256'] = digest(job[field])
            job['status'] = 'passed' if job['validation']['passed'] else 'failed'
            atomic_json(dest / 'result.json', job)
            save()
            print('END', job['name'], job['status'], flush=True)
        state['status'] = 'passed' if all(job['status'] == 'passed' for job in jobs) else 'failed'
    except BaseException as error:
        state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'error'
        state['error'] = repr(error)
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if args.resume_process:
            state['previous_simulator'] = [resume_previous(path) for path in args.resume_process]
        state['finished'] = time.time()
        save()


if __name__ == '__main__':
    main()
