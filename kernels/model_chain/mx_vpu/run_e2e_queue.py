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
            summary = re.search(r'stages=(\d+) check_chunks=(\d+) tohost=0', log)
            result['format_all_stage_checks_pass'] = bool(summary and int(summary[1]) == job['expected_stages'] and int(summary[2]) > 0)
            result['generated_tokens'] = [int(x) for x in re.findall(r'generated_token_\d+=(-?\d+)', log)]
            result['greedy_tokens_pass'] = result['generated_tokens'] == job['reference_generated_token_ids']
    elif job['kind'] == 'hpc':
        output = Path(job['output'])
        result['tohost_pass'] = 'isa-test passed with tohost=0' in log
        result['full_output_pass'] = (output.exists() and output.stat().st_size == job['expected_bytes'] and digest(output) == job['expected_output_sha256'])
        if output.exists():
            result['output_sha256'] = digest(output)
            result['output_bytes'] = output.stat().st_size
    elif job['kind'] == 'gemm':
        from check_fp8_gemm import check
        result.update(check(Path(job['output']), Path(job['golden']), log, job['clusters']))
    elif job['kind'] == 'isa':
        if job.get('execution_mode') == 'functional_core':
            config = tomllib.loads(Path(job['command'][1]).read_text())
            result['functional_mode_config_pass'] = (
                config.get('sim',{}).get('timing') is False and
                config.get('gemmini',{}).get('mode') == 'both' and
                config.get('gemmini',{}).get('preset') == 'radiance' and
                not config.get('gemmini',{}).get('overrides','') and
                not config.get('timing',{}).get('gmem',{}).get('hbm') and
                not job.get('hbm_report'))
            result['execution_scope'] = 'functional core ISA plus libgemmini both; no HBM timing or program-latency claim'
        clusters = job.get('num_clusters')
        if clusters is None:
            config = tomllib.loads(Path(job['command'][1]).read_text())
            clusters = config.get('sim', {}).get('num_clusters', 1)
        finished = [int(x) for x in re.findall(r'cluster (\d+) isa-test passed with tohost=0', log)]
        result['tohost_pass'] = len(finished) == clusters and set(finished) == set(range(clusters))
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
        # Entries prove execution reached an address, not that a stage completed.
        result['journal_scope'] = 'core 0 warp 0 retired function entries; acceptance uses final ELF verification'
    else:
        raise ValueError('Unknown checker: ' + job['kind'])
    if job.get('hbm_report'):
        report = Path(job['hbm_report'])
        hbm = json.loads(report.read_text()) if report.exists() else None
        result['hbm'] = hbm
        result['hbm_drained_pass'] = bool(hbm and hbm['unconsumed_requests'] == 0 and hbm['pending_dram'] == 0)
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
