#!/usr/bin/env python3
"""Publish an honest snapshot of completed, failed and outstanding checks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def summarize(root):
    audit = root / 'artifacts/e2e-hbm-audit'
    graph = read(audit / 'graph-audit.json', {}).get('models', [])
    native = read(audit / 'native-runs/results.json', {}).get('jobs', [])
    device_dir = 'device-final-runs' if (audit / 'device-final-runs/results.json').exists() else 'device-current-runs'
    device_receipt = read(audit / device_dir / 'results.json', {})
    device = device_receipt.get('jobs', [])
    hpc = read(audit / 'hpc-current-runs/results.json', {})
    controls = read(audit / 'cluster-control-runs/results.json', {})
    runtime = read(audit / 'runtime-control-runs/results.json', {})
    models = []
    for entry in graph:
        name = entry['model']
        manifest = read(root / 'artifacts/models' / name / 'manifest.json')
        host = next((j for j in native if j['model'] == name), {})
        isa = next((j for j in device if j['model'] == name), {})
        validation = host.get('validation', {})
        original = manifest.get('original_action_comparison')
        if original is None:
            original = dict(passed=manifest['format_comparison']['original_checkpoint_greedy_exact'],
                            metric='exact greedy token IDs', tokens=host.get('reference_generated_token_ids'))
        models.append(dict(model=name, stages=entry['stages'], graph_and_image_binding=entry['status'],
                           native_format_final_pass=validation.get('format_final_pass', validation.get('format_all_stage_checks_pass')),
                           native_all_checks=host.get('status', 'pending'),
                           original_checkpoint_comparison=original,
                           isa_status=isa.get('status', 'pending'), isa_pid=isa.get('pid'),
                           isa_result_dir=isa.get('result_dir'), progress=isa.get('progress'),
                           accepted=bool(host.get('status') == 'passed' and isa.get('status') == 'passed' and original['passed'])))
    count = lambda jobs: {status: sum(j.get('status') == status for j in jobs) for status in ('passed', 'failed', 'running', 'pending', 'waiting_for_simulator_slot')}
    known_failure = any(m['native_all_checks'] == 'failed' or not m['original_checkpoint_comparison']['passed'] or m['isa_status'] == 'failed' for m in models)
    known_failure = known_failure or hpc.get('status') == 'failed' or runtime.get('status') == 'failed' or any(j.get('status') == 'failed' for j in controls.get('jobs', []) if j['kind'] == 'gemm')
    complete = bool(device) and device_receipt.get('status') in ('passed', 'failed') and hpc.get('status') in ('passed', 'failed') and runtime.get('status') in ('passed', 'failed')
    result = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                  status=('failed' if complete else 'failed_with_checks_outstanding') if known_failure else ('passed' if complete else 'checks_outstanding'),
                  verification_finished=complete, full_one_sm_models=models,
                  hpc=dict(status=hpc.get('status'), counts=count(hpc.get('jobs', [])), result=str(audit / 'hpc-current-runs/results.json'),
                           scope='complete output hashes for STREAM, five Spatter transfer families, original-size xRAGE gather/scatter; 1SM with 1/2 logical partitions'),
                  fp8_gemm=[{k: j.get(k) for k in ('name', 'status', 'validation', 'simulated_cycles', 'result_dir')} for j in controls.get('jobs', []) if j['kind'] == 'gemm'],
                  runtime_regressions=[{k: j.get(k) for k in ('name', 'status', 'validation', 'simulated_cycles', 'result_dir')} for j in runtime.get('jobs', [])],
                  smolvla_boundaries=read(audit / 'smolvla-boundary-failures.json'),
                  limits=['Native execution is generated stage C++ on the host, not device ISA execution.',
                          'Full two-SM model runs, complete Radiance RTL/FireSim and FPGA runs are not covered by this queue.',
                          'HBM clocks use a 1:1 control ratio, not calibrated Radiance wall time.'],
                  device_queue_result=str(audit / device_dir / 'results.json'))
    temporary = audit / 'summary.new'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(audit / 'summary.json')
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        done = summarize(args.task.resolve())
        if done or not args.watch:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
