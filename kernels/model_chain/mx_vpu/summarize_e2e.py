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
    optimized_path = audit / 'optimized-qualification-v3/runs/results.json'
    optimized = read(optimized_path, {})
    if optimized:
        native = [j for j in optimized['jobs'] if j['kind'] == 'native']
        latest_device = [j for j in optimized['jobs'] if j['kind'] == 'isa']
        replaced = {j['model'] for j in latest_device}
        device = latest_device + [j for j in device if j['model'] not in replaced]
        device_receipt = optimized
    csr_path = audit / 'csr-regression-v1/runs/results.json'
    csr = read(csr_path, {})
    if csr:
        hpc = dict(status=csr['status'], jobs=[j for j in csr['jobs'] if j['kind'] == 'hpc'])
        controls = dict(jobs=[j for j in csr['jobs'] if j['kind'] == 'gemm'])
    prepared = read(audit / 'prepared-overlay-full-native-v1/runs/results.json', {})
    if prepared:
        replaced = {j['model'] for j in prepared['jobs']}
        native = prepared['jobs']+[j for j in native if j['model'] not in replaced]
    smolvla = read(audit / 'smolvla-current-candidate.json', {})
    functional_queue = read(audit / 'functional-isa-and-checked-hpc-v1/runs/results.json', {})
    previous_hpc_path = audit/'owned-writeback-hpc-v1/runs/results.json'
    hpc_queue_path = audit/'helper25-hbm-hpc-regression-v1/runs/results.json'
    latest_planned = read(audit/'helper25-hbm-hpc-regression-v1/jobs.json',[])
    hpc_queue = read(hpc_queue_path,{})
    if not hpc_queue and latest_planned:
        hpc_queue = dict(status='pending',jobs=[dict(job,status='pending')for job in latest_planned])
    elif not hpc_queue:
        hpc_queue_path = previous_hpc_path
        hpc_queue = read(previous_hpc_path,{}) or functional_queue
    if hpc_queue:
        hpc_jobs = [j for j in hpc_queue['jobs'] if j['kind'] == 'hpc']
        hpc = dict(status='passed' if all(j['status'] == 'passed' for j in hpc_jobs) else
                   'failed' if any(j['status'] == 'failed' for j in hpc_jobs) else 'pending',jobs=hpc_jobs)
        controls = dict(jobs=[j for j in hpc_queue['jobs'] if j['kind'] == 'gemm'])
    models = []
    for entry in graph:
        name = entry['model']
        manifest = read(root / 'artifacts/models' / name / 'manifest.json')
        host = next((j for j in native if j['model'] == name), {})
        isa = next((j for j in device if j['model'] == name), {})
        earlier_isa = isa
        current_elf = None
        if any(j['model'] == name for j in prepared.get('jobs',[])):
            variant = audit / 'prepared-queries-overlay-v1' / name
            current_elf = str(variant/'kernel.radiance.elf')
        if name == 'smolvla_base' and smolvla.get('full_host_passed'):
            manifest = read(Path(smolvla['manifest']))
            qualified = read(Path(smolvla['native_result']))
            host = dict(status=qualified['status'],validation=dict(format_final_pass=qualified['failed_output_elements'] == 0))
            current_elf = str(Path(smolvla['manifest']).parent/'kernel.radiance.elf')
        if current_elf and isa.get('command',[None]*4)[3] != current_elf:
            isa = {}
        functional = next((j for j in functional_queue.get('jobs',[]) if j.get('model') == name
                           and j.get('execution_mode') == 'functional_core'),{})
        validation = host.get('validation', {})
        original = manifest.get('original_action_comparison')
        if name == 'smolvla_base' and smolvla.get('full_host_passed'):
            original = smolvla['actual_original_action_comparison']
        if original is None:
            original = dict(passed=manifest['format_comparison']['original_checkpoint_greedy_exact'],
                            metric='exact greedy token IDs', tokens=host.get('reference_generated_token_ids'))
        models.append(dict(model=name, stages=entry['stages'], graph_and_image_binding=entry['status'],
                           native_format_final_pass=validation.get('format_final_pass', validation.get('format_all_stage_checks_pass')),
                           native_all_checks=host.get('status', 'pending'),
                           original_checkpoint_comparison=original,
                           isa_status=isa.get('status', 'pending'), isa_pid=isa.get('pid'),
                           isa_result_dir=isa.get('result_dir'), progress=isa.get('progress'),
                           current_variant_elf=current_elf,
                           earlier_variant_isa={k:earlier_isa.get(k) for k in ('status','pid','result_dir','progress')},
                           functional_isa_status=functional.get('status','pending'),
                           functional_isa_progress=functional.get('progress'),
                           accepted=bool(host.get('status') == 'passed' and isa.get('status') == 'passed' and original['passed'])))
    count = lambda jobs: {status: sum(j.get('status') == status for j in jobs) for status in ('passed', 'failed', 'running', 'pending', 'waiting_for_simulator_slot')}
    known_failure = any(m['native_all_checks'] == 'failed' or not m['original_checkpoint_comparison']['passed']
                        or m['isa_status'] == 'failed' or m['functional_isa_status'] == 'failed' for m in models)
    known_failure = known_failure or hpc.get('status') == 'failed' or runtime.get('status') == 'failed' or any(j.get('status') == 'failed' for j in controls.get('jobs', []) if j['kind'] == 'gemm')
    full_assisted = {}
    for name in ('smolvla_base','tinyllama','deepseek_r1_distill_qwen_1_5b','gemma_2_2b_it'):
        if name == 'smolvla_base':
            folder = audit/'smolvla-stack-fixed-qualification-v1'
            qualified_path = folder/'qualified-result-v2.json'
            device_path = folder/'runs/results.json'
            native_path = None
        else:
            folder = audit/'decoder-full-qualification-v1'/name
            qualified_path = folder/'qualified-result.json'
            device_path = folder/'device-runs/results.json'
            native_path = folder/'native-runs/results.json'
        qualified = read(qualified_path,{})
        queue = read(device_path,{})
        full_assisted[name] = dict(
            status='passed' if qualified.get('passed') is True else queue.get('status','pending'),
            qualified=qualified.get('passed') is True,
            qualified_result=str(qualified_path) if qualified else None,
            native_result=read(native_path,{}) if native_path else None,
            device_jobs=[{k:j.get(k) for k in ('name','status','pid','progress','result_dir','validation')}
                         for j in queue.get('jobs',[])],
            scope='complete one/two SM functional graphs with pure helper assistance; no HBM/program latency')
    complete = (len(models) == 4 and all(m['native_all_checks'] in ('passed', 'failed')
                and m['isa_status'] in ('passed', 'failed') for m in models)
                and device_receipt.get('status') in ('passed', 'failed')
                and hpc.get('status') in ('passed', 'failed')
                and runtime.get('status') in ('passed', 'failed'))
    model_audits = read(audit/'full-decoder-audits-v1/supervisor.json',{})
    hbm_regression = read(audit/'helper25-hbm-hpc-regression-v1/supervisor.json',{})
    long_launch = read(audit/'long-launch-hbm-qualification-v1/supervisor.json',{})
    goal_audit = read(audit/'software-goal-completion-audit.json',{})
    scheduled_finished = (all(m['qualified']for m in full_assisted.values()) and
        model_audits.get('status')=='passed' and hbm_regression.get('status')=='passed' and
        long_launch.get('status')=='passed')
    current_failure = (any(m['status'] in ('failed','error','interrupted')for m in full_assisted.values()) or
        any(item.get('status')=='failed'for item in (model_audits,hbm_regression,long_launch)))
    objective_done = scheduled_finished and goal_audit.get('objective_complete') is True
    result = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                  status='passed' if objective_done else 'failed_with_checks_outstanding' if current_failure else
                      'completion_audit_pending' if scheduled_finished else 'checks_outstanding',
                  verification_finished=objective_done, objective_complete=objective_done,
                  scheduled_current_checks_finished=scheduled_finished,
                  historical_one_sm_timing_attempts=models,
                  historical_timing_attempts_finished=complete,
                  historical_timing_attempts_have_failure=known_failure,
                  full_assisted_functional_models=full_assisted,
                  all_four_full_assisted_functional_qualified=all(m['qualified'] for m in full_assisted.values()),
                  full_decoder_result_audits=model_audits,
                  latest_hbm_hpc_regression=hbm_regression,
                  long_launch_hbm_regression=long_launch,
                  software_goal_completion_audit=goal_audit,
                  smolvla_stage_order_audit=read(audit/'smolvla-stage-order-audit-v1/result.json'),
                  full_two_sm_models_status='assisted_functional_qualified' if all(m['qualified'] for m in full_assisted.values()) else 'checks_outstanding',
                  decoder_host_stage_controls=read(audit/'host-stage-controls-v1/qualified-result.json'),
                  decoder_host_stage_differential=read(audit/'host-stage-controls-v1/assistance-bit-differential.json'),
                  modeling_performance=read(audit / 'packing-compact-performance.json'),
                  activation_reuse_performance=read(audit / 'activation-reuse-performance.json'),
                  activation_reuse_full_native=read(audit / 'activation-reuse-full-native-v1/runs/results.json'),
                  prepared_query_full_native=prepared,
                  prepared_query_projection=read(audit / 'prepared-query-full-projection-overlay-v1/result.json'),
                  functional_projection=read(audit / 'prepared-query-functional-projection-v2/result.json'),
                  checked_simulator_regression=functional_queue,
                  owned_writeback_regression=read(previous_hpc_path),
                  owned_writeback_performance=read(audit / 'owned-writeback-performance-v1/result.json'),
                  smolvla_current_candidate=read(audit / 'smolvla-current-candidate.json'),
                  smolvla_two_sm_candidate=read(audit / 'smolvla-replicas-current.json'),
                  pure_helper_assistance=read(audit / 'functional-assist-current.json'),
                  smolvla_full_assisted=read(audit / 'smolvla-stack-fixed-qualification-v1/qualified-result-v2.json'),
                  worker_stack_runtime=read(audit / 'worker-stack-fix-v1/diagnosis.json'),
                  tinyllama_stall_diagnosis=read(audit / 'tinyllama-stall-diagnosis-v1/result.json'),
                  simulator_performance=read(audit / 'csr-performance.json'),
                  hpc=dict(status=hpc.get('status'), counts=count(hpc.get('jobs', [])), result=str(
                           hpc_queue_path if hpc_queue_path.exists() else
                           audit/'helper25-hbm-hpc-regression-v1/supervisor.json' if latest_planned else
                           audit / 'functional-isa-and-checked-hpc-v1/runs/results.json' if functional_queue else
                           csr_path if csr else audit / 'hpc-current-runs/results.json'),
                           scope='complete output hashes for STREAM, five Spatter transfer families, original-size xRAGE gather/scatter; 1SM with 1/2 logical partitions'),
                  fp8_gemm=[{k: j.get(k) for k in ('name', 'status', 'validation', 'simulated_cycles', 'result_dir')} for j in controls.get('jobs', []) if j['kind'] == 'gemm'],
                  runtime_regressions=[{k: j.get(k) for k in ('name', 'status', 'validation', 'simulated_cycles', 'result_dir')} for j in runtime.get('jobs', [])],
                  smolvla_boundaries=read(audit / 'smolvla-boundary-failures.json'),
                  smolvla_boundaries_scope='historical action-input variant before patch-order and compensated-norm fixes; current candidate is separate',
                  limits=['Assisted functional execution is separate from unassisted ISA and timing/HBM execution.',
                          'Native execution is generated stage C++ on the host, not device ISA execution.',
                          'Complete assisted functional graphs are reported separately from preserved earlier timing attempts.',
                          'Full coupled HBM model qualification remains outstanding; RTL/FireSim and FPGA are separate.',
                          'HBM clocks use a 1:1 control ratio, not calibrated Radiance wall time.'],
                  device_queue_result=str(optimized_path if optimized else audit / device_dir / 'results.json'))
    temporary = audit / 'summary.new'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(audit / 'summary.json')
    return objective_done


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
