#!/usr/bin/env python3
"""Audit completed full decoder runs against frozen fixtures and actual GPU RAM.

This reads retained outputs; it never runs a model, supplies GPU reference data,
or promotes functional ticks to hardware latency. One-SM audits remain partial.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from host_stage_checks import gold_lines
from run_e2e_queue import validate
sys.path.insert(0, str(HERE.parents[1]/'evaluation/llm'))
from stitch import model_specs

LINEAR_OPERATIONS = frozenset(('1','2','3','18','19','20','21','22','23'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            result.update(data)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def check_hashes(hashes):
    require(bool(hashes), 'missing pinned input hashes')
    for name, expected in hashes.items():
        require(digest(name) == expected, 'input changed since execution: '+name)


def stage_order(path, clusters, stages):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    orders = {cluster: [] for cluster in range(clusters)}
    for record in records:
        if record['event'] != 'stage_entry':
            continue
        require(record['cluster'] in orders, 'unexpected cluster in stage journal')
        indices = {int(match[1]) for symbol in record['symbols']
                   if (match := re.search(r'stage_(\d+)Pv', symbol))}
        require(len(indices) == 1, 'ambiguous stage identity in journal')
        orders[record['cluster']].append(indices.pop())
    require(all(indices == list(range(stages)) for indices in orders.values()),
            'stage journal omits, repeats, or reorders a graph stage')
    return {str(cluster): len(indices) for cluster,indices in orders.items()}


def expected_accelerator_counts(manifest, references, spec, native_counts):
    """Derive query tile/VPU work from shapes, separately from native counters."""
    require(manifest['mx_linear_policy'] == 'q_proj', 'unsupported matrix policy')
    by_symbol = {row['symbol']:row for row in references['records']}
    require(len(by_symbol) == manifest['stages'], 'missing distinct stage references')
    k = spec['hidden_size']
    n = spec['num_attention_heads'] * spec['head_dim']
    loops = 0
    vpu = 0
    mx_macs = 0
    for stage in manifest['stage_backends']:
        symbol = 'v_'+stage['stage'].replace('.', '_')
        require(symbol in by_symbol, 'backend stage has no output reference: '+symbol)
        count = by_symbol[symbol]['elements']
        if stage['backend'] == 'fp8_mx':
            require(stage['stage'].endswith('.q_proj'), 'unexpected MX projection')
            require(count % n == 0, 'query output shape does not fit model heads')
            m = count // n
            loops += ((m+15)//16) * ((k+31)//32) * ((n+15)//16)
            mx_macs += m*k*n
        elif stage['backend'] == 'bf16_vpu':
            require(stage['stage'].endswith(('.attn_residual','.ffn_residual',
                                             '.q_bias','.k_bias','.v_bias')),
                    'unexpected decoder VPU operation')
            vpu += (count+255)//256
        else:
            require(stage['backend'] == 'muon', 'unknown stage backend')
    require(loops > 0 and vpu > 0, 'model lacks useful MX/VPU work')
    require(mx_macs == manifest['mx_linear_macs'], 'MX MAC count differs from graph shapes')
    require(len(native_counts) == 1, 'missing native command summary')
    total, native_loops, native_vpu, _ = map(int, native_counts[0])
    replicas = manifest['batch_clusters']
    require(total % replicas == 0 and native_loops == loops*replicas and
            native_vpu == vpu*replicas, 'native counters differ from graph-derived command work')
    return dict(total_commands=total//replicas, mx_loops=loops,
                mesh_tiles=2*loops, vpu_commands=vpu)


def qualify(target, frozen, native_receipt, device_receipt, clusters=(1,2)):
    target = target.resolve()
    manifest = read(target/'manifest.json')
    baseline = read(frozen/'manifest.json')
    specs = model_specs()
    model = manifest['model']
    require(model in specs, 'unsupported model')
    require(manifest.get('full_model_dimensions') and manifest.get('checkpoint_weights') and
            manifest['layers'] == specs[model]['num_hidden_layers'] and
            manifest['stage_limit'] is None and manifest['batch_clusters'] == 2,
            'qualification requires an untruncated full checkpoint with two replicas')
    require(manifest.get('host_stage_checks') and manifest.get('fallback_call_abi') and
            manifest['execution_backend'] == 'rakanic_mx_vpu' and manifest['generation'] == 'greedy',
            'unsupported full decoder verification mode')
    fields = ('model','layers','prefill_tokens','decode_steps','input_token_ids',
              'checkpoint_sha256','weight_image_sha256','stages',
              'reference_generated_token_ids','source_spec_sha256')
    require(all(manifest[key] == baseline[key] for key in fields), 'frozen model or fixture changed')
    require(manifest['format_comparison']['original_checkpoint_greedy_exact'] is True,
            'format policy fails original-checkpoint greedy acceptance')
    require(gold_lines(target/'model_data.cpp') == gold_lines(frozen/'model_data.cpp'),
            'format stage oracles differ from frozen package')
    references = read(target/'host-stage-references.json')
    require(references['stages'] == manifest['stages'] and
            references['frozen_data_sha256'] == digest(frozen/'model_data.cpp') and
            references['rtol'] == .005 and references['atol'] == .0005,
            'frozen reference binding or original tolerance changed')
    require(len(manifest['verified_tensors']) == len(manifest['stage_backends']) == manifest['stages'],
            'stage verification/backend coverage is incomplete')
    build = read(target/'build.json')
    require(build['image_placement_verified'] and build['all_frozen_stage_references_identical'],
            'missing full image/reference preflight')
    require(digest(target/'kernel.radiance.elf') == build['elf_sha256'] == manifest['radiance_elf_sha256'],
            'GPU ELF binding changed')
    require(digest(target/'manifest.json') == build['manifest_sha256'], 'built manifest changed')
    for key in ('source_sha256','objects_sha256','native_objects_sha256'):
        hashes = build[key]
        if key == 'source_sha256': hashes = {str(target/name):value for name,value in hashes.items()}
        check_hashes(hashes)
    nq = read(native_receipt)
    native_jobs = [job for job in nq['jobs'] if job.get('model') == model]
    require(nq['status'] == 'passed' and len(native_jobs) == 1, 'full native qualification incomplete')
    native = native_jobs[0]
    require(native['kind'] == 'native' and native['num_replicas'] == 2 and
            native['expected_stages'] == manifest['stages'] and
            native['reference_generated_token_ids'] == manifest['reference_generated_token_ids'],
            'native receipt covers a different graph or replica count')
    native_log = Path(native['result_dir'])/'run.log'
    check_hashes(native['input_sha256'])
    require(digest(native_log) == native['log_sha256'], 'native log changed')
    require(validate(native,native_log.read_text(),native['returncode'])['passed'],
            'native stage or original greedy check failed')
    expected = expected_accelerator_counts(manifest,references,specs[model],native['native_command_counts'])
    dq = read(device_receipt)
    jobs = dq['jobs']
    require(len(jobs) == 2 and sorted(job['num_clusters'] for job in jobs) == [1,2],
            'device queue does not contain the matching one-/two-SM pair')
    results = []
    tensor_digests = {}
    for n in clusters:
        job = next(job for job in jobs if job['num_clusters'] == n)
        require(job['status'] == 'passed' and job['model'] == model and
                job['execution_mode'] == 'assisted_functional_core' and
                job['expected_stages'] == manifest['stages'] and
                Path(job['model_manifest']).resolve() == target/'manifest.json' and
                Path(job['command'][3]).resolve() == target/'kernel.radiance.elf',
                'full requested GPU run is incomplete or uses a different model/ELF')
        check_hashes(job['input_sha256'])
        log_path = Path(job['result_dir'])/'run.log'
        require(digest(log_path) == job['log_sha256'], 'GPU log changed')
        require(digest(job['progress']) == job['progress_sha256'], 'GPU journal changed')
        current = validate(dict(job,expected_gemmini=expected),log_path.read_text(),job['returncode'])
        require(current['passed'], 'actual GPU/reference/command checks failed')
        require(sorted(job['device_command_counts']) == [[str(c),str(expected['total_commands'])]for c in range(n)],
                'total commands differ from native graph execution')
        order = stage_order(job['progress'],n,manifest['stages'])
        statistics = current['assistance_report']['statistics']
        fallback_macs = sum(statistics['operations'].get(op,{}).get('multiply_accumulate_terms',0)
                            for op in LINEAR_OPERATIONS)
        expected_macs = manifest['total_linear_macs']-manifest['mx_linear_macs']
        require(fallback_macs == n*expected_macs, 'fallback linear work differs from complete graph')
        actual = Path(job['host_stage_result_dir'])/'tensors'
        for cluster in range(n):
            for row in references['records']:
                value = digest(actual/f'{cluster}-{row["symbol"]}.bin')
                previous = tensor_digests.setdefault(row['symbol'],value)
                require(previous == value, 'full stage differs between one/two-SM replicas')
        results.append(dict(num_clusters=n,passed=True,stages_per_cluster=order,
                            actual_stage_checks=current['host_stage_checks'],
                            accelerator_counts=current['accelerator_counts'],
                            linear_fallback_macs=fallback_macs,
                            seconds_including_queue_polling=job['seconds'],
                            job_result_sha256=digest(Path(job['result_dir'])/'result.json'),
                            log_sha256=digest(log_path),progress_sha256=digest(job['progress'])))
    return dict(schema='radiance-full-decoder-qualified-v1',passed=True,model=model,
                qualified_sm_counts=list(clusters),full_one_two_sm_qualified=tuple(clusters)==(1,2),
                original_greedy_tokens=manifest['reference_generated_token_ids'],
                original_checkpoint_greedy_exact=True,stage_buffers_bitidentical=True,
                expected_accelerator_counts_per_sm=expected,total_linear_macs=manifest['total_linear_macs'],
                mx_linear_macs=manifest['mx_linear_macs'],
                mx_linear_fraction=manifest['mx_linear_macs']/manifest['total_linear_macs'],
                source_build_sha256=digest(target/'build.json'),frozen_manifest_sha256=digest(frozen/'manifest.json'),
                native_receipt_sha256=digest(native_receipt),
                device_receipt_sha256=digest(device_receipt) if dq['status']=='passed' else None,
                qualifier_source_sha256=digest(Path(__file__)),
                whole_program_latency_valid=False,includes_hbm=False,results=results,
                scope='complete actual GPU graphs with qualified pure helper assistance; every typed stage buffer checked against frozen format references; actual MX/VPU; not hardware latency')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--frozen',type=Path,required=True)
    parser.add_argument('--native-receipt',type=Path,required=True)
    parser.add_argument('--device-receipt',type=Path,required=True)
    parser.add_argument('--clusters',nargs='+',type=int,choices=(1,2),default=[1,2])
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    require(args.clusters in ([1],[1,2]), 'audit one SM or the complete ordered pair')
    require(not args.out.exists(), 'refusing to replace an earlier audit')
    source = Path(__file__).read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    try:
        result = qualify(args.target,args.frozen,args.native_receipt,args.device_receipt,tuple(args.clusters))
        require(digest(Path(__file__)) == source_sha, 'auditor source changed during checking')
    except (ValueError,KeyError,OSError) as error:
        result = dict(schema='radiance-full-decoder-qualified-v1',passed=False,error=str(error))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    snapshot = args.out.parent/'audit-sources'/(source_sha+'.py')
    snapshot.parent.mkdir(exist_ok=True)
    if snapshot.exists():
        require(snapshot.read_bytes() == source, 'auditor source snapshot changed')
    else:
        with snapshot.open('xb') as output: output.write(source)
    result['qualifier_source_sha256'] = source_sha
    result['qualifier_source_snapshot'] = str(snapshot.resolve())
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result.get(key) for key in ('passed','model','qualified_sm_counts','error')}))
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
