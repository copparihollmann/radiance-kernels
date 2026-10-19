"""Acceptance guards must reject incomplete outputs even after successful tohost."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from check_fp8_gemm import check
from run_e2e_queue import validate


class CompleteOutputChecks(unittest.TestCase):
    def test_two_sm_hpc_requires_both_completions_and_core_traffic(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as folder:
            d=Path(folder);out=d/'out';out.write_bytes(b'correct')
            report=d/'hbm.json'
            hbm=dict(unconsumed_requests=0,pending_dram=0,sources={str(c):1 for c in range(4)})
            report.write_text(json.dumps(hbm))
            job=dict(kind='hpc',num_clusters=2,output=str(out),expected_bytes=7,
                     expected_output_sha256=hashlib.sha256(b'correct').hexdigest(),
                     hbm_report=str(report),expected_hbm_core_sources=[0,1,2,3])
            log='cluster 0 isa-test passed with tohost=0\ncluster 1 isa-test passed with tohost=0\n'
            self.assertTrue(validate(job,log,0)['passed'])
            self.assertFalse(validate(job,log.splitlines()[0],0)['passed'])
            self.assertFalse(validate(job,log.replace('cluster 1','cluster 0'),0)['passed'])
            hbm['sources']['2']=0;report.write_text(json.dumps(hbm))
            self.assertFalse(validate(job,log,0)['passed'])

    def test_coupled_hbm_needs_timing_report_binding_and_matching_clocks(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as folder:
            d=Path(folder);config=d/'cyclotron.toml';report=d/'hbm.json';journal=d/'progress.jsonl'
            journal.write_text(json.dumps(dict(event='stage_entry',cluster=0))+'\n'+
                               json.dumps(dict(event='finish',passed=True))+'\n')
            text=('[sim]\ntiming=true\n[gemmini]\nenabled=true\nmode="both"\npreset="radiance"\n'
                  '[timing.gmem.hbm]\nconfig="memory.yaml"\nreport="hbm.json"\nsource_ticks=3\nmemory_ticks=2\n')
            config.write_text(text)
            hbm=dict(unconsumed_requests=0,pending_dram=0,source_ticks_per_period=3,memory_ticks_per_period=2)
            report.write_text(json.dumps(hbm))
            job=dict(kind='isa',execution_mode='coupled_timing_hbm',num_clusters=1,expected_stages=1,
                     progress=str(journal),command=['simulator',str(config)],hbm_report=str(report))
            memory=d/'memory.yaml';memory.write_bytes(b'frozen memory configuration')
            job['prepared_timing_configuration_sha256']={str(memory):hashlib.sha256(memory.read_bytes()).hexdigest()}
            log='cluster 0 isa-test passed with tohost=0'
            self.assertTrue(validate(job,log,0)['passed'])
            memory.write_bytes(b'changed configuration with same clock ratio')
            self.assertFalse(validate(job,log,0)['passed'])
            memory.write_bytes(b'frozen memory configuration')
            output=d/'output.bin';output.write_bytes(b'complete output')
            hpc=dict(job,kind='hpc',output=str(output),expected_bytes=15,
                     expected_output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertTrue(validate(hpc,log,0)['passed'])
            config.write_text(text.replace('timing=true','timing=false'))
            self.assertFalse(validate(hpc,log,0)['passed'])
            for bad in (text.replace('timing=true','timing=false'),text.replace('enabled=true','enabled=false'),
                        text.replace('"both"','"func"'),text.replace('"radiance"','"e4m3_vpu"'),
                        text.replace('report="hbm.json"','report="other.json"'),
                        text.replace('source_ticks=3','source_ticks=2'),
                        text.replace('source_ticks=3','source_ticks=0')):
                config.write_text(bad)
                self.assertFalse(validate(job,log,0)['passed'])
            config.write_text(text)
            report.write_text(json.dumps(dict(hbm,memory_ticks_per_period=1)))
            self.assertFalse(validate(job,log,0)['passed'])

    def test_host_oracles_require_actual_gpu_outputs_from_every_replica(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as folder:
            d = Path(folder)
            manifest = d/'manifest.json'
            manifest.write_text(json.dumps(dict(host_stage_checks=True)))
            reference = d/'gold.bin'
            reference.write_bytes(bytes.fromhex('0000803f'))  # FP32 1.0
            refs = dict(stages=1,rtol=.005,atol=.0005,records=[dict(
                symbol='output',dtype='float32',elements=1,path=str(reference),
                sha256=hashlib.sha256(reference.read_bytes()).hexdigest())])
            (d/'host-stage-references.json').write_text(json.dumps(refs))
            journal = d/'progress.jsonl'
            journal.write_text(''.join(json.dumps(row)+'\n'for row in (
                dict(event='stage_entry',cluster=0),dict(event='stage_entry',cluster=1),
                dict(event='finish',passed=True))))
            job = dict(kind='isa',num_clusters=2,expected_stages=1,
                       progress=str(journal),model_manifest=str(manifest))
            log = ''.join(f'cluster {i} isa-test passed with tohost=0\n'for i in range(2))
            self.assertFalse(validate(job,log,0)['passed'])
            job['host_stage_result_dir'] = str(d)
            (d/'tensors').mkdir()
            (d/'tensors/0-output.bin').write_bytes(reference.read_bytes())
            self.assertFalse(validate(job,log,0)['passed'])
            second = d/'tensors/1-output.bin'
            second.write_bytes(reference.read_bytes())
            self.assertTrue(validate(job,log,0)['passed'])
            second.write_bytes(bytes.fromhex('00000040'))  # FP32 2.0
            self.assertFalse(validate(job,log,0)['passed'])

    def test_fourth_sm_missing_or_wrong_stage_output_cannot_pass(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as folder:
            d=Path(folder);manifest=d/'manifest.json'
            manifest.write_text(json.dumps(dict(host_stage_checks=True)))
            ref=d/'gold.bin';ref.write_bytes(bytes.fromhex('0000803f'))
            (d/'host-stage-references.json').write_text(json.dumps(dict(stages=1,rtol=.005,atol=.0005,
                records=[dict(symbol='output',dtype='float32',elements=1,path=str(ref),
                              sha256=hashlib.sha256(ref.read_bytes()).hexdigest())])))
            journal=d/'progress.jsonl';journal.write_text(''.join(json.dumps(row)+'\n'for row in (
                *[dict(event='stage_entry',cluster=c)for c in range(4)],dict(event='finish',passed=True))))
            (d/'tensors').mkdir()
            for c in range(4):(d/'tensors'/f'{c}-output.bin').write_bytes(ref.read_bytes())
            job=dict(kind='isa',num_clusters=4,expected_stages=1,progress=str(journal),
                     model_manifest=str(manifest),host_stage_result_dir=str(d))
            log=''.join(f'cluster {c} isa-test passed with tohost=0\n'for c in range(4))
            self.assertTrue(validate(job,log,0)['passed'])
            last=d/'tensors/3-output.bin';last.unlink()
            self.assertFalse(validate(job,log,0)['passed'])
            last.write_bytes(bytes.fromhex('00000040'))
            self.assertFalse(validate(job,log,0)['passed'])
            last.write_bytes(ref.read_bytes())
            self.assertFalse(validate(job,log.replace('cluster 3','cluster 2'),0)['passed'])

    def test_native_decoder_needs_every_replica_check_and_greedy_token(self):
        job = dict(kind='native', model='tinyllama', expected_stages=41,
                   num_replicas=2, reference_generated_token_ids=[61])
        one = 'generated_token_0=61\nstages=41 check_chunks=3 tohost=0\n'
        log = ''.join(f'REPLICA_BEGIN {i}\n'+one+f'REPLICA_DONE {i} status=0\n'for i in range(2))
        self.assertTrue(validate(job,log,0)['passed'])
        for broken in (log.replace('REPLICA_DONE 1 status=0','REPLICA_DONE 1 status=1'),
                       log.rsplit('generated_token_0=61',1)[0],
                       log.replace('generated_token_0=61','generated_token_0=62',1),
                       log.replace('stages=41','stages=40',1),
                       log.replace('REPLICA_BEGIN 1','REPLICA_BEGIN 0')):
            self.assertFalse(validate(job,broken,0)['passed'])

    def test_gemm_checks_every_element_and_requires_gpu_agreement(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temporary:
            directory = Path(temporary)
            golden = directory / 'golden.bin'
            output = directory / 'output.bin'
            golden.write_bytes(b'\x80\x3f' * (512 * 512))  # BF16 1.0
            output.write_bytes(golden.read_bytes())
            slots = [0x5ef1d0e5, 262144, 262144, 0, 0, 0, 0, 0]
            words = ' '.join(f'{v:08x} {v:08x}' for v in slots)
            log = 'nightly host released verify GO after all END stamps\n'
            for cluster in range(2):
                log += f'cluster {cluster} isa-test passed with tohost=0\ncluster {cluster} smem 0x80080: {words}\n'
            result = check(output, golden, log, 2)
            self.assertTrue(all(v for k, v in result.items() if k.endswith('_pass')))
            damaged = bytearray(output.read_bytes())
            damaged[-2:] = b'\xee\xee'
            output.write_bytes(damaged)
            result = check(output, golden, log, 2)
            self.assertFalse(result['full_output_pass'])
            self.assertEqual(result['poison'], 1)
            output.write_bytes(golden.read_bytes())
            result = check(output, golden, log.replace(words, ' '.join(words.split()[:-2] + ['00000065', '00000065']), 1), 2)
            self.assertFalse(result['gpu_verification_pass'])
            self.assertFalse(result['gpu_reports_agree_pass'])

    def test_isa_needs_success_and_progress_from_every_cluster(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temporary:
            journal = Path(temporary) / 'progress.jsonl'
            records = [dict(event='stage_entry', cluster=i) for i in range(2)] + [dict(event='finish', passed=True)]
            journal.write_text(''.join(json.dumps(r) + '\n' for r in records))
            job = dict(kind='isa', num_clusters=2, progress=str(journal))
            one = 'cluster 0 isa-test passed with tohost=0\n'
            both = one + 'cluster 1 isa-test passed with tohost=0\n'
            self.assertFalse(validate(job, one, 0)['passed'])
            self.assertTrue(validate(job, both, 0)['passed'])
            job['expected_stages'] = 2
            self.assertFalse(validate(job, both, 0)['passed'])

            complete = records[:-1] * 2 + records[-1:]
            journal.write_text(''.join(json.dumps(r) + '\n' for r in complete))
            self.assertTrue(validate(job, both, 0)['passed'])
            journal.write_text(''.join(json.dumps(r) + '\n' for r in records[1:]))
            self.assertFalse(validate(job, both, 0)['passed'])

    def test_mx_loops_and_physical_tiles_are_checked_in_their_own_units(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as folder:
            journal=Path(folder)/'progress.jsonl'
            journal.write_text(json.dumps(dict(event='stage_entry',cluster=0))+'\n'+json.dumps(dict(event='finish',passed=True))+'\n')
            job=dict(kind='isa',num_clusters=1,progress=str(journal),expected_stages=1,
                     expected_gemmini=dict(mx_loops=4608,mesh_tiles=9216,vpu_commands=3072))
            log=('cluster 0 isa-test passed with tohost=0\n'
                 'LOOP_WS pending 0, retired 4608\n'
                 'mesh: 9216 tiles, 147456 busy cycles; VPU: 3072 commands\n')
            self.assertTrue(validate(job,log,0)['passed'])
            for broken in (log.replace('retired 4608','retired 9216'),
                           log.replace('9216 tiles','4608 tiles'),
                           log.replace('3072 commands','3071 commands'),
                           log.replace('pending 0','pending 1')):
                self.assertFalse(validate(job,broken,0)['passed'])

    def test_hpc_zero_exit_needs_complete_oracle_and_drained_hbm(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temporary:
            directory = Path(temporary)
            output = directory / 'output.bin'
            output.write_bytes(b'full output')
            report = directory / 'hbm.json'
            report.write_text(json.dumps(dict(unconsumed_requests=0, pending_dram=0)))
            job = dict(kind='hpc', output=str(output), expected_bytes=11,
                       expected_output_sha256=hashlib.sha256(b'full output').hexdigest(), hbm_report=str(report))
            log = 'isa-test passed with tohost=0'
            self.assertTrue(validate(job, log, 0)['passed'])
            output.write_bytes(b'full outpuX')
            self.assertFalse(validate(job, log, 0)['passed'])
            output.write_bytes(b'full output')
            report.write_text(json.dumps(dict(unconsumed_requests=1, pending_dram=0)))
            self.assertFalse(validate(job, log, 0)['passed'])

    def test_functional_isa_scope_requires_explicit_matching_configuration(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temporary:
            directory=Path(temporary);config=directory/'cyclotron.toml';journal=directory/'progress.jsonl'
            journal.write_text(json.dumps(dict(event='stage_entry',cluster=0))+'\n'+
                               json.dumps(dict(event='finish',passed=True))+'\n')
            text='[sim]\ntiming=false\n[gemmini]\nmode="both"\npreset="radiance"\n'
            config.write_text(text)
            job=dict(kind='isa',execution_mode='functional_core',num_clusters=1,expected_stages=1,
                     progress=str(journal),command=['simulator',str(config)])
            log='cluster 0 isa-test passed with tohost=0'
            self.assertTrue(validate(job,log,0)['passed'])
            for bad in (text.replace('false','true'),text.replace('"both"','"perf"'),
                        text+'[timing.gmem.hbm]\nconfig="hbm.yaml"\n',
                        text.replace('timing=false','timing=false\nfunctional_assist="assist.json"')):
                config.write_text(bad)
                self.assertFalse(validate(job,log,0)['passed'])

    def test_assistance_cannot_be_counted_as_unassisted_or_timed_execution(self):
        import hashlib
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temporary:
            d=Path(temporary);elf=d/'kernel.elf';elf.write_bytes(b'bound ELF bytes')
            config=d/'cyclotron.toml';plan=d/'assist.json';report=d/'assist-result.json';journal=d/'progress.jsonl'
            config.write_text('[sim]\ntiming=false\nfunctional_assist="assist.json"\n[gemmini]\nmode="both"\npreset="radiance"\n')
            journal.write_text(json.dumps(dict(event='stage_entry',cluster=0))+'\n'+json.dumps(dict(event='finish',passed=True))+'\n')
            binding=dict(elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),plugin_sha256='plugin',combined_source_sha256='source')
            plan.write_text(json.dumps(dict(report=str(report),**binding)))
            valid=dict(schema='radiance-pure-fallback-result-v1',completed=True,whole_program_latency_valid=False,
                       includes_hbm=False,statistics=dict(warp_calls=1,operations={'1':dict(work_items=4)}),**binding)
            report.write_text(json.dumps(valid))
            job=dict(kind='isa',execution_mode='assisted_functional_core',num_clusters=1,expected_stages=1,
                     progress=str(journal),command=['simulator',str(config),'--binary-path',str(elf)],assistance_report=str(report))
            log='cluster 0 isa-test passed with tohost=0'
            self.assertTrue(validate(job,log,0)['passed'])
            for mode in ('functional_core',None):
                self.assertFalse(validate(dict(job,execution_mode=mode),log,0)['passed'])
            for changes in (dict(completed=False),dict(elf_sha256='other'),dict(includes_hbm=True),dict(whole_program_latency_valid=True)):
                report.write_text(json.dumps(dict(valid,**changes)))
                self.assertFalse(validate(job,log,0)['passed'])


if __name__ == '__main__':
    unittest.main()
