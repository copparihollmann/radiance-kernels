"""Timing acceptance needs complete ordered execution and real output readbacks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from full_model_timing import cluster_initialization, prepare, report, model_workloads, make_job, observe, process_snapshot, runtime_library_pin, save_timing, summarize
from run_e2e_queue import validate


class FullModelTiming(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.journal = self.root / 'progress.jsonl'
        self.case = dict(num_clusters=2, expected_stages=2,
            stages=[dict(stage='prefill.embedding'), dict(stage='decode0.norm')])
        self.events = [dict(event='stage_entry', cluster=cluster, cycle=cycle,
            symbols=[f'_Z7stage_{index}Pvjjj'])
            for cluster, index, cycle in [(0, 0, 10), (1, 0, 12), (0, 1, 20), (1, 1, 25)]]
        self.events.append(dict(event='finish', cycle=40, passed=True))

    def write_journal(self, events):
        self.journal.write_text(''.join(json.dumps(event) + '\n' for event in events))

    def test_section_intervals_reconcile_and_clusters_are_not_summed(self):
        self.write_journal(self.events)
        result = summarize(self.journal, self.case)
        self.assertEqual(result['global_finish_cycle'], 40)
        self.assertEqual(result['clusters'][0]['section_cycles'], {'prefill': 10, 'decode0': 20})
        self.assertEqual(result['clusters'][1]['section_cycles'], {'prefill': 13, 'decode0': 15})
        for cluster in result['clusters']:
            self.assertEqual(sum(cluster['section_cycles'].values()),
                cluster['cycles_from_first_stage_to_global_finish'])

    def test_incomplete_reordered_missing_cluster_and_reversed_time_are_rejected(self):
        bad = []
        failed = copy.deepcopy(self.events)
        failed[-1]['passed'] = False
        bad.append(failed)
        bad.append([event for event in self.events if event.get('cluster') != 1])
        bad.append([self.events[0], *self.events[2:]])
        wrong = copy.deepcopy(self.events)
        wrong[2]['symbols'] = ['_Z7stage_0Pvjjj']
        bad.append(wrong)
        wrong_time = copy.deepcopy(self.events)
        wrong_time[2]['cycle'] = 9
        bad.append(wrong_time)
        for events in bad:
            with self.subTest(events=events):
                self.write_journal(events)
                with self.assertRaises(ValueError):
                    summarize(self.journal, self.case)

    def test_smolvla_named_stage_symbols_follow_the_graph_order(self):
        case = dict(self.case, expected_stage_order=['stage_v_camera0_patch_embed', 'stage_v_denoise0_euler'])
        events = copy.deepcopy(self.events)
        for event in events[:-1]:
            index = 0 if 'stage_0' in event['symbols'][0] else 1
            event['symbols'] = ['_Z32' + case['expected_stage_order'][index] + 'Pvjjj']
        self.write_journal(events)
        self.assertTrue(summarize(self.journal, case)['completed'])
        events[2]['symbols'] = events[0]['symbols']
        self.write_journal(events)
        with self.assertRaises(ValueError):
            summarize(self.journal, case)

    def test_successful_tohost_cannot_hide_wrong_order_or_missing_second_output(self):
        self.write_journal(self.events)
        outputs = self.root / 'outputs'
        outputs.mkdir()
        checks = []
        for cluster in range(2):
            name = f'{cluster}.bin'
            (outputs / name).write_bytes(b'correct')
            checks.append(dict(file=name, bytes=7, mode='sha256',
                expected_sha256=hashlib.sha256(b'correct').hexdigest()))
        (self.root / 'manifest.json').write_text(json.dumps(dict(
            schema='radiance-portable-workloads-v1',
            workloads=[dict(name='model-2sm', checks=checks, smem_reports=[])])))
        job = dict(kind='isa', num_clusters=2, expected_stages=2,
            expected_stage_order=[0, 1], progress=str(self.journal),
            portable_bundle=str(self.root), portable_workload='model-2sm',
            portable_results=str(outputs))
        log = 'cluster 0 isa-test passed with tohost=0\ncluster 1 isa-test passed with tohost=0\n'
        self.assertTrue(validate(job, log, 0)['passed'])
        changed = copy.deepcopy(self.events)
        changed[2]['symbols'] = ['_Z7stage_0Pvjjj']
        self.write_journal(changed)
        self.assertFalse(validate(job, log, 0)['passed'])
        self.write_journal(self.events)
        (outputs / '1.bin').unlink()
        self.assertFalse(validate(job, log, 0)['passed'])


class TimingObservation(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        script = self.root / 'full_model_timing.py'
        script.write_text('import time; time.sleep(60)\n')
        self.command = [sys.executable, str(script), '--root', str(self.root)]
        self.worker = self.child()
        (self.root / 'supervisor.json').write_text(json.dumps(dict(
            pid=self.worker.pid, status='running', current_case='model-1sm',
            current_attempt=str(self.root / 'attempt-001'))))
        (self.root / 'timing-summary.json').write_text(json.dumps(dict(
            completed_cases=0, expected_cases=8, passed=False)))

    def child(self):
        process = subprocess.Popen(self.command)
        def cleanup():
            if process.poll() is None:
                os.kill(process.pid, signal.SIGCONT)
                process.terminate()
                process.wait(timeout=5)
        self.addCleanup(cleanup)
        return process

    def test_exited_worker_does_not_look_live_from_receipt(self):
        self.worker.terminate()
        self.worker.wait(timeout=5)
        status = observe(self.root)
        self.assertFalse(status['worker']['confirmed_live'])
        self.assertEqual(status['worker_status'], 'running_receipt_without_confirmed_worker')
        self.assertFalse(status['full_model_latency_established'])

    def test_pid_with_a_different_command_does_not_confirm_identity(self):
        self.assertTrue(process_snapshot(self.worker.pid, self.command)['confirmed_live'])
        changed = process_snapshot(self.worker.pid, [*self.command, 'another-attempt'])
        self.assertFalse(changed['confirmed_live'])

    def test_paused_simulator_and_live_serial_control_are_visible(self):
        simulator = self.child()
        control = self.child()
        attempt = self.root / 'attempt-001'
        (attempt / 'queue').mkdir(parents=True)
        job = dict(pid=simulator.pid, command=self.command, status='running', name='model')
        (attempt / 'queue/results.json').write_text(json.dumps(dict(jobs=[job])))
        controls = self.root / 'controls.json'
        controls.write_text(json.dumps(dict(jobs=[dict(job, pid=control.pid, name='projection')])))
        (attempt / 'progress.jsonl').write_text(
            json.dumps(dict(event='heartbeat', cycle=100, stage_entries=17)) + '\n' +
            json.dumps(dict(event='stage_entry', cycle=101, symbols=['stage_next'])) + '\n' +
            '{"event":"heartbeat"')
        os.kill(simulator.pid, signal.SIGSTOP)
        os.waitpid(simulator.pid, os.WUNTRACED)
        status = observe(self.root, [controls])
        self.assertTrue(status['worker']['confirmed_live'])
        self.assertTrue(status['simulators'][0]['paused'])
        self.assertEqual(status['live_serial_controls'][0]['name'], 'projection')
        self.assertEqual(status['stage_entries_observed'], 18)
        self.assertEqual(status['current_cycle'], 101)
        self.assertEqual(status['completed_cases'], 0)


class CompletedTimingIdentity(unittest.TestCase):
    """Output validation success must not allow a different case/engine label."""
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.old = self.root / 'old-simulator'
        self.new = self.root / 'new-simulator'
        self.elf = self.root / 'device.elf'
        for path in [self.old, self.new, self.elf]:
            path.write_bytes(path.name.encode())
        self.sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        self.case = dict(name='tinyllama-1sm', num_clusters=1, expected_stages=1,
                         expected_stage_order=[0], expected_gemmini={},
                         prepared_timing_configuration_sha256={},
                         stages=[dict(stage='prefill.embedding')])
        self.source = self.root / self.case['name']
        self.source.mkdir()
        self.row = dict(name=self.case['name'], device_elf=dict(path='device.elf', sha256=self.sha(self.elf)),
                        loads=[], checks=[])
        (self.source / 'workload.json').write_text(json.dumps(self.row))
        (self.root / 'manifest.json').write_text(json.dumps(dict(workloads=[self.row])))
        self.attempt = self.source / 'attempt-001'
        (self.attempt / 'queue').mkdir(parents=True)
        self.queue = self.attempt / 'queue/results.json'
        self.queue.write_text('{}')
        self.config = self.attempt / 'cyclotron.toml'
        self.config.write_text('[sim]\ntiming=true\n')
        (self.attempt / 'progress.jsonl').write_text(
            json.dumps(dict(event='stage_entry', cluster=0, cycle=10, symbols=['_Z7stage_0Pvjjj'])) + '\n' +
            json.dumps(dict(event='finish', cycle=40, passed=True)) + '\n')
        self.hbm = self.attempt / 'hbm.json'
        self.hbm.write_text(json.dumps(dict(source_cycles=43, source_ticks_per_period=1, memory_ticks_per_period=1,
                                           line_bytes=32, sets=512, ways=8)))
        self.plan = dict(bundle=str(self.root), simulator=str(self.old),
                         simulator_sha256=self.sha(self.old), source_ticks=1, memory_ticks=1)
        self.job = dict(name=self.case['name'], kind='isa', execution_mode='coupled_timing_hbm',
                        num_clusters=1, expected_stages=1, expected_stage_order=[0], expected_gemmini={},
                        prepared_timing_configuration_sha256={},
                        command=[str(self.old), str(self.config), '--binary-path', str(self.elf), '--gen-trace', 'false'],
                        portable_bundle=str(self.root), portable_workload=self.case['name'],
                        input_sha256={str(p): self.sha(p) for p in [self.old, self.elf, self.config]},
                        simulated_cycles=41, hbm_report=str(self.hbm))

    def save(self, plan=None, case=None, job=None):
        # The seam here is attribution after an independently validated job;
        # numerical/output validation itself is covered by the queue tests.
        with patch('full_model_timing.validated_job', return_value=job or self.job):
            save_timing(self.root, plan or self.plan, case or self.case, self.queue)

    def test_case_override_keeps_actual_engine_instead_of_batch_default(self):
        plan = dict(self.plan, simulator=str(self.new), simulator_sha256=self.sha(self.new))
        case = dict(self.case, simulator_pin=dict(path=str(self.old), sha256=self.sha(self.old)))
        self.save(plan, case)
        result = json.loads((self.source / 'result.json').read_text())
        self.assertEqual(result['simulator_sha256'], self.sha(self.old))
        self.assertEqual(result['simulator'], str(self.old))
        self.assertFalse(result['memory_model']['private_data_caches'])
        self.assertEqual(result['memory_model']['shared_cache'], dict(line_bytes=32, sets=512, ways=8))
        self.assertIn('bypassed', result['memory_model']['scope'])

    def test_new_job_uses_and_hashes_its_case_engine_override(self):
        (self.source / 'config-template.toml').write_text('[sim]\nreport="hbm.json"\n')
        attempt = self.source / 'attempt-002'
        attempt.mkdir()
        plan = dict(self.plan, simulator=str(self.new), simulator_sha256=self.sha(self.new))
        case = dict(self.case, simulator_pin=dict(path=str(self.old), sha256=self.sha(self.old)))
        job = make_job(self.root, plan, case, attempt)
        self.assertEqual(job['command'][0], str(self.old))
        self.assertIn(str(self.old), job['hash_inputs'])
        self.assertNotIn(str(self.new), job['hash_inputs'])

    def test_retained_case_keeps_actual_bundle_and_manifest(self):
        newer = self.root / 'new-bundle'
        newer.mkdir()
        (newer / 'manifest.json').write_text('{"workloads": []}')
        plan = dict(self.plan, bundle=str(newer), manifest_sha256=self.sha(newer / 'manifest.json'))
        pin = dict(path=str(self.root), manifest_sha256=self.sha(self.root / 'manifest.json'))
        case = dict(self.case, bundle_pin=pin)
        job = copy.deepcopy(self.job)
        job['input_sha256'][str(self.root / 'manifest.json')] = pin['manifest_sha256']
        self.save(plan, case, job)
        result = json.loads((self.source / 'result.json').read_text())
        self.assertEqual(result['bundle_pin'], pin)
        with self.assertRaisesRegex(ValueError, 'different workload bundle'):
            self.save(plan, self.case, job)
        job['input_sha256'][str(self.root / 'manifest.json')] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'different bundle manifest'):
            self.save(plan, case, job)
        (self.root / 'manifest.json').write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, 'case bundle changed'):
            self.save(plan, case, job)

    def test_new_job_uses_retained_bundle_for_elf_inputs_and_checker(self):
        (self.source / 'config-template.toml').write_text('[sim]\nreport="hbm.json"\n')
        newer = self.root / 'new-bundle'
        newer.mkdir()
        (newer / 'manifest.json').write_text('{"workloads": []}')
        plan = dict(self.plan, bundle=str(newer), manifest_sha256=self.sha(newer / 'manifest.json'))
        case = dict(self.case, bundle_pin=dict(path=str(self.root),
                                             manifest_sha256=self.sha(self.root / 'manifest.json')))
        attempt = self.source / 'attempt-002'
        attempt.mkdir()
        job = make_job(self.root, plan, case, attempt)
        self.assertEqual(job['portable_bundle'], str(self.root))
        self.assertEqual(job['command'][3], str(self.elf))
        self.assertIn(str(self.root / 'manifest.json'), job['hash_inputs'])
        self.assertNotIn(str(newer / 'manifest.json'), job['hash_inputs'])

    def test_a_passed_job_cannot_be_relabelled_with_another_default_engine(self):
        plan = dict(self.plan, simulator=str(self.new), simulator_sha256=self.sha(self.new))
        with self.assertRaises(ValueError):
            self.save(plan)
        self.assertFalse((self.source / 'result.json').exists())

    def test_wrong_case_topology_elf_accelerator_counts_or_clocks_are_rejected(self):
        variants = [dict(self.job, name='other-model-1sm'), dict(self.job, num_clusters=2),
                    dict(self.job, expected_gemmini=dict(mx_loops=999)),
                    dict(self.job, portable_workload='other-model-1sm'),
                    dict(self.job, execution_mode='functional')]
        wrong_elf = copy.deepcopy(self.job)
        wrong_elf['command'][3] = str(self.new)
        variants.append(wrong_elf)
        wrong_configuration = copy.deepcopy(self.job)
        wrong_configuration['prepared_timing_configuration_sha256'] = {'wrong-clocks.toml': '0' * 64}
        variants.append(wrong_configuration)
        for job in variants:
            with self.subTest(job=job), self.assertRaises(ValueError):
                self.save(job=job)
        self.assertFalse((self.source / 'result.json').exists())

    def test_changed_pinned_engine_is_rejected_before_result_creation(self):
        self.old.write_bytes(b'changed simulator')
        with self.assertRaises(ValueError):
            self.save()
        self.assertFalse((self.source / 'result.json').exists())

    def test_clock_ratio_cannot_be_relabelled_by_a_new_plan(self):
        with self.assertRaisesRegex(ValueError, 'clock ratio'):
            self.save(dict(self.plan, source_ticks=2))
        self.assertFalse((self.source / 'result.json').exists())


class RuntimeLibraryBinding(unittest.TestCase):
    def test_each_job_loads_and_hashes_the_pinned_library_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / 'runtime/libramulator.so'
            library.parent.mkdir()
            library.write_bytes(b'original runtime')
            case = dict(name='model-1sm', num_clusters=1, expected_stages=1,
                        expected_stage_order=[0], expected_gemmini={},
                        prepared_timing_configuration_sha256={})
            source = root / case['name']
            source.mkdir()
            (source / 'workload.json').write_text(json.dumps(dict(
                device_elf=dict(path='device.elf'), loads=[], checks=[])))
            (source / 'config-template.toml').write_text('[sim]\nreport="hbm.json"\n')
            attempt = source / 'attempt-001'
            attempt.mkdir()
            simulator = root / 'simulator'
            simulator.write_bytes(b'simulator')
            plan = dict(bundle=str(root), simulator=str(simulator),
                        simulator_sha256=hashlib.sha256(simulator.read_bytes()).hexdigest(),
                        runtime_library=runtime_library_pin(library))
            job = make_job(root, plan, case, attempt)
            self.assertIn(str(library), job['hash_inputs'])
            self.assertEqual(job['env']['LD_LIBRARY_PATH'], str(library.parent))
            library.write_bytes(b'changed runtime')
            second = source / 'attempt-002'
            second.mkdir()
            with self.assertRaisesRegex(ValueError, 'runtime library changed'):
                make_job(root, plan, case, second)
            self.assertFalse((second / 'cyclotron.toml').exists())

    def test_missing_or_misnamed_library_cannot_be_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['libramulator.so', 'wrong-name.so']:
                path = root / name
                if name != 'libramulator.so':
                    path.write_bytes(b'runtime')
                with self.assertRaises(ValueError):
                    runtime_library_pin(path)



class FourSMTimingPreparation(unittest.TestCase):
    def test_native_configuration_keeps_physical_tags_and_discards_assistance(self):
        from workload_bundle import Store
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle = root / 'bundle'
            bundle.mkdir()
            store = Store(bundle)
            def blob(name, data):
                path = root / name
                path.write_bytes(data)
                return store.put(path)
            tags = [[[0x80038, 0xc1d00000 | c], [0x8003c, 0xc1d00000 | c]] for c in range(4)]
            cfg = ('[muon]\nnum_lanes=16\nnum_warps=8\nnum_cores=2\ncluster_id_bits=1\n'
                   '[sim]\nfunctional_assist="old-assist.json"\nsmem_init_per_cluster=' +
                   json.dumps(tags) + '\n')
            model = dict(model='tinyllama', stages=1,
                         stage_backends=[dict(stage='prefill.embedding')])
            row = dict(name='tinyllama-4sm', family='model', num_clusters=4, loads=[], checks=[],
                       smem_init=[[0x80000, 1]], smem_init_per_cluster=tags,
                       device_elf=blob('device', b'qualified-device'),
                       sources=dict(manifest=blob('model', json.dumps(model).encode()),
                                    qualified_simulator_configuration=blob('config', cfg.encode())))
            summary = dict(models=[dict(model='tinyllama', mx_loops_per_sm=1,
                                        mesh_tiles_per_sm=2, vpu_commands_per_sm=1)])
            manifest = dict(workloads=[row], evidence=dict(
                all_full_models=blob('summary', json.dumps(summary).encode())))
            (bundle / 'manifest.json').write_text(json.dumps(manifest))
            cyclotron = root / 'cyclotron'
            for name in ['radiance/gmem.toml', 'radiance/icache.toml', 'lsu_scheduler.toml',
                         'smem.toml', 'writeback_operand_fetch.toml', 'barrier.toml',
                         'fence.toml', 'dma_tensor.toml', 'radiance/execute.toml',
                         'radiance/hazards.toml']:
                path = cyclotron / 'config/timing' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# pinned timing control\n')
            hbm = cyclotron / 'configs/hbm/hbm2-4ch.yaml'
            hbm.parent.mkdir(parents=True)
            hbm.write_text('pinned HBM control\n')
            simulator = root / 'simulator'
            simulator.write_bytes(b'pinned engine')
            with patch('full_model_timing.model_workloads', return_value=[row]):
                plan = prepare(bundle, cyclotron, simulator, root / 'timing', 100, 1, 1)
            config = tomllib.loads((root / 'timing/tinyllama-4sm/config-template.toml').read_text())
            self.assertEqual(config['muon']['cluster_id_bits'], 1)
            self.assertEqual(config['sim']['smem_init_per_cluster'], tags)
            self.assertEqual(config['sim']['num_clusters'], 4)
            self.assertTrue(config['sim']['timing'])
            self.assertNotIn('functional_assist', config['sim'])
            self.assertNotIn('gmem_patches', config['sim'])
            self.assertEqual(plan['cases'][0]['expected_gemmini']['mx_loops'], 1)
            with self.assertRaises(ValueError):
                cluster_initialization(dict(row, smem_init_per_cluster=tags[:3]), {})
            with self.assertRaises(ValueError):
                cluster_initialization(row, dict(sim=dict(smem_init_per_cluster=tags[::-1])))


class TimingCoverage(unittest.TestCase):
    def test_four_sm_timing_requires_every_model_and_rejects_duplicates(self):
        models = ('tinyllama', 'deepseek_r1_distill_qwen_1_5b',
                  'gemma_2_2b_it', 'smolvla_base')
        rows = [dict(name=f'{model}-{n}sm', family='model', num_clusters=n)
                for model in models for n in (1, 2, 4)]
        self.assertEqual(len(model_workloads(dict(workloads=rows))), 12)
        self.assertEqual(len(model_workloads(dict(workloads=[
            row for row in rows if row['num_clusters'] != 4]))), 8)
        for bad in (rows[:-1], rows + [rows[-1]],
                    [dict(row, num_clusters=8) if row == rows[-1] else row for row in rows],
                    [dict(row, name='unknown-4sm') if row == rows[-1] else row for row in rows]):
            with self.assertRaises(ValueError):
                model_workloads(dict(workloads=bad))


class TimingReportCoverage(unittest.TestCase):
    def test_all_twelve_cases_are_required_and_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            cases=[dict(name=f'{model}-{n}sm',num_clusters=n)
                   for model in ('tinyllama','deepseek','gemma','smol') for n in (1,2,4)]
            (root/'plan.json').write_text(json.dumps(dict(cases=cases,source_ticks=1,memory_ticks=1)))
            for case in cases:
                d=root/case['name'];d.mkdir()
            for index, case in enumerate(cases[:8]):
                (root/case['name']/'result.json').write_text(json.dumps(dict(
                    global_finish_cycle=99,simulated_cycles=100,hbm_source_cycles_after_drain=101,
                    simulator=f'engine-{index % 2}',simulator_sha256=str(index % 2)*64)))
            partial=report(root)
            self.assertFalse(partial['passed'])
            self.assertEqual((partial['completed_cases'],partial['expected_cases']),(8,12))
            self.assertEqual([row['simulator_sha256'] for row in partial['rows']],
                             [str(index % 2)*64 for index in range(8)] + [None]*4)
            (root/'plan.json').write_text(json.dumps(dict(cases=cases[:8],source_ticks=1,memory_ticks=1)))
            legacy=report(root)
            self.assertTrue(legacy['passed'])
            self.assertEqual(legacy['expected_cases'],8)
            (root/'plan.json').write_text(json.dumps(dict(cases=cases,source_ticks=1,memory_ticks=1)))
            for case in cases[8:]:
                (root/case['name']/'result.json').write_text(json.dumps(dict(
                    global_finish_cycle=99,simulated_cycles=100,hbm_source_cycles_after_drain=101)))
            completed=report(root)
            self.assertTrue(completed['passed'])
            self.assertEqual((completed['completed_cases'],completed['expected_cases']),(12,12))
            self.assertEqual(len((root/'cycles.csv').read_text().splitlines()),13)

if __name__ == '__main__':
    unittest.main()
