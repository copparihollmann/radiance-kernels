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
import unittest

from full_model_timing import observe, process_snapshot, summarize
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


if __name__ == '__main__':
    unittest.main()
