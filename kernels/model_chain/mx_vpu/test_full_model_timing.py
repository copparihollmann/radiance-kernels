"""Timing acceptance needs complete ordered execution and real output readbacks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from full_model_timing import summarize
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


if __name__ == '__main__':
    unittest.main()
