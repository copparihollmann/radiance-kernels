import copy
from pathlib import Path
import unittest

from workloads import describe, run_plan, select


def row(name, family, sms, partitions=None, status='passed', **extra):
    return dict(name=name, family=family, num_clusters=sms,
                logical_partitions=partitions, device_elf={}, soc_elf={}, loads=[], checks=[],
                software_qualification=dict(status=status), **extra)


class WorkloadSelectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(workloads=[
            row('tinyllama-2sm', 'model', 2),
            row('tinyllama-4sm', 'model', 4, status='pending'),
            row('stream--copy--2', 'hpc', 1, 2),
            row('stream--copy--2sm', 'hpc', 2, 2),
            row('layer0', 'model', 2, kind='layer', model='tinyllama'),
            row('gemm-4sm', 'gemm', 4, 4)])

    def test_physical_sms_do_not_follow_partition_suffix(self):
        selected = select(self.manifest, kinds=['hpc'], sms=[2])
        self.assertEqual([item['name'] for _, item in selected], ['stream--copy--2sm'])
        self.assertEqual(describe(self.manifest['workloads'][2])['sm_count'], 1)

    def test_full_graph_does_not_include_explicit_layer(self):
        selected = select(self.manifest, kinds=['full_model'], models=['tinyllama'])
        self.assertEqual([item['name'] for _, item in selected],
                         ['tinyllama-2sm', 'tinyllama-4sm'])

    def test_pass_filter_excludes_pending_replacements(self):
        selected = select(self.manifest, kinds=['full_model'], status='passed')
        self.assertEqual([item['name'] for _, item in selected], ['tinyllama-2sm'])

    def test_duplicate_unknown_and_conflicting_topology_rejected(self):
        with self.assertRaises(ValueError):
            select(self.manifest, names=['missing'])
        bad = copy.deepcopy(self.manifest)
        bad['workloads'].append(copy.deepcopy(bad['workloads'][0]))
        with self.assertRaises(ValueError):
            select(bad)
        bad = copy.deepcopy(self.manifest)
        bad['workloads'][0]['sm_count'] = 4
        with self.assertRaises(ValueError):
            select(bad)

    def test_unsafe_names_and_unknown_categories_rejected(self):
        for value in [row('../escape', 'hpc', 2), row('unknown', 'unknown', 2)]:
            with self.assertRaises(ValueError):
                describe(value)

    def test_portable_plan_uses_argument_arrays_and_keeps_board_contract(self):
        selected = select(self.manifest, names=['tinyllama-2sm'])
        original = copy.deepcopy(selected)
        plan = run_plan(Path('bundle with spaces'), selected, Path('runs'), 'direct-hashed',
                        python='python3')
        self.assertEqual(selected, original)
        job = plan['jobs'][0]
        self.assertEqual(job['sm_count'], 2)
        self.assertIn(str(Path('bundle with spaces/workload_bundle.py').resolve()), job['prepare_argv'])
        self.assertIn('--hash-size', job['prepare_argv'])
        self.assertIsNone(job['board_run_argv'])
        self.assertEqual(job['timing_status'], 'pending')
        self.assertFalse(job['fpga_execution_verified'])


if __name__ == '__main__':
    unittest.main()
