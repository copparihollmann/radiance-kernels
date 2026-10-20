import json
import unittest
from profile_output import summarize


class ProfileOutputTests(unittest.TestCase):
    def setUp(self):
        self.context = dict(schema=1, workload='test-2sm', sm_count=2, cores_per_sm=2,
            backend='cyclotron-coupled', memory_model='hbm2', clock_domain='muon',
            device_elf_sha256='a'*64, common_cycle_epoch=True,
            instruction_counter_semantics='issued_warp_instructions', peak_issue_per_core_per_cycle=1)
        self.rows = [dict(schema=1, sm=sm, core=core, cycle_begin=(1 << 33) + sm*10,
            cycle_end=(1 << 33) + 100 + sm*10,
            instructions_begin=(1 << 34), instructions_end=(1 << 34) + 25)
            for sm in range(2) for core in range(2)]

    def run_rows(self, rows=None, context=None):
        return summarize(['unrelated console message\n'] + [
            'RADIANCE_PROFILE ' + json.dumps(r) + '\n' for r in (self.rows if rows is None else rows)],
            self.context if context is None else context)

    def test_64bit_and_global_epoch_include_launch_skew(self):
        result = self.run_rows()
        self.assertEqual(result['total']['work_cycles'], 110)
        self.assertEqual(result['total']['instructions'], 100)
        self.assertEqual(result['total']['sum_core_cycles'], 400)
        self.assertAlmostEqual(result['total']['simt_issue_utilization'], 100/440)

    def test_no_global_epoch_no_claimed_whole_program_duration(self):
        context = self.context | dict(common_cycle_epoch=False)
        result = self.run_rows(context=context)
        self.assertIsNone(result['total']['work_cycles'])
        self.assertIsNone(result['total']['ipc'])
        self.assertEqual(result['total']['mean_core_ipc'], 0.25)

    def test_unsupported_counters_stay_null(self):
        context = self.context | dict(instruction_counter_semantics='unavailable', peak_issue_per_core_per_cycle=None)
        rows = [r | dict(instructions_begin=None, instructions_end=None) for r in self.rows]
        result = self.run_rows(rows, context)
        for key in ['instructions', 'ipc', 'simt_issue_utilization', 'mx_utilization', 'vpu_utilization', 'lane_efficiency']:
            self.assertIsNone(result['total'][key])
        with self.assertRaises(ValueError):
            self.run_rows(context=context)

    def test_truncated_duplicate_wrong_topology_and_reset_rejected(self):
        for rows in [self.rows[:-1], self.rows + [self.rows[0]],
                     [self.rows[0] | dict(sm=4)] + self.rows[1:],
                     [self.rows[0] | dict(cycle_end=20)] + self.rows[1:]]:
            with self.assertRaises(ValueError): self.run_rows(rows)
        with self.assertRaises(ValueError):
            summarize(['RADIANCE_PROFILE {'], self.context)
        for malformed in ['[]', 'null', '12', '{"schema":true}']:
            with self.assertRaises(ValueError):
                summarize(['RADIANCE_PROFILE ' + malformed], self.context)

    def test_retired_counter_cannot_be_used_as_issue_utilization(self):
        with self.assertRaises(ValueError):
            self.run_rows(context=self.context | dict(instruction_counter_semantics='retired_warp_instructions'))

    def test_bool_nan_and_missing_identity_rejected(self):
        for field, value in [('sm_count', True), ('peak_issue_per_core_per_cycle', float('nan')),
                             ('common_cycle_epoch', None), ('device_elf_sha256', 'unknown')]:
            with self.assertRaises(ValueError): self.run_rows(context=self.context | {field: value})


if __name__ == '__main__': unittest.main()
