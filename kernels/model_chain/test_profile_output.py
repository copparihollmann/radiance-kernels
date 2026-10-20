import json
import unittest
from profile_output import summarize, parse_log, add_lane_trace
import os
from pathlib import Path
import tempfile


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

    def test_embedded_context_completion_mx_clocks_and_host_are_separate(self):
        context = self.context | dict(mx_useful_macs=25600, mx_instances=2, mx_peak_macs_per_cycle=256)
        lines = ['RADIANCE_CONTEXT '+json.dumps(context)] + [
            'RADIANCE_PROFILE '+json.dumps(r) for r in self.rows] + [
            'RADIANCE_HOST '+json.dumps(dict(setup_cycles=100000, wait_cycles=200000,
                                           setup_instructions=10000,wait_instructions=20000)),
            'RADIANCE_DONE '+json.dumps(dict(workload='test-2sm', sms=2, device_code=0))]
        result = parse_log(lines, mx_clock_ratio=0.5)
        self.assertEqual(result['total']['work_cycles'], 110)
        self.assertAlmostEqual(result['total']['mx_utilization'],25600/(110*.5*256*2))
        self.assertEqual(result['host']['setup_cycles'],100000)
        self.assertTrue(result['complete_run_log'])
        with self.assertRaises(ValueError): parse_log(lines+lines)
        with self.assertRaises(ValueError): parse_log(lines+['RADIANCE_FAILURE {"reason":"timeout"}'])

    def test_zero_instruction_counter_is_not_a_measured_zero_ipc(self):
        rows=[r | dict(instructions_end=r['instructions_begin']) for r in self.rows]
        with self.assertRaises(ValueError): self.run_rows(rows)

    def test_complete_issue_trace_lane_efficiency_is_scoped(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as directory:
            path=Path(directory)/'trace.csv'
            text='cycle,sm,core,tmask\n'
            for r in self.rows:
                text+=f'{r["cycle_begin"]},'+f'{r["sm"]},{r["core"]},0xff\n'
                text+=f'{r["cycle_end"]},'+f'{r["sm"]},{r["core"]},0xffff\n'
            path.write_text(text)
            result=self.run_rows();add_lane_trace(result,path)
            self.assertEqual(result['total']['lane_efficiency'],.5)

    def test_optional_work_memory_and_vpu_counters(self):
        lines=['RADIANCE_PROFILE '+json.dumps(r) for r in self.rows]+[
            'RADIANCE_MEMORY '+json.dumps(dict(scope='work',source='test counters',read_bytes=100,write_bytes=10)),
            'RADIANCE_ENGINE '+json.dumps(dict(scope='work',source='test counters',engine='vpu',busy_cycles=20,capacity_cycles=100))]
        result=parse_log(lines,self.context,gpu_clock_hz=1000)
        self.assertEqual(result['total']['memory_bandwidth_bytes_per_second'],1000)
        self.assertEqual(result['total']['vpu_utilization'],.2)


if __name__ == '__main__': unittest.main()
