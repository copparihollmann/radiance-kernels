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


if __name__ == '__main__':
    unittest.main()
