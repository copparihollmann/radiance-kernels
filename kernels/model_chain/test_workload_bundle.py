import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

from workload_bundle import Store, check_outputs, digest, local, patched_load, verify, write_json


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'bundle'; self.bundle.mkdir()
        self.results = self.root / 'results'; self.results.mkdir()

    def test_relocation_dedup_and_corruption(self):
        src = self.root / 'source'; src.write_bytes(b'checkpoint')
        store = Store(self.bundle)
        one = store.put(src)
        alias = self.root / 'alias'; shutil.copyfile(src, alias)
        self.assertEqual(one, store.put(alias))
        write_json(self.bundle / 'manifest.json', dict(workloads=[dict(name='x', loads=[one])]))
        from workload_bundle import digest
        (self.bundle / 'SHA256SUMS').write_text(''.join(
            f'{digest(p)}  {p.relative_to(self.bundle)}\n'
            for p in self.bundle.rglob('*') if p.is_file()))
        moved = self.root / 'moved'; self.bundle.rename(moved)
        self.assertTrue(verify(moved)['passed'])
        (moved / one['path']).write_bytes(b'corruption')
        with self.assertRaises(ValueError): verify(moved)
        with self.assertRaises(ValueError): local(moved, '../source')

    def test_full_stage_readbacks_missing_replica_and_nonfinite(self):
        src = self.root / 'reference'; src.write_bytes(struct.pack('<2f', 1., 2.))
        ref = Store(self.bundle).put(src)
        checks = [dict(file=f'{c}-tensor.bin', address=4096+c*8, bytes=8,
                       mode='close', dtype='float32', atol=.0005, rtol=.005,
                       reference=ref) for c in range(2)]
        write_json(self.bundle/'manifest.json', dict(workloads=[dict(name='model', checks=checks)]))
        (self.results/'0-tensor.bin').write_bytes(src.read_bytes())
        self.assertFalse(check_outputs(self.bundle, 'model', self.results)['passed'])
        (self.results/'1-tensor.bin').write_bytes(src.read_bytes())
        self.assertTrue(check_outputs(self.bundle, 'model', self.results)['passed'])
        (self.results/'1-tensor.bin').write_bytes(struct.pack('<2f', 1., float('nan')))
        self.assertFalse(check_outputs(self.bundle, 'model', self.results)['passed'])

    def test_gemm_requires_output_and_both_mailboxes(self):
        src = self.root/'golden'; src.write_bytes(struct.pack('<H', 0x3f80) * (512*512))
        ref = Store(self.bundle).put(src)
        write_json(self.bundle/'manifest.json', dict(workloads=[dict(name='gemm',
            checks=[dict(file='output.bin', bytes=src.stat().st_size, mode='gemm', dtype='bf16', reference=ref)],
            smem_reports=[dict(file=f'mailbox-{c}.bin') for c in range(2)])]))
        (self.results/'output.bin').write_bytes(src.read_bytes())
        vals=[0x5ef1d0e5,512*512,512*512,0,0,0,0,0]
        data=struct.pack('<16I', *[v for v in vals for _ in range(2)])
        (self.results/'mailbox-0.bin').write_bytes(data)
        self.assertFalse(check_outputs(self.bundle,'gemm',self.results)['passed'])
        (self.results/'mailbox-1.bin').write_bytes(data)
        self.assertTrue(check_outputs(self.bundle,'gemm',self.results)['passed'])
        (self.results/'output.bin').write_bytes(b'\xee\xee' * (512*512))
        self.assertFalse(check_outputs(self.bundle,'gemm',self.results)['passed'])

    def test_initialization_overlays_materialized_and_checked(self):
        import hashlib
        src=self.root/'base'; src.write_bytes(b'a'*64)
        patch=self.root/'patch'; patch.write_bytes(b'b'*16)
        load=dict(path=str(src),address=4096,bytes=64)
        spec=dict(path=str(patch),address=4112,bytes=16,
                  expected_old_sha256=hashlib.sha256(b'a'*16).hexdigest(),
                  expected_new_sha256=digest(patch))
        store=Store(self.bundle)
        blob=patched_load(store,load,[spec])
        self.assertEqual((self.bundle/blob['path']).read_bytes(),b'a'*16+b'b'*16+b'a'*32)
        self.assertEqual(src.read_bytes(),b'a'*64)
        spec['expected_old_sha256']='0'*64
        with self.assertRaises(ValueError):patched_load(store,load,[spec])


if __name__ == '__main__': unittest.main()
