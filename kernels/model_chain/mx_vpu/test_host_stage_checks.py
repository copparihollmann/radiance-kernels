"""Require every actual GPU tensor and both replicas, with frozen references."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from host_stage_checks import prepare, check
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from compile_decoder import generate


class HostStageChecks(unittest.TestCase):
    def test_oracles_leave_gpu_source_and_second_replica_failures_are_rejected(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder)
            old=generate('tinyllama',1,2,1,'greedy',root/'old',device_check_all_stages=True,stages_per_object=20)
            target=generate('tinyllama',1,2,1,'greedy',root/'new',device_check_all_stages=True,stages_per_object=20,batch_clusters=2)
            refs=prepare(target,old/'model_data.cpp')
            self.assertNotIn('v_gold_', (target/'model_data.device.cpp').read_text())
            self.assertNotIn('verify_chunk_', (target/'kernel.device.cpp').read_text())
            self.assertIn('v_decode0_token_ids[replica*16u]', (target/'kernel.device.cpp').read_text())
            makefile = (target/'Makefile').read_text()
            self.assertIn('MU_SRCS = kernel.device.cpp',makefile)
            self.assertIn('model_data.device.cpp',makefile)
            self.assertNotIn('verify_chunk_',makefile)
            self.assertIn('kernel.radiance.elf: kernel.device.radiance.elf',makefile)
            self.assertEqual(refs['stages'],41)
            actual=root/'actual'
            (actual/'tensors').mkdir(parents=True)
            for replica in range(2):
                for row in refs['records']:
                    (actual/'tensors'/f'{replica}-{row["symbol"]}.bin').write_bytes(Path(row['path']).read_bytes())
            self.assertTrue(check(target,actual,2)['passed'])
            for kind in ('float32','int32'):
                row=next(x for x in refs['records']if x['dtype']==kind)
                path=actual/'tensors'/f'1-{row["symbol"]}.bin'
                original=path.read_bytes()
                data=np.frombuffer(original,dtype='<f4'if kind=='float32'else'<i4').copy()
                data[-1]+=1000
                path.write_bytes(data.tobytes())
                self.assertTrue(check(target,actual,1)['passed'])
                self.assertFalse(check(target,actual,2)['passed'])
                path.write_bytes(original)
                path.unlink()
                self.assertFalse(check(target,actual,2)['passed'])
                path.write_bytes(original)
            path=Path(refs['records'][0]['path'])
            path.write_bytes(b'changed oracle')
            with self.assertRaisesRegex(ValueError,'reference changed'):
                check(target,actual,2)


if __name__=='__main__':unittest.main()
