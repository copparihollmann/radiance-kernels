"""Exercise portable selection/install and reject misleading console receipts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT=Path(__file__).with_name('debug.py')


class PortableDebug(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        elf=self.root/'host-htif-2sm-logical.elf'
        elf.write_bytes(b'fixture ELF')
        self.row=dict(name='host-htif-2sm-logical',sm_count=2,stage='host-htif',loader='logical',
            boot_elf=elf.name,sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),expected_pass='PASS host-htif')
        (self.root/'workloads.json').write_text(json.dumps(dict(workloads=[self.row])))
        configs=self.root/'firesim-workloads';configs.mkdir()
        (configs/(self.row['name']+'.json')).write_text(json.dumps(dict(benchmark_name=self.row['name'],common_bootbinary='boot.elf')))
        self.log=self.root/'uartlog'

    def call(self,*args):
        return subprocess.run([sys.executable,str(SCRIPT),'--bundle',str(self.root),*args],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)

    def check(self,text,exit_code=0):
        self.log.write_text(text)
        return self.call('check','--sms','2','--stage','host-htif','--loader','logical',
            '--log',str(self.log),'--firesim-exit-code',str(exit_code)).returncode

    def test_install_complete_selected_workload(self):
        deploy=self.root/'deploy'
        p=self.call('install','--sms','2','--stage','host-htif','--loader','logical','--deploy-dir',str(deploy))
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual((deploy/'workloads'/self.row['name']/'boot.elf').read_bytes(),b'fixture ELF')
        self.assertTrue((deploy/'workloads'/(self.row['name']+'.json')).is_file())

    def test_reject_modified_elf(self):
        (self.root/self.row['boot_elf']).write_bytes(b'changed')
        p=self.call('install','--sms','2','--stage','host-htif','--loader','logical','--deploy-dir',str(self.root/'deploy'))
        self.assertNotEqual(p.returncode,0)

    def test_accept_matching_completed_console(self):
        self.assertEqual(self.check('[boot-debug] HOST_BOOT stage=0x0000000000000000 sms=0x0000000000000002\n[boot-debug] PASS host-htif\n'),0)

    def test_reject_wrong_stage_wrong_sm_trap_or_failed_exit(self):
        banner='[boot-debug] HOST_BOOT stage=0x0000000000000000 sms=0x0000000000000002\n'
        log=banner+'[boot-debug] PASS host-htif\n'
        for bad in [log.replace('stage=0x0000000000000000','stage=0x0000000000000002'),
            log.replace('sms=0x0000000000000002','sms=0x0000000000000004'),
            log+'[boot-debug] HOST_TRAP\n',log+'[boot-debug] FAIL\n',banner]:
            with self.subTest(log=bad): self.assertNotEqual(self.check(bad),0)
        self.assertNotEqual(self.check(log,1),0)


if __name__=='__main__': unittest.main()
