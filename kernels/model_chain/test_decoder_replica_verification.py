"""Run the generated sharded device verifiers against both buffer replicas."""
import ctypes
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from compile_decoder import generate


class ReplicaVerifierTests(unittest.TestCase):
    def test_second_replica_float_and_token_corruption_are_detected(self):
        self.check_replica_verifiers(2)

    def test_fourth_replica_float_and_token_corruption_are_detected(self):
        self.check_replica_verifiers(4)

    def check_replica_verifiers(self, replicas):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = generate('tinyllama', 1, 2, 1, 'greedy', root,
                              batch_clusters=replicas, device_check_all_stages=True,
                              stages_per_object=20)
            # Only replace the hardware tohost transport. The actual generated
            # verifier functions, tensor layouts and comparisons run unchanged.
            (root/'mu_intrinsics.h').write_text('#pragma once\n#define __global\n')
            (root/'kernel_verify.h').write_text('''#pragma once
#include <stdint.h>
extern uint32_t last_tohost;
inline void mu_tohost(uint32_t code){last_tohost=code;}
inline bool mu_close(float a,float b,float r,float t){
 return __builtin_fabsf(a-b)<=r*__builtin_fabsf(b)+t;
}
''')
            shards = sorted(target.glob('verify_chunk_*.cpp'))
            self.assertGreater(len(shards), 1)
            bridge = '#include <stdint.h>\nuint32_t last_tohost=0;\n'
            for i in range(len(shards)):
                bridge += f'bool verify_chunk_{i:03d}(uint32_t);\n'
            bridge += 'extern "C" bool run_verify(uint32_t replica){return '
            bridge += ' && '.join(f'verify_chunk_{i:03d}(replica)'for i in range(len(shards)))+';}\n'
            (root/'bridge.cpp').write_text(bridge)
            library = root/'verifier.so'
            subprocess.run(['g++','-O2','-shared','-fPIC','-I'+str(root),
                            str(target/'model_data.cpp'),*map(str,shards),
                            str(root/'bridge.cpp'),'-o',str(library)],check=True)
            lib = ctypes.CDLL(str(library))
            lib.run_verify.argtypes = [ctypes.c_uint32]
            lib.run_verify.restype = ctypes.c_bool
            floats, integers = [], []
            pattern = (r'i < (\d+);[^\n]*\n?\s*if \('
                       r'(?:!mu_close\()?([\w]+)\[i \+ replica \* (\d+)u\]'
                       r'(?:, | != )([\w]+)\[i\]')
            for path in shards:
                source = path.read_text()
                for size,out,stride,gold in re.findall(pattern,source):
                    size,stride = int(size),int(stride)
                    integer = f'extern __global int32_t {out}[];' in source
                    scalar = ctypes.c_int32 if integer else ctypes.c_float
                    actual = (scalar*(replicas*stride)).in_dll(lib,out)
                    expected = (scalar*size).in_dll(lib,gold)
                    for replica in range(replicas):
                        for index in range(size):
                            actual[replica*stride+index] = expected[index]
                    (integers if integer else floats).append((actual,stride))
            self.assertTrue(floats and integers)
            for replica in range(replicas):
                self.assertTrue(lib.run_verify(replica))
            for actual,stride in (floats[0],integers[0]):
                last = (replicas - 1) * stride
                old = actual[last]
                actual[last] = old+10000
                for replica in range(replicas - 1):
                    self.assertTrue(lib.run_verify(replica))
                self.assertFalse(lib.run_verify(replicas - 1))
                self.assertNotEqual(ctypes.c_uint32.in_dll(lib,'last_tohost').value,0)
                actual[last] = old
                self.assertTrue(lib.run_verify(replicas - 1))


if __name__ == '__main__':
    unittest.main()
