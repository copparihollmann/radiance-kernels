"""Replica isolation and full-run acceptance, including second-replica failures."""

import ctypes
from pathlib import Path
import subprocess
import struct
import tempfile
import types
import unittest

import numpy as np

from compile_smolvla import emit_stage
from native_replicas import validate_log
from replica_buffers import check_images, plan_replicas
from split_decoder_weights import verify_image_placement


HERE = Path(__file__).resolve().parent


def full_log():
    return ''.join(f'''REPLICA_BEGIN {i}
completed 3673/3673: last
actions=1600 failures=0 first_failure=0 max_abs_error=0.000005
BOUNDARY_TOTAL checked=1930 failed_stages=0
ORIGINAL actions=1600 failures=0 max_abs_error=0.00044632
REPLICA_DONE {i} status=0
''' for i in range(2)) + 'REPLICA_GUARDS failed_bytes=0\n'


class ReplicaTests(unittest.TestCase):
    def test_second_cluster_stack_range_is_reserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'program.elf'
            header = bytearray(52)
            header[:7] = b'\x7fELF\x01\x01\x01'
            struct.pack_into('<I', header, 28, 52)
            struct.pack_into('<HH', header, 42, 32, 1)
            path.write_bytes(header + struct.pack('<8I', 1, 0, 0x10000000, 0, 256, 256, 7, 4096))
            image = dict(segments=[dict(gpu_base_address=0x7ED00000, image_size_bytes=4096)])
            verify_image_placement(path, [image], num_clusters=1)
            with self.assertRaisesRegex(ValueError, 'warp stacks'):
                verify_image_placement(path, [image], num_clusters=2)

    def test_second_replica_and_guard_failures_are_rejected(self):
        good = full_log()
        self.assertTrue(validate_log(good, 2, 3673, True, True)['passed'])
        for broken in (good.replace('REPLICA_DONE 1 status=0', 'REPLICA_DONE 1 status=1'),
                       good.replace('REPLICA_BEGIN 1', 'REPLICA_BEGIN 0'),
                       good.rsplit('REPLICA_BEGIN 1', 1)[0],
                       good.replace('failed_bytes=0', 'failed_bytes=1'),
                       good.replace('checked=1930', 'checked=1929'),
                       good.replace('actions=1600 failures=0', 'actions=1599 failures=0'),
                       good.replace('ORIGINAL actions=1600 failures=0', 'ORIGINAL actions=1600 failures=1')):
            with self.subTest(broken=broken[-100:]):
                self.assertFalse(validate_log(broken, 2, 3673, True, True)['passed'])

    def test_four_sm_pool_fits_below_program_and_reserves_four_stacks(self):
        plan = plan_replicas(47193024, 262144, replicas=4)
        self.assertEqual(plan['bytes'], 4 * plan['stride'])
        self.assertLess(plan['base'] + plan['bytes'], 0x10000000)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'program.elf'
            header = bytearray(52)
            header[:7] = b'\x7fELF\x01\x01\x01'
            struct.pack_into('<I', header, 28, 52)
            struct.pack_into('<HH', header, 42, 32, 1)
            path.write_bytes(header + struct.pack('<8I', 1, 0, 0x10000000, 0, 256, 256, 7, 4096))
            image = dict(segments=[dict(gpu_base_address=0x7E900000, image_size_bytes=4096)])
            verify_image_placement(path, [image], num_clusters=2)
            with self.assertRaisesRegex(ValueError, 'warp stacks'):
                verify_image_placement(path, [image], num_clusters=4)

    def test_pool_rejects_image_overlap_and_overflow(self):
        plan = plan_replicas(47193024, 262144)
        self.assertLess(plan['base'] + plan['bytes'], 0x10000000)
        self.assertEqual(plan['bytes'], 2 * plan['stride'])
        for base, size in ((plan['base'] - 4096, 8192),
                           (plan['base'] + plan['bytes'] - 1, 1)):
            with self.assertRaises(ValueError):
                check_images(plan, [dict(segments=[dict(gpu_base_address=base, image_size_bytes=size)])])
        check_images(plan, [dict(segments=[dict(gpu_base_address=plan['base'] + plan['bytes'], image_size_bytes=4096)])])
        for args in ((1 << 29, 4096), (4096, 0), (-1, 4096)):
            with self.assertRaises(ValueError):
                plan_replicas(*args)

    def test_emitted_attention_uses_separate_context_and_scratch(self):
        self.check_attention_replicas(2)

    def test_four_sm_attention_uses_disjoint_contexts_and_guards(self):
        self.check_attention_replicas(4)

    def check_attention_replicas(self, replicas):
        plan = plan_replicas(256, 32 * 1024 * 4, replicas=replicas)
        graph = types.SimpleNamespace(tensors={})
        allocation = {name: dict(offset_bytes=i*64) for i, name in enumerate(('q', 'k', 'v', 'out'))}
        stage = dict(id='replica_attention', op='bidirectional_mha',
                     attrs=dict(heads=2, head_dim=2), shape=[1, 3, 2, 2],
                     reads=['q', 'k', 'v'], writes='out')
        # Exercise the original body and the explicit helper ABI against the
        # same independent expectation; scratch must stay relative to context.
        for explicit_abi in (False, True):
            source = ('#include <stdint.h>\n#define __global\n#define MU_NUM_THREADS 32\n'
                      '#define MU_NUM_CORES 1\n#include "pipeline_smolvla.hpp"\n'
                      + '#include \"fallback_abi.hpp\"\n'
                      + emit_stage(stage, graph, allocation, {}, replicas=plan, fallback_call_abi=explicit_abi)
                      + 'extern "C" void run(void* p) {for(unsigned i=0;i<32;++i)'
                        'stage_v_replica_attention(p,i,32,0);}\n')
            with tempfile.TemporaryDirectory() as folder:
                cpp, lib = Path(folder)/'control.cpp', Path(folder)/'control.so'
                cpp.write_text(source)
                subprocess.run(['g++', '-O2', '-ffp-contract=off', '-shared', '-fPIC',
                                '-I'+str(HERE), str(cpp), str(HERE/'fallback_runtime.cpp'), '-o', str(lib)], check=True)
                run = ctypes.CDLL(str(lib)).run
                run.argtypes = [ctypes.c_void_p]
                pool = np.full(plan['bytes'], 0xa5, np.uint8)
                for replica in range(replicas):
                    start = replica * plan['stride'] + plan['guard_bytes']
                    body = pool[start:start+plan['body_bytes']]
                    body[:] = 0
                    for i, name in enumerate(('q', 'k', 'v')):
                        values = np.full(12, (replica+1)*(i+1)*0.25, np.float32)
                        body[i*64:i*64+48] = values.view('u1')
                untouched = pool[plan['stride']:].copy()
                run(pool.ctypes.data + plan['guard_bytes'])
                np.testing.assert_array_equal(pool[plan['stride']:], untouched)
                for replica in range(1, replicas):
                    run(pool.ctypes.data + replica * plan['stride'] + plan['guard_bytes'])
                for replica in range(replicas):
                    start = replica * plan['stride'] + plan['guard_bytes']
                    actual = pool[start+192:start+240].view('f4')
                    np.testing.assert_allclose(actual, (replica+1)*0.75, atol=1e-6, rtol=1e-6)
                    np.testing.assert_array_equal(pool[replica*plan['stride']:start], 0xa5)
                    end = start + plan['body_bytes']
                    np.testing.assert_array_equal(pool[end:end+plan['guard_bytes']], 0xa5)


if __name__ == '__main__':
    unittest.main()
