"""Cross-check actual C API MX/VPU commands against independent format math."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mx_vpu import reference, fp32_reference, prepare_weights


class BackendChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archive = os.environ.get("RADIANCE_LIBGEMMINI_ARCHIVE")
        lib = os.environ.get("RADIANCE_LIBGEMMINI_ROOT")
        if not archive or not lib:
            raise unittest.SkipTest("set RADIANCE_LIBGEMMINI_ARCHIVE and RADIANCE_LIBGEMMINI_ROOT")
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.binary = cls.directory / "backend-control"
        command = ["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-no-pie",
                   "-I"+str(HERE.parent), "-I"+lib,
                   str(HERE/"test_backend.cpp"), str(HERE/"runtime.cpp"),
                   archive, "-o", str(cls.binary)]
        subprocess.run(command, check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def check(self, m, k, n, storage, columns=None, prepared=None):
        rng = np.random.default_rng(123+storage)
        x = rng.normal(0, 0.4, (m, k)).astype(np.float32)
        x[0] = 0  # exercise a zero microscaling group and padded tiles
        skip = rng.normal(0, 0.1, (m, n)).astype(np.float32)
        tied = storage >= 5
        shape = (n, k) if tied else (k, n)
        scales = np.empty(0, np.float32)
        if storage in (0, 5):
            w = rng.normal(0, 0.03, shape).astype(np.float32)
            decoded = w
            payload = w.tobytes()
        elif storage in (1, 6):
            w = rng.normal(0, 0.03, shape).astype(np.float16)
            decoded = w.astype(np.float32)
            payload = w.tobytes()
        else:
            bits = {2:8, 3:4, 4:6, 7:8, 8:5}[storage]
            codes = rng.integers(-(1 << (bits-1)), 1 << (bits-1), shape, dtype=np.int8)
            if prepared in (2,3):
                for block in range((k+31)//32):
                    bands = (0,1,2,4,8) if prepared == 2 else (0,1,2,4,8,16,32,64,128)
                    limit = bands[block % len(bands)]
                    codes[block*32:(block+1)*32] = (0 if not limit else
                        rng.integers(-limit,limit,(min(32,k-block*32),n),dtype=np.int8))
            if storage == 8:
                self.assertEqual(k % 64, 0)
                scales = rng.uniform(0.001, 0.01, (n, k//64)).astype(np.float32)
                decoded = codes.astype(np.float32) * np.repeat(scales, 64, axis=1)
            else:
                scales = rng.uniform(0.001, 0.01, n).astype(np.float32)
                if prepared in (2,3):
                    scales[::3] *= -1
                    scales[0] = 0
                decoded = codes.astype(np.float32) * (scales[:, None] if tied else scales)
            if bits == 8:
                payload = codes.tobytes()
            else:
                unsigned = codes.ravel().astype(np.int32) & ((1 << bits)-1)
                packed = np.zeros((unsigned.size*bits+7)//8, np.uint8)
                for i, code in enumerate(unsigned):
                    for b in range(bits):
                        bit = i*bits+b
                        packed[bit//8] |= ((int(code) >> b) & 1) << (bit % 8)
                payload = packed.tobytes()
        expected = reference.linear(x, decoded.T if tied else decoded)
        if prepared:
            if prepared == 1:
                payload,aux = prepare_weights.direct(decoded.T if tied else decoded)
            elif prepared == 2:
                self.assertEqual(storage,3)
                aux = prepare_weights.int4_lookup(codes,scales)
            else:
                self.assertEqual(storage,2)
                aux = prepare_weights.int8_lookup(codes,scales)
            scales = np.frombuffer(aux+b'\0'*((-len(aux))%4),np.float32)
            storage = 8+prepared
        if columns:
            first, count = columns
            ordinary = fp32_reference.linear(x, decoded.T if tied else decoded)
            ordinary[:, first:first+count] = expected[:, first:first+count]
            expected = ordinary
        source = self.directory / "input.bin"
        output = self.directory / "output.bin"
        source.write_bytes(struct.pack("<6I", m,k,n,storage,len(payload),scales.size)
                           + x.tobytes()+payload+scales.tobytes()+skip.tobytes())
        command = [str(self.binary), str(source), str(output)]
        if columns:
            command += [str(v) for v in columns]
        run = subprocess.run(command,
                             capture_output=True, text=True, check=True)
        y, z = np.fromfile(output, np.float32).reshape(2,m,n)
        np.testing.assert_array_equal(y, expected)
        added = reference.add(expected, skip)
        if columns:
            ordinary = expected + skip
            ordinary[:, first:first+count] = added[:, first:first+count]
            added = ordinary
        np.testing.assert_array_equal(z, added)
        self.assertRegex(run.stderr, r"MX_VPU_MODEL commands=\d+ mx=[1-9]\d* vpu=[1-9]\d*")

    def test_odd_shape_and_partial_k(self):
        self.check(17, 47, 19, 0)

    def test_compact_and_tied_weight_formats(self):
        for storage in range(1,9):
            with self.subTest(storage=storage):
                self.check(2, 64, 17, storage)

    def test_partition_keeps_other_columns_fp32(self):
        self.check(17, 47, 35, 0, (3, 17))
        for storage in range(1, 9):
            with self.subTest(storage=storage):
                self.check(3, 64, 35, storage, (16, 8))

    def test_prepared_weights_match_original_format(self):
        self.check(17,47,19,0,prepared=1)
        self.check(2,64,17,2,prepared=1)
        self.check(2,163,19,3,prepared=2)
        self.check(2,289,19,2,prepared=3)


if __name__ == "__main__":
    unittest.main()
