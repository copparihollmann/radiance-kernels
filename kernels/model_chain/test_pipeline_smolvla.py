"""Numerical checks for SmolVLA vision and attention index mappings."""

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent


class SmolVLAPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        source = Path(cls.temp.name) / "wrapper.cpp"
        library = Path(cls.temp.name) / "pipeline.so"
        source.write_text('''
#include "pipeline_smolvla.hpp"
extern "C" void run_patch(const float* image, const float* weight,
                          const float* bias, const float* position, float* out) {
  model_chain::patch_embed(image, weight, bias, position, out,
                           4, 2, 3, 2, 0, 1);
}
extern "C" void run_shuffle(const float* input, float* out) {
  model_chain::pixel_shuffle(input, out, 4, 2, 2, 0, 1);
}
extern "C" void run_attention(const float* q, const float* k, const float* v,
                              const uint32_t* mask, float* out, float* scratch) {
  model_chain::masked_gqa(q, k, v, mask, out, scratch, 2, 3, 2, 1, 2, 0, 1);
}
''')
        subprocess.run(["clang++", "-O2", "-nostdlib++", "-shared", "-fPIC",
                        "-I", str(HERE), str(source), "-o", str(library)], check=True)
        cls.library = ctypes.CDLL(str(library))
        f32 = ctypes.POINTER(ctypes.c_float)
        cls.library.run_patch.argtypes = [f32] * 5
        cls.library.run_shuffle.argtypes = [f32, f32]
        cls.library.run_attention.argtypes = [f32, f32, f32,
                                              ctypes.POINTER(ctypes.c_uint32), f32, f32]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def ptr(value):
        return value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def test_patch_embedding_uses_nchw_and_checkpoint_filter_order(self):
        rng = np.random.default_rng(41)
        image = rng.standard_normal((3, 4, 4)).astype(np.float32)
        weight = rng.standard_normal((2, 3, 2, 2)).astype(np.float32)
        bias = rng.standard_normal(2).astype(np.float32)
        position = rng.standard_normal((4, 2)).astype(np.float32)
        actual = np.empty((4, 2), dtype=np.float32)
        self.library.run_patch(*(self.ptr(value) for value in
                                 (image, weight, bias, position, actual)))
        expected = np.empty_like(actual)
        for row in range(2):
            for col in range(2):
                patch = image[:, row * 2:row * 2 + 2, col * 2:col * 2 + 2]
                expected[row * 2 + col] = np.einsum("chw,ochw->o", patch, weight)
        expected += bias + position
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_pixel_shuffle_matches_upstream_reshape_order(self):
        source = np.arange(4 * 4 * 2, dtype=np.float32).reshape(1, 16, 2)
        actual = np.empty((1, 4, 8), dtype=np.float32)
        self.library.run_shuffle(self.ptr(source), self.ptr(actual))
        expected = source.reshape(1, 4, 4, 2)
        expected = expected.reshape(1, 4, 2, 4).transpose(0, 2, 1, 3)
        expected = expected.reshape(1, 2, 2, 8).transpose(0, 2, 1, 3)
        expected = expected.reshape(1, 4, 8)
        np.testing.assert_array_equal(actual, expected)

    def test_masked_grouped_query_attention(self):
        rng = np.random.default_rng(42)
        q = rng.standard_normal((2, 2, 2)).astype(np.float32)
        k = rng.standard_normal((3, 1, 2)).astype(np.float32)
        v = rng.standard_normal((3, 1, 2)).astype(np.float32)
        mask = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint32)
        actual = np.empty((2, 4), dtype=np.float32)
        scratch = np.empty(3, dtype=np.float32)
        self.library.run_attention(self.ptr(q), self.ptr(k), self.ptr(v),
                                   mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                                   self.ptr(actual), self.ptr(scratch))
        expected = np.empty_like(actual)
        for query in range(2):
            for head in range(2):
                scores = (k[:, 0] @ q[query, head]) / np.sqrt(2.0)
                scores = scores[:query + 1]
                weights = np.exp(scores - scores.max())
                weights /= weights.sum()
                expected[query, head * 2:head * 2 + 2] = weights @ v[:query + 1, 0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
