"""Compare the C++ spectral path with NumPy's orthonormal real 2D FFT."""

import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent


class SpectralMathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary_root = HERE.parents[2] / ".tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        source = Path(cls.temporary.name) / "spectral_test.cpp"
        library = Path(cls.temporary.name) / "spectral_test.so"
        source.write_text('''
#include "spectral.hpp"
extern "C" void run_spectral(const float* input, const float* filter,
                              const float* cos_h, const float* sin_h,
                              const float* cos_w, const float* sin_w,
                              float* output, unsigned h, unsigned w,
                              unsigned channels) {
  const unsigned bins = w / 2 + 1, size = h * bins * channels * 2;
  float* a = new float[size]; float* b = new float[size];
  for (unsigned tid = 0; tid < 32; ++tid)
    spectformer::real_dft_rows(input, a, cos_w, sin_w, h, w, channels, tid, 32);
  for (unsigned tid = 0; tid < 32; ++tid)
    spectformer::complex_dft_columns(a, b, cos_h, sin_h, h, bins, channels,
                                      false, tid, 32);
  for (unsigned tid = 0; tid < 32; ++tid)
    spectformer::complex_filter(b, filter, a, h * bins * channels, tid, 32);
  for (unsigned tid = 0; tid < 32; ++tid)
    spectformer::complex_dft_columns(a, b, cos_h, sin_h, h, bins, channels,
                                      true, tid, 32);
  for (unsigned tid = 0; tid < 32; ++tid)
    spectformer::inverse_real_dft_rows(b, output, cos_w, sin_w, h, w,
                                        channels, tid, 32);
  delete[] a; delete[] b;
}
extern "C" void run_attention(const float* qkv, float* output,
                                unsigned tokens, unsigned channels,
                                unsigned heads) {
  const unsigned threads = 4;
  float* scratch = new float[threads * tokens];
  for (unsigned tid = 0; tid < threads; ++tid)
    spectformer::attention_qkv(qkv, output, scratch, tokens, channels,
                                heads, tid, threads);
  delete[] scratch;
}
''')
        subprocess.run(["g++", "-O2", "-shared", "-fPIC", "-I", str(HERE),
                        "-I", str(HERE.parent / "model_chain"),
                        str(source), "-o", str(library)], check=True)
        cls.function = ctypes.CDLL(str(library)).run_spectral
        cls.function.argtypes = [ctypes.POINTER(ctypes.c_float)] * 7 + [
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
        cls.attention = ctypes.CDLL(str(library)).run_attention
        cls.attention.argtypes = [ctypes.POINTER(ctypes.c_float)] * 2 + [
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_spatial_grids(self):
        rng = np.random.default_rng(8421)
        for h, w, c in ((14, 14, 3), (28, 28, 2), (56, 56, 1)):
            with self.subTest(shape=(h, w, c)):
                x = rng.normal(size=(h, w, c)).astype(np.float32)
                weights = (rng.normal(size=(h, w // 2 + 1, c)) + 1j *
                           rng.normal(size=(h, w // 2 + 1, c))).astype(np.complex64)
                values = np.arange(h)
                cos_h = np.cos(2 * np.pi * np.outer(values, values) / h).astype(np.float32)
                sin_h = np.sin(2 * np.pi * np.outer(values, values) / h).astype(np.float32)
                values = np.arange(w)
                cos_w = np.cos(2 * np.pi * np.outer(values, values) / w).astype(np.float32)
                sin_w = np.sin(2 * np.pi * np.outer(values, values) / w).astype(np.float32)
                expected = np.fft.irfft2(
                    np.fft.rfft2(x, axes=(0, 1), norm="ortho") * weights,
                    s=(h, w), axes=(0, 1), norm="ortho").astype(np.float32)
                actual = np.empty_like(x)
                arrays = (x, weights.view(np.float32), cos_h, sin_h,
                          cos_w, sin_w, actual)
                ptrs = [a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        for a in arrays]
                self.function(*ptrs, h, w, c)
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)

    def test_attention_softmax_matches_numpy(self):
        rng = np.random.default_rng(8421)
        tokens, channels, heads = 7, 8, 2
        qkv = (rng.normal(size=(tokens, 3, channels)) * 2).astype(np.float32)
        expected = np.empty((tokens, channels), dtype=np.float32)
        width = channels // heads
        for head in range(heads):
            span = slice(head * width, (head + 1) * width)
            q, k, v = (qkv[:, index, span].astype(np.float64)
                       for index in range(3))
            scores = q @ k.T / np.sqrt(width)
            scores -= scores.max(axis=1, keepdims=True)
            weights = np.exp(scores)
            weights /= weights.sum(axis=1, keepdims=True)
            expected[:, span] = weights @ v
        actual = np.empty_like(expected)
        ptr = ctypes.POINTER(ctypes.c_float)
        self.attention(qkv.ctypes.data_as(ptr), actual.ctypes.data_as(ptr),
                       tokens, channels, heads)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
