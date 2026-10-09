"""Independent NumPy checks for the shared-buffer math compiled by Muon."""

import ctypes
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPECS = json.loads((HERE / "../evaluation/llm/pr1-models.json").read_text())


class PipelineMathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        source = Path(cls.temp.name) / "wrapper.cpp"
        library = Path(cls.temp.name) / "pipeline.so"
        source.write_text('''
#include "pipeline_math.hpp"
extern "C" void run_chain(const float* x, const float* gamma, const float* weight,
                          const float* bias, const float* skip, float* normalized,
                          float* projected, float* output,
                          unsigned m, unsigned k, unsigned n, float epsilon) {
  model_chain::rmsnorm(x, gamma, normalized, m, k, epsilon, 0, 1);
  model_chain::linear(normalized, weight, bias, projected, m, k, n, 0, 1);
  model_chain::residual(projected, skip, output, m * n, 0, 1);
}
extern "C" float run_exp(float x) { return model_chain::exponential(x); }
extern "C" float run_tanh(float x) { return model_chain::hyperbolic_tangent(x); }
''')
        subprocess.run(["clang++", "-nostdlib++", "-O2", "-shared", "-fPIC", "-I", str(HERE),
                        str(source), "-o", str(library)], check=True)
        cls.run_chain = ctypes.CDLL(str(library)).run_chain
        ptr = ctypes.POINTER(ctypes.c_float)
        cls.run_chain.argtypes = [ptr] * 8 + [ctypes.c_uint32] * 3 + [ctypes.c_float]
        cls.run_exp = ctypes.CDLL(str(library)).run_exp
        cls.run_exp.argtypes = [ctypes.c_float]
        cls.run_exp.restype = ctypes.c_float
        cls.run_tanh = ctypes.CDLL(str(library)).run_tanh
        cls.run_tanh.argtypes = [ctypes.c_float]
        cls.run_tanh.restype = ctypes.c_float

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def check_shape(self, width, output_width):
        rng = np.random.default_rng(width + output_width)
        m = 2
        x = (rng.standard_normal((m, width)) * 0.1).astype(np.float32)
        gamma = (1 + rng.standard_normal(width) * 0.01).astype(np.float32)
        weight = (rng.standard_normal((width, output_width)) /
                  np.sqrt(width)).astype(np.float32)
        bias = (rng.standard_normal(output_width) * 0.01).astype(np.float32)
        skip = (rng.standard_normal((m, output_width)) * 0.1).astype(np.float32)
        normalized = np.empty_like(x)
        projected = np.empty_like(skip)
        output = np.empty_like(skip)
        arrays = (x, gamma, weight, bias, skip, normalized, projected, output)
        pointers = [array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for array in arrays]
        self.run_chain(*pointers, m, width, output_width, 1e-5)
        expected_norm = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5) * gamma
        expected_projected = expected_norm @ weight + bias
        np.testing.assert_allclose(normalized, expected_norm, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(projected, expected_projected, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(output, expected_projected + skip, rtol=3e-5, atol=3e-6)

    def test_four_model_widths(self):
        shapes = [(SPECS["tinyllama"]["hidden_size"], 64),
                  (SPECS["deepseek_r1_distill_qwen_1_5b"]["hidden_size"], 128),
                  (SPECS["gemma_2_2b_it"]["hidden_size"], 256),
                  (SPECS["smolvla_base"]["vlm_hidden_size"], 64)]
        for width, output_width in shapes:
            with self.subTest(width=width, output_width=output_width):
                self.check_shape(width, output_width)

    def test_software_transcendentals(self):
        values = np.linspace(-20, 10, 301, dtype=np.float32)
        result = np.array([self.run_exp(float(x)) for x in values])
        np.testing.assert_allclose(result, np.exp(values), rtol=3e-6, atol=1e-8)
        values = np.linspace(-5, 5, 201, dtype=np.float32)
        result = np.array([self.run_tanh(float(x)) for x in values])
        np.testing.assert_allclose(result, np.tanh(values), rtol=3e-5, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
