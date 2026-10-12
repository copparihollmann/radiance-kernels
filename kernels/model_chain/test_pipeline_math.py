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
#include "pipeline_int8.hpp"
#include "pipeline_tied.hpp"
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
extern "C" void run_f16_linear(const float* x, const uint16_t* weight,
                                 float* out, unsigned m, unsigned k, unsigned n) {
  model_chain::linear_f16(x, weight, out, m, k, n, 0, 1);
}
extern "C" void run_f16_embedding(const int32_t* ids, const uint16_t* table,
                                    float* out, unsigned count, unsigned width, float scale) {
  model_chain::embedding_f16(ids, table, out, count, width, scale, 0, 1);
}
extern "C" float run_f16_convert(uint16_t bits) {
  return model_chain::fp16_to_fp32(bits);
}
extern "C" void run_i8_linear(const float* x, const int8_t* weight,
                                 const float* scales, float* out,
                                 unsigned m, unsigned k, unsigned n) {
  model_chain::linear_i8(x, weight, scales, out, m, k, n, 0, 1);
}
extern "C" void run_i8_embedding(const int32_t* ids, const int8_t* table,
                                    const float* scales, float* out,
                                    unsigned count, unsigned width, float factor) {
  model_chain::embedding_i8(ids, table, scales, out, count, width, factor, 0, 1);
}
extern "C" void run_f16_tied(const float* x, const uint16_t* table,
                                float* out, unsigned m, unsigned hidden,
                                unsigned vocab) {
  model_chain::linear_f16_tied(x, table, out, m, hidden, vocab, 0, 1);
}
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
        cls.run_f16_linear = ctypes.CDLL(str(library)).run_f16_linear
        cls.run_f16_linear.argtypes = [ptr, ctypes.POINTER(ctypes.c_uint16), ptr,
                                       ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        cls.run_f16_embedding = ctypes.CDLL(str(library)).run_f16_embedding
        cls.run_f16_embedding.argtypes = [ctypes.POINTER(ctypes.c_int32),
                                          ctypes.POINTER(ctypes.c_uint16), ptr,
                                          ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float]
        cls.run_f16_convert = ctypes.CDLL(str(library)).run_f16_convert
        cls.run_f16_convert.argtypes = [ctypes.c_uint16]
        cls.run_f16_convert.restype = ctypes.c_float
        cls.run_i8_linear = ctypes.CDLL(str(library)).run_i8_linear
        cls.run_i8_linear.argtypes = [ptr, ctypes.POINTER(ctypes.c_int8), ptr, ptr,
                                      ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        cls.run_i8_embedding = ctypes.CDLL(str(library)).run_i8_embedding
        cls.run_i8_embedding.argtypes = [ctypes.POINTER(ctypes.c_int32),
                                         ctypes.POINTER(ctypes.c_int8), ptr, ptr,
                                         ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float]
        cls.run_f16_tied = ctypes.CDLL(str(library)).run_f16_tied
        cls.run_f16_tied.argtypes = [ptr, ctypes.POINTER(ctypes.c_uint16), ptr,
                                     ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]

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

    def test_fp16_weight_storage_and_conversion(self):
        rng = np.random.default_rng(71)
        x = rng.standard_normal((2, 64)).astype(np.float32)
        weight = rng.standard_normal((64, 16)).astype(np.float16)
        output = np.empty((2, 16), dtype=np.float32)
        self.run_f16_linear(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            weight.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            2, 64, 16)
        np.testing.assert_allclose(output, x @ weight.astype(np.float32),
                                   rtol=1e-5, atol=1e-5)
        ids = np.array([0, 3], dtype=np.int32)
        table = rng.standard_normal((4, 64)).astype(np.float16)
        embedded = np.empty((2, 64), dtype=np.float32)
        self.run_f16_embedding(ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                               table.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                               embedded.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               2, 64, 0.5)
        np.testing.assert_array_equal(embedded, table[ids].astype(np.float32) * 0.5)
        for value in (0.0, -0.0, 1.0, -3.25, np.float16(2 ** -20)):
            half = np.float16(value)
            self.assertEqual(self.run_f16_convert(int(half.view(np.uint16))),
                             float(half))

    def test_int8_weight_storage_with_channel_scales(self):
        from export_decoder_int8 import quantize as quantize_per_channel

        rng = np.random.default_rng(81)
        x = rng.standard_normal((2, 16)).astype(np.float32)
        weight = rng.standard_normal((16, 8)).astype(np.float32)
        packed, scales, _ = quantize_per_channel(weight, "linear")
        output = np.empty((2, 8), dtype=np.float32)
        self.run_i8_linear(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           packed.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                           scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           2, 16, 8)
        np.testing.assert_allclose(output,
                                   x @ (packed.astype(np.float32) * scales),
                                   rtol=1e-5, atol=1e-5)
        table = rng.standard_normal((4, 16)).astype(np.float32)
        packed, scales, _ = quantize_per_channel(table, "embedding")
        ids = np.array([0, 3], dtype=np.int32)
        embedded = np.empty((2, 16), dtype=np.float32)
        self.run_i8_embedding(ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                              packed.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                              scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              embedded.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              2, 16, 0.5)
        np.testing.assert_array_equal(embedded,
                                      packed[ids].astype(np.float32) * scales[ids, None] * 0.5)

    def test_tied_fp16_embedding_and_output_projection_share_storage(self):
        rng = np.random.default_rng(91)
        input_values = rng.standard_normal((2, 8)).astype(np.float32)
        table = rng.standard_normal((5, 8)).astype(np.float16)
        output = np.empty((2, 5), dtype=np.float32)
        self.run_f16_tied(input_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          table.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                          output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          2, 8, 5)
        np.testing.assert_allclose(output, input_values @ table.astype(np.float32).T,
                                   rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
