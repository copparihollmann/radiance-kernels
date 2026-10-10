"""Check full-depth checkpoint placement and the quantized NumPy reference."""

import unittest

import numpy as np

from export_decoder_fp16 import CONSOLE_MMIO, STACK_BOTTOM, STACK_TOP, plan
from export_decoder_int8 import plan as plan_int8, quantize
from checkpoint import MixedFP16Weights


class FP16DecoderImageTest(unittest.TestCase):
    def test_full_depth_plans_keep_parameters_out_of_reserved_ranges(self):
        for model, depth in (("tinyllama", 22),
                             ("deepseek_r1_distill_qwen_1_5b", 28)):
            with self.subTest(model=model):
                layout = plan(model, depth)
                self.assertEqual(layout["dtype"], "mixed_fp16")
                self.assertEqual(len(layout["segments"]), 2)
                self.assertGreater(layout["parameter_uses"],
                                   layout["distinct_parameters"])
                for segment in layout["segments"]:
                    low = segment["gpu_base_address"]
                    high = low + segment["image_size_bytes"]
                    self.assertTrue(high <= STACK_BOTTOM or low >= STACK_TOP)
                    self.assertLess(high, CONSOLE_MMIO)
                for parameter in layout["parameters"]:
                    segment = layout["segments"][parameter["segment_index"]]
                    self.assertEqual(parameter["gpu_address"],
                                     segment["gpu_base_address"] + parameter["offset_bytes"])
                    self.assertEqual(parameter["storage_dtype"],
                                     "fp32" if parameter["operation"] in
                                     ("rmsnorm", "bias_add") else "fp16")

    def test_gemma_does_not_fit_mixed_fp16_placement(self):
        with self.assertRaisesRegex(ValueError, "exceeds safe GMEM"):
            plan("gemma_2_2b_it", 26)

    def test_quantized_reference_preserves_norms_and_rounds_linear_weights(self):
        class Source:
            def __call__(self, logical, shape):
                return np.full(shape, 1.0001, dtype=np.float32)

            def embedding_row(self, logical, row, width):
                return np.full(width, 1.0001, dtype=np.float32)

        reference = MixedFP16Weights(Source())
        self.assertEqual(reference("layer.norm", (2,))[0], np.float32(1.0001))
        self.assertEqual(reference("layer.q_bias", (2,))[0], np.float32(1.0001))
        self.assertEqual(reference("layer.q_proj", (2, 2))[0, 0], 1.0)
        self.assertEqual(reference.embedding_row("embed", 0, 2)[0], 1.0)

    def test_gemma_int8_plan_and_per_channel_conversion(self):
        layout = plan_int8(26)
        self.assertEqual(layout["graph_stages"], 996)
        self.assertEqual(layout["dtype"], "int8_fp16_tied")
        self.assertEqual(len(layout["segments"]), 2)
        embedding = next(item for item in layout["parameters"]
                         if item["logical_name"] == "embed_tokens")
        head = next(item for item in layout["parameters"]
                    if item["logical_name"] == "lm_head")
        self.assertEqual(embedding["storage_dtype"], "fp16")
        self.assertEqual(head["storage_dtype"], "fp16_tied")
        self.assertEqual(head["gpu_address"], embedding["gpu_address"])
        for segment in layout["segments"]:
            low = segment["gpu_base_address"]
            high = low + segment["image_size_bytes"]
            self.assertTrue(high <= STACK_BOTTOM or low >= STACK_TOP)
            self.assertLess(high, CONSOLE_MMIO)
        for parameter in layout["parameters"]:
            if parameter["storage_dtype"] == "int8_scaled":
                self.assertEqual(parameter["scale_gpu_address"] % 64, 0)
        matrix = np.array([[1.0, 2.0], [-1.0, 4.0]], dtype=np.float32)
        packed, scales, metric = quantize(matrix, "linear")
        np.testing.assert_allclose(packed * scales, matrix, rtol=0, atol=0.02)
        self.assertGreater(metric["roundtrip_changed_elements"], 0)
        packed, scales, _ = quantize(matrix, "embedding")
        np.testing.assert_allclose(packed * scales[:, None], matrix,
                                   rtol=0, atol=0.02)


if __name__ == "__main__":
    unittest.main()
