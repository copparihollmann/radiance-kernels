"""Capacity checks for the full-depth INT8 decoder plans."""

import unittest

from export_decoder_int8_untied import GPU_SIZE, MODELS, plan
from split_decoder_weights import STACK_BOTTOM
from stitch import model_specs


class FullDepthINT8PlanTest(unittest.TestCase):
    def test_both_full_models_fit_below_the_stack(self):
        specs = model_specs()
        for model in MODELS:
            with self.subTest(model=model):
                image = plan(model, specs[model]["num_hidden_layers"])
                self.assertLess(image["gpu_end_address_exclusive"], STACK_BOTTOM)
                self.assertLess(image["gpu_end_address_exclusive"], GPU_SIZE)
                self.assertEqual(image["layers"], image["pinned_model_layers"])
                self.assertTrue(any(item["storage_dtype"] == "int8_scaled"
                                    for item in image["parameters"]))

    def test_deepseek_rejects_a_later_base(self):
        with self.assertRaisesRegex(ValueError, "exceeds the 2 GiB"):
            plan("deepseek_r1_distill_qwen_1_5b", 28, 0x15000000)


if __name__ == "__main__":
    unittest.main()
