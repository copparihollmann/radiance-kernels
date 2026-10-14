"""Check the Gemma capacity plan and signed nibble layout."""

import unittest

import numpy as np

from checkpoint import (quantize_int4_groupwise, quantize_int4_per_channel,
                        unpack_int4)
from export_gemma_int4_tied import GPU_SIZE, plan
from split_decoder_weights import STACK_BOTTOM


class GemmaINT4PlanTest(unittest.TestCase):
    def test_full_model_fits_and_shares_embedding(self):
        image = plan(26)
        parameters = {item["logical_name"]: item for item in image["parameters"]}
        self.assertEqual(image["layers"], image["pinned_model_layers"])
        self.assertLess(image["gpu_end_address_exclusive"], STACK_BOTTOM)
        self.assertLess(image["gpu_end_address_exclusive"], GPU_SIZE)
        self.assertEqual(parameters["lm_head"]["gpu_address"],
                         parameters["embed_tokens"]["gpu_address"])
        self.assertEqual(parameters["lm_head"]["scale_gpu_address"],
                         parameters["embed_tokens"]["scale_gpu_address"])
        self.assertTrue(any(item["storage_dtype"] == "int4_scaled"
                            for item in image["parameters"]))

    def test_int4_pack_low_nibble_first_across_odd_rows(self):
        value = np.asarray([[-7, 7, 7], [3, -4, 2]], dtype=np.float32)
        packed, scales, _ = quantize_int4_per_channel(value)
        self.assertEqual(packed.tolist(), [0x79, 0x37, 0x2C])
        self.assertEqual(unpack_int4(packed, value.size).tolist(),
                         [-7, 7, 7, 3, -4, 2])
        np.testing.assert_array_equal(scales, np.ones(3, dtype=np.float32))

    def test_rejects_later_base(self):
        with self.assertRaisesRegex(ValueError, "exceeds the 2 GiB"):
            plan(26, 0x35000000)

    def test_groupwise_scales_follow_input_rows(self):
        value = np.asarray([[7, -7], [0, 2], [2, 0], [14, -14]],
                           dtype=np.float32)
        packed, scales, _ = quantize_int4_groupwise(value, group_size=2)
        self.assertEqual(scales.tolist(), [[1.0, 1.0], [2.0, 2.0]])
        self.assertEqual(unpack_int4(packed, value.size).tolist(),
                         [7, -7, 0, 2, 1, 0, 7, -7])


if __name__ == "__main__":
    unittest.main()
