import math
import unittest

from minicpm3_mla_reference import (
    Dimensions,
    OfficialConfig,
    Parameters,
    deterministic_fixture,
    official_parameter_count,
    run_oracle,
    split_reconstructed_head,
)


class MiniCPM3MlaReferenceTest(unittest.TestCase):
    def test_pinned_parameter_count_and_dimensions(self):
        config = OfficialConfig()
        self.assertEqual(official_parameter_count(config), 4_073_875_968)
        self.assertEqual(config.hidden_size // config.num_attention_heads, 64)
        self.assertEqual(
            config.kv_lora_rank + config.qk_rope_head_dim, 288
        )
        self.assertEqual(
            config.num_attention_heads
            * (config.qk_nope_head_dim + config.v_head_dim),
            5_120,
        )

    def test_small_hand_computable_case(self):
        dims = Dimensions(1, 2, 1, 1, 1, 1, 1)
        hidden = [2.0, -1.0]
        parameters = Parameters(
            kv_a_weight=[0.5, 1.0, 2.0, -1.0],
            kv_norm_weight=[1.25],
            kv_b_weight=[2.0, -3.0],
        )
        result = run_oracle(dims, hidden, parameters, rms_norm_epsilon=1.0e-5)
        self.assertEqual(result.latent, [0.0])
        self.assertEqual(result.rope_key, [5.0])
        self.assertEqual(result.normalized_latent, [0.0])
        self.assertEqual(result.reconstructed_kv, [0.0, 0.0])

    def test_determinism_shapes_and_head_partition(self):
        dims = Dimensions(7, 64, 16, 8, 4, 16, 16)
        hidden_a, parameters_a = deterministic_fixture(dims)
        hidden_b, parameters_b = deterministic_fixture(dims)
        self.assertEqual(hidden_a, hidden_b)
        self.assertEqual(parameters_a, parameters_b)
        result = run_oracle(dims, hidden_a, parameters_a)
        self.assertEqual(len(result.flatten()), dims.output_elements)
        key, value = split_reconstructed_head(dims, result, 3, 2)
        self.assertEqual(len(key), dims.qk_nope_head_dim)
        self.assertEqual(len(value), dims.v_head_dim)
        per_head = dims.qk_nope_head_dim + dims.v_head_dim
        begin = 3 * dims.reconstruction_width + 2 * per_head
        self.assertEqual(key + value, result.reconstructed_kv[begin : begin + per_head])
        self.assertTrue(all(math.isfinite(item) for item in result.flatten()))

    def test_rejects_malformed_input(self):
        dims = Dimensions(1, 2, 1, 1, 1, 1, 1)
        _, parameters = deterministic_fixture(dims)
        with self.assertRaises(ValueError):
            run_oracle(dims, [1.0], parameters)


if __name__ == "__main__":
    unittest.main()
