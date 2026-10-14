import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inventory import inventory, layer_kinds


class InventoryTest(unittest.TestCase):
    def test_pinned_cases_keep_layer_order(self):
        cases = {entry["case"]: entry for entry in
                 inventory(Path(__file__).resolve().parent / "inputs")["cases"]}
        self.assertEqual(len(cases), 5)
        self.assertEqual(cases["olmo3-7b"]["layer_kinds"][:4],
                         ["sliding_attention+dense_ffn"] * 3 +
                         ["full_attention+dense_ffn"])
        self.assertEqual(cases["olmo3-7b"]["counts"],
                         {"full_attention+dense_ffn": 8,
                          "sliding_attention+dense_ffn": 24})
        self.assertFalse(cases["olmo3-7b"]["checkpoint_use_cache"])
        self.assertEqual(cases["qwen3-30b-a3b"]["counts"],
                         {"full_attention+moe": 48})
        self.assertEqual(cases["qwen3-30b-a3b"]["experts_per_token"], 8)
        self.assertEqual(cases["nemotron3-nano-4b"]["counts"],
                         {"full_attention": 4, "mamba2": 21, "mlp_only": 17})

    def test_sparse_step_and_explicit_mlp_layers(self):
        config = {"model_type": "qwen3_moe", "num_hidden_layers": 4,
                  "num_experts": 8, "num_experts_per_tok": 2,
                  "decoder_sparse_step": 2, "mlp_only_layers": [3]}
        self.assertEqual(layer_kinds(config),
                         ["full_attention+dense_ffn", "full_attention+moe",
                          "full_attention+dense_ffn", "full_attention+dense_ffn"])

    def test_rejects_unknown_layer_semantics(self):
        with self.assertRaises(ValueError):
            layer_kinds({"model_type": "nemotron_h", "num_hidden_layers": 2,
                         "hybrid_override_pattern": "M?"})
        with self.assertRaises(ValueError):
            layer_kinds({"model_type": "mistral", "num_hidden_layers": 2,
                         "sliding_window": 1024})


if __name__ == "__main__":
    unittest.main()
