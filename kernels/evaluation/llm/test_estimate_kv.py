import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimate_kv import attention_windows, kv_sizes


class KVSizeTest(unittest.TestCase):
    def test_dense_and_local_full(self):
        dense = {"num_hidden_layers": 2, "num_attention_heads": 4,
                 "num_key_value_heads": 2, "hidden_size": 32,
                 "torch_dtype": "bfloat16"}
        self.assertEqual(kv_sizes(dense, 1, 16), (2, 2048, 1024, 128))
        local = {**dense, "layer_types": ["sliding_attention", "full_attention"],
                 "sliding_window": 4}
        self.assertEqual(kv_sizes(local, 1, 16), (2, 1280, 1024, 128))

    def test_hybrid_counts_attention_only(self):
        hybrid = {"num_hidden_layers": 4, "num_attention_heads": 4,
                  "num_key_value_heads": 2, "head_dim": 8,
                  "hidden_size": 32, "torch_dtype": "bfloat16",
                  "hybrid_override_pattern": "M-**"}
        self.assertEqual(attention_windows(hybrid), [None, None])
        self.assertEqual(kv_sizes(hybrid, 1, 16), (2, 2048, 1024, 128))


if __name__ == "__main__":
    unittest.main()
