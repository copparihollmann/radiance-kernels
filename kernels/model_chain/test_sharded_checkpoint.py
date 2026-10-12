"""A sharded checkpoint must resolve tensors through its safetensors index."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

from export_decoder_weights import SafeTensorWeights


class ShardedCheckpointTest(unittest.TestCase):
    def test_index_resolves_tied_embedding_across_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_file({"model.embed_tokens.weight": np.arange(12, dtype=np.float32).reshape(3, 4)},
                      root / "weights-1.safetensors")
            save_file({"model.norm.weight": np.ones(4, dtype=np.float32)},
                      root / "weights-2.safetensors")
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {
                    "model.embed_tokens.weight": "weights-1.safetensors",
                    "model.norm.weight": "weights-2.safetensors",
                }}))
            weights = SafeTensorWeights(root, "gemma2")
            np.testing.assert_array_equal(weights.embedding_row("embed_tokens", 2, 4),
                                          [8, 9, 10, 11])
            self.assertEqual(weights.key("lm_head"), "model.embed_tokens.weight")
            np.testing.assert_array_equal(weights("lm_head", (4, 3)),
                                          np.arange(12, dtype=np.float32).reshape(3, 4).T)
            np.testing.assert_array_equal(weights("final_norm", (4,)),
                                          np.ones(4, dtype=np.float32))

    def test_index_rejects_shard_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"model.norm.weight": "../outside.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "invalid checkpoint shard"):
                SafeTensorWeights(root, "gemma2")


if __name__ == "__main__":
    unittest.main()
