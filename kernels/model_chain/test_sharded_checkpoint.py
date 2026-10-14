"""A sharded checkpoint must resolve tensors through its safetensors index."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

from export_decoder_weights import SafeTensorWeights
from checkpoint import MixedFP16Weights
import reference
import stitch


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

    def test_qwen3_tied_embedding_head_norm_and_projection_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            table = np.arange(12, dtype=np.float32).reshape(3, 4)
            projection = np.arange(16, dtype=np.float32).reshape(4, 4)
            save_file({
                "model.embed_tokens.weight": table,
                "lm_head.weight": table.copy(),
                "model.layers.0.self_attn.q_norm.weight": np.ones(4, dtype=np.float32),
                "model.layers.0.self_attn.k_norm.weight": np.full(4, 2, dtype=np.float32),
                "model.layers.0.self_attn.q_proj.weight": projection,
            }, path)
            weights = SafeTensorWeights(path, "qwen3")
            np.testing.assert_array_equal(weights.embedding_table("embed_tokens", 3, 4),
                                          table)
            np.testing.assert_array_equal(
                np.concatenate(list(weights.embedding_blocks("embed_tokens", 3, 4,
                                                            block_rows=2))), table)
            np.testing.assert_array_equal(weights("embed_tokens", (3, 4)), table)
            np.testing.assert_array_equal(weights("layers.0.q_proj", (4, 4)),
                                          projection.T)
            np.testing.assert_array_equal(weights("layers.0.q_norm", (4,)),
                                          np.ones(4, dtype=np.float32))
            np.testing.assert_array_equal(weights("layers.0.k_norm", (4,)),
                                          np.full(4, 2, dtype=np.float32))
            self.assertEqual(weights.verify_tied_lm_head()["compared_elements"], 12)

    def test_qwen3_saved_head_must_match_every_embedding_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            table = np.arange(1200, dtype=np.float32).reshape(300, 4)
            head = table.copy()
            head[-1, -1] += 1
            save_file({"model.embed_tokens.weight": table,
                       "lm_head.weight": head}, path)
            weights = SafeTensorWeights(path, "qwen3")
            np.testing.assert_array_equal(weights.embedding_row("embed_tokens", 299, 4),
                                          table[-1])
            self.assertIsNone(weights._cached_name)
            with self.assertRaisesRegex(ValueError, "head differs"):
                weights.verify_tied_lm_head()

    def test_qwen3_fp16_reference_streams_tied_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            table = (np.arange(1200, dtype=np.float32).reshape(300, 4) / 37)
            save_file({"model.embed_tokens.weight": table}, path)
            source = SafeTensorWeights(path, "qwen3")
            graph = stitch.Graph("tied-head-control", {"family": "qwen3"})
            graph.input("hidden", (1, 1, 4))
            graph.add("head", "linear", ("hidden",), (1, 1, 300), attrs={
                "parameter": "embed_tokens", "weight_shape": [300, 4],
                "weight_transposed": True})
            hidden = np.array([[[1, 2, 3, 4]]], dtype=np.float32)
            actual = reference.execute(
                graph, {"hidden": hidden}, weights=MixedFP16Weights(source))["head"]
            expected = hidden @ table.astype(np.float16).astype(np.float32).T
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)
            self.assertIsNone(source._cached_name)


if __name__ == "__main__":
    unittest.main()
