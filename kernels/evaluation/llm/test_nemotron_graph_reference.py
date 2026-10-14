"""Check the ordered Nemotron H graph and its carried state."""

import unittest

import numpy as np

import nemotron_graph_reference as nemotron
import stitch


class NemotronGraphReferenceTest(unittest.TestCase):
    def test_pinned_prefix_through_first_attention(self):
        spec = nemotron.control_spec(stitch.model_specs()["nemotron3_nano_4b"], 13)
        self.assertEqual(spec["layer_kinds"][-1], "full_attention")
        self.assertEqual(spec["layer_kinds"].count("mamba2"), 7)
        self.assertEqual(spec["layer_kinds"].count("mlp_only"), 5)
        token_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        graph = stitch.build("nemotron3_nano_4b", prefill=3, decode_steps=2,
                             generation="teacher_forced",
                             specs={"nemotron3_nano_4b": spec})
        self.assertEqual(len(graph.stages), 429)
        incremental = nemotron.execute(graph, {
            "prefill.token_ids": token_ids[:, :3],
            "decode0.token_ids": token_ids[:, 3:4],
            "decode1.token_ids": token_ids[:, 4:5],
        })
        full = stitch.build("nemotron3_nano_4b", prefill=5, decode_steps=0,
                            specs={"nemotron3_nano_4b": spec})
        full_values = nemotron.execute(full, {"prefill.token_ids": token_ids})
        self.assertTrue(np.allclose(incremental[graph.outputs[-1]],
                                    full_values[full.outputs[-1]],
                                    rtol=1e-4, atol=1e-5))

        changed = {"prefill.token_ids": token_ids[:, :3],
                   "decode0.token_ids": token_ids[:, 3:4],
                   "decode1.token_ids": np.array([[6]], dtype=np.int32)}
        changed_values = nemotron.execute(graph, changed)
        self.assertGreater(float(np.max(np.abs(
            incremental[graph.outputs[-1]] - changed_values[graph.outputs[-1]]))),
            0.0)

        conv = next(stage for stage in graph.stages
                    if stage["id"] == "decode0.layer00.depthwise_conv")
        original = conv["reads"]
        self.assertEqual(len(original), 2)
        try:
            conv["reads"] = original[:1]
            no_cache = nemotron.execute(graph, {
                "prefill.token_ids": token_ids[:, :3],
                "decode0.token_ids": token_ids[:, 3:4],
                "decode1.token_ids": token_ids[:, 4:5],
            })
        finally:
            conv["reads"] = original
        self.assertGreater(float(np.max(np.abs(
            incremental[conv["writes"]] - no_cache[conv["writes"]]))),
            1e-7)


if __name__ == "__main__":
    unittest.main()
