"""Compare greedy Qwen3's opt-in last-token logits with the full projection."""

from __future__ import annotations

import unittest

import numpy as np

import last_token_graph_reference
import reference
import stitch


class LastPrefillTokenTest(unittest.TestCase):
    def test_prefill_and_greedy_decode_match_full_logit_graph(self) -> None:
        spec = reference.reduced_spec(stitch.model_specs()["qwen3_0_6b"])
        spec["num_hidden_layers"] = 2
        kwargs = {"prefill": 3, "decode_steps": 2,
                  "generation": "greedy", "specs": {"qwen3_0_6b": spec}}
        full = stitch.build("qwen3_0_6b", **kwargs)
        last = stitch.build("qwen3_0_6b", prefill_logits="last", **kwargs)
        inputs = {"prefill.token_ids": np.array([[1, 2, 3]], dtype=np.int32)}
        original = reference.execute(full, inputs)
        selected = last_token_graph_reference.execute(
            last, inputs, retain=("prefill.last_hidden", "decode0.token_ids",
                                  "decode1.token_ids"))
        self.assertEqual(original[full.outputs[0]].shape, (1, 3, 64))
        self.assertEqual(selected[last.outputs[0]].shape, (1, 1, 64))
        np.testing.assert_allclose(selected[last.outputs[0]],
                                   original[full.outputs[0]][:, -1:, :],
                                   rtol=1e-6, atol=1e-6)
        for step in range(2):
            name = f"decode{step}.token_ids"
            np.testing.assert_array_equal(selected[name], original[name])
            np.testing.assert_allclose(selected[last.outputs[step + 1]],
                                       original[full.outputs[step + 1]],
                                       rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(selected["prefill.last_hidden"],
                                      original["prefill.final_norm"][:, -1:, :])
        self.assertGreater(float(np.max(np.abs(
            original["prefill.final_norm"][:, :1, :] -
            selected["prefill.last_hidden"]))), 1e-5)


if __name__ == "__main__":
    unittest.main()
