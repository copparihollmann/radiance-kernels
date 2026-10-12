"""Guard the SmolVLA action and expert dependencies against flattening."""

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

from audit_full_models import check_decoder_lineage, check_smolvla_lineage


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kernels/evaluation/llm"))
from stitch import build  # noqa: E402


class SmolVlaLineageTest(unittest.TestCase):
    def setUp(self):
        self.graph = build("smolvla_base")
        self.digest = hashlib.sha256(json.dumps(
            self.graph.execution_schedule, sort_keys=True).encode()).hexdigest()

    def stage(self, graph, name):
        return next(stage for stage in graph.stages if stage["id"] == name)

    def test_full_loop_is_connected(self):
        result = check_smolvla_lineage(self.graph, self.digest)
        self.assertEqual((result["prefix_stages"], result["denoise_steps"],
                          result["expert_layers_per_step"]), (913, 10, 16))

    def test_reusing_initial_action_in_later_step_is_rejected(self):
        graph = copy.deepcopy(self.graph)
        self.stage(graph, "denoise1.action_in_linear")["reads"] = ["action.noise"]
        with self.assertRaisesRegex(ValueError, "does not carry the action"):
            check_smolvla_lineage(graph, self.digest)

    def test_resetting_euler_carry_is_rejected(self):
        graph = copy.deepcopy(self.graph)
        self.stage(graph, "denoise1.euler")["reads"] = [
            "action.noise", "denoise1.action_out"]
        with self.assertRaisesRegex(ValueError, "does not carry the action"):
            check_smolvla_lineage(graph, self.digest)

    def test_skipping_expert_layer_is_rejected(self):
        graph = copy.deepcopy(self.graph)
        self.stage(graph, "denoise0.expert01.input_norm")["reads"] = [
            "denoise0.action_embed"]
        with self.assertRaisesRegex(ValueError, "expert layer 1 is disconnected"):
            check_smolvla_lineage(graph, self.digest)


class DecoderLineageTest(unittest.TestCase):
    def setUp(self):
        self.graph = build("tinyllama", prefill=3, decode_steps=2)

    def stage(self, graph, name):
        return next(stage for stage in graph.stages if stage["id"] == name)

    def test_all_decoder_families_keep_causal_cache(self):
        for model in ("tinyllama", "deepseek_r1_distill_qwen_1_5b",
                      "gemma_2_2b_it"):
            with self.subTest(model=model):
                graph = build(model, prefill=3, decode_steps=2)
                result = check_decoder_lineage(graph, 3, 2)
                self.assertEqual(result["cache_handoffs_checked"],
                                 2 * graph.spec["num_hidden_layers"])

    def test_skipping_decode_cache_is_rejected(self):
        graph = copy.deepcopy(self.graph)
        cache = self.stage(graph, "decode1.layer00.k_cache")
        cache["reads"][0] = "prefill.layer00.k_cache"
        with self.assertRaisesRegex(ValueError, "causal cache lineage"):
            check_decoder_lineage(graph, 3, 2)

    def test_resetting_hidden_state_between_layers_is_rejected(self):
        graph = copy.deepcopy(self.graph)
        self.stage(graph, "decode1.layer01.attn_norm")["reads"] = [
            "decode1.embedding"]
        with self.assertRaisesRegex(ValueError, "causal cache lineage"):
            check_decoder_lineage(graph, 3, 2)

    def test_resetting_decode_position_is_rejected(self):
        graph = copy.deepcopy(self.graph)
        self.stage(graph, "decode1.layer00.attention")["attrs"]["query_start"] = 0
        with self.assertRaisesRegex(ValueError, "causal cache lineage"):
            check_decoder_lineage(graph, 3, 2)


if __name__ == "__main__":
    unittest.main()
