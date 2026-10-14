"""Check the workload distinctions used in the architecture shape matrix."""

import unittest

from profile_architecture_shapes import attention_dot_macs, matrix


class ArchitectureShapesTest(unittest.TestCase):
    def test_visible_pair_count_respects_local_window(self):
        stage = {"op": "causal_gqa", "shape": [1, 5, 64],
                 "attrs": {"query_start": 3, "window": 4,
                           "q_heads": 2, "head_dim": 8}}
        # The five queries can see 4, 4, 4, 4, and 4 keys.
        self.assertEqual(attention_dot_macs(stage), 2 * 2 * 8 * 20)
        stage["attrs"]["window"] = None
        self.assertEqual(attention_dot_macs(stage), 2 * 2 * 8 * 30)

    def test_pinned_classes_keep_distinct_work_and_cache_shapes(self):
        result = matrix()
        cases = {case["case"]: case for case in result["cases"]}
        self.assertEqual(len(cases), 5)
        for item in cases.values():
            self.assertEqual(set(item["scenarios"]),
                             {"short_b1", "short_b8", "long_b1"})
            one = item["scenarios"]["short_b1"]
            eight = item["scenarios"]["short_b8"]
            self.assertEqual([one["phase_work"][phase]["stages"]
                              for phase in ("prefill", "decode0", "decode1")],
                             [eight["phase_work"][phase]["stages"]
                              for phase in ("prefill", "decode0", "decode1")])
            self.assertEqual(eight["phase_work"]["prefill"]["linear_macs"],
                             8 * one["phase_work"]["prefill"]["linear_macs"])
        moe = cases["qwen3-30b-a3b"]["scenarios"]["short_b1"]
        self.assertGreater(moe["phase_work"]["prefill"]["expert_macs"], 0)
        self.assertEqual(cases["qwen3-0.6b"]["scenarios"]["short_b1"]
                         ["phase_work"]["prefill"]["expert_macs"], 0)
        olmo = cases["olmo3-7b"]["scenarios"]["long_b1"]
        self.assertFalse(cases["olmo3-7b"]["checkpoint_use_cache"])
        self.assertLess(olmo["window_trimmed_kv_logical_bytes"],
                        olmo["final_full_history_kv_logical_bytes"])
        self.assertEqual(cases["nemotron3-nano-4b"]["scenarios"]["short_b1"]
                         ["attention_layers"], 4)


if __name__ == "__main__":
    unittest.main()
