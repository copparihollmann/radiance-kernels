"""Check storage reuse without changing the model's dependency schedule."""

import unittest

from plan_buffers import plan
from stitch import build


class BufferPlanTest(unittest.TestCase):
    def test_decoder_cache_lifetime_is_preserved(self):
        graph = build("tinyllama", prefill=3, decode_steps=2)
        result = plan(graph)
        self.assertLess(result["arena_bytes"], result["unreused_bytes"])
        prefill_cache = result["allocation"]["prefill.layer00.k_cache"]
        decode_cache = result["allocation"]["decode0.layer00.k_cache"]
        self.assertLessEqual(prefill_cache["birth"], decode_cache["birth"])
        self.assertGreaterEqual(prefill_cache["last_use"], decode_cache["birth"])

    def test_smolvla_nested_schedule_is_unchanged(self):
        graph = build("smolvla_base")
        original_schedule = graph.execution_schedule
        result = plan(graph)
        self.assertEqual(graph.execution_schedule, original_schedule)
        self.assertLess(result["arena_bytes"], 64 * 1024 * 1024)
        self.assertEqual(result["stage_count"], 3673)
        self.assertEqual(len(original_schedule["denoise_loop"]["iterations"]), 10)


if __name__ == "__main__":
    unittest.main()
