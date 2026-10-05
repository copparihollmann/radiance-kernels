"""Small serial checks for the Spatter address mapping and input parser."""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import (destination, fnv, normalize, parse_pattern, payload, reference,
                 sample_digest, source_index, source_tag, write_source)
from plan import execute_reference, fuse_gather_scatter, stitch_reference
from run_chain import expected_output


class SpatterMappingTest(unittest.TestCase):
    def test_standard_gpu_stream_families(self):
        suite = json.loads((Path(__file__).parent / "smoke.json").read_text())
        for raw in suite[:5]:
            with self.subTest(kind=raw["kernel"]):
                case = normalize(raw)
                result = [0] * case["dst_length"]
                tag = source_tag(case["source"])
                for i in range(case["count"]):
                    for j in range(case["length"]):
                        dst = (j + case["length"] * (i % case["wrap"])
                               if case["kind"] in ("gather", "multigather")
                               else destination(case, i, j))
                        result[dst] = payload(tag, source_index(case, i, j))
                digest, overlap = reference(case)
                self.assertFalse(overlap)
                self.assertEqual(digest, fnv(result))
                source = [payload(tag, i) for i in range(case["src_length"])]
                self.assertEqual(execute_reference(case, source), result)
                self.assertEqual(sample_digest(case, case["dst_length"]), digest)
                n = min(3, case["dst_length"])
                positions = [i * (case["dst_length"] - 1) // (n - 1)
                             for i in range(n)] if n > 1 else [0]
                self.assertEqual(sample_digest(case, n),
                                 fnv(result[pos] for pos in positions))

    def test_wrap_and_collisions(self):
        wrapped = normalize({"kernel": "Gather", "pattern": [3, 1],
                             "count": 5, "wrap": 2})
        digest, overlap = reference(wrapped)
        self.assertFalse(overlap)
        self.assertEqual(digest, fnv([
            payload(1, 35), payload(1, 33),
            payload(1, 27), payload(1, 25),
        ]))
        colliding = normalize({"kernel": "Scatter", "pattern": [0, 0], "count": 1})
        self.assertEqual(reference(colliding), (None, True))
        ordered = normalize({"kernel": "Scatter", "pattern": [0, 0],
                             "count": 1, "collision-policy": "ordered"})
        self.assertEqual(ordered["_plan"].schedule, "destination_owner")
        self.assertEqual(reference(ordered),
                         (fnv([payload(2, 1)]), True))

    def test_wrap_one_gather_final_chunk_matches_full_count(self):
        full = normalize({"kernel": "Gather", "pattern": [4, 1, 3],
                          "count": 5, "delta": 3, "wrap": 1})
        final = normalize({"kernel": "Gather", "pattern": [4, 1, 3],
                           "count": 2, "delta": 3, "wrap": 1})
        final["source_index_base"] = 3 * full["delta"]
        self.assertEqual(reference(final), reference(full))
        self.assertEqual(sample_digest(final, 3), reference(full)[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.bin"
            write_source(path, final["src_length"], final["payload_tag"],
                         final["source_index_base"])
            values = struct.unpack(f"<{final['src_length']}Q", path.read_bytes())
            self.assertEqual(values[4], payload(full["payload_tag"], 13))

    def test_stitch_gather_and_scatter_as_gs(self):
        gather = normalize({"kernel": "Gather", "pattern": [4, 1, 3],
                            "count": 2, "delta": 5, "wrap": 2})
        scatter = normalize({"kernel": "Scatter", "pattern": [2, 0, 1],
                             "count": 2, "delta": 5, "wrap": 2})
        fused = normalize(fuse_gather_scatter(gather, scatter))
        self.assertEqual(gather["dst_length"], scatter["src_length"])
        self.assertEqual(fused["payload_tag"], gather["payload_tag"])
        source = list(range(gather["src_length"]))
        self.assertEqual(stitch_reference([gather, scatter], source),
                         execute_reference(fused, source))
        generated_source = [payload(gather["payload_tag"], i)
                            for i in range(gather["src_length"])]
        self.assertEqual(fnv(stitch_reference([gather, scatter], generated_source)),
                         reference(fused)[0])
        with self.assertRaisesRegex(ValueError, "stage output length"):
            stitch_reference([gather, gather], source)
        wrapped_gather = normalize({"kernel": "Gather", "pattern": [4, 1, 3],
                                    "count": 3, "delta": 5, "wrap": 2})
        wrapped_scatter = normalize({"kernel": "Scatter", "pattern": [2, 0, 1],
                                     "count": 3, "delta": 5, "wrap": 2})
        wrapped_fused = normalize(fuse_gather_scatter(wrapped_gather, wrapped_scatter))
        wrapped_source = [payload(wrapped_gather["payload_tag"], i)
                          for i in range(wrapped_gather["src_length"])]
        expected = stitch_reference([wrapped_gather, wrapped_scatter], wrapped_source)
        self.assertEqual(execute_reference(wrapped_fused, wrapped_source), expected)
        self.assertEqual(reference(wrapped_fused), (fnv(expected), False))
        ordered_scatter = normalize({"kernel": "Scatter", "pattern": [0, 0, 1],
                                     "count": 3, "delta": 5, "wrap": 2,
                                     "collision-policy": "ordered"})
        ordered_fused = normalize(fuse_gather_scatter(wrapped_gather, ordered_scatter))
        self.assertEqual(ordered_fused["collision_policy"], "ordered")
        ordered_expected = stitch_reference([wrapped_gather, ordered_scatter], wrapped_source)
        self.assertEqual(execute_reference(ordered_fused, wrapped_source), ordered_expected)
        self.assertEqual(reference(ordered_fused), (fnv(ordered_expected), True))
        mismatched_wrap = normalize({"kernel": "Gather", "pattern": [4, 1, 3],
                                     "count": 3, "delta": 5, "wrap": 1})
        with self.assertRaisesRegex(ValueError, "matching wrap"):
            fuse_gather_scatter(mismatched_wrap, wrapped_scatter)

    def test_materialized_chain_with_different_stage_counts(self):
        gather = normalize({"kernel": "Gather", "pattern": [4, 1, 3],
                            "count": 3, "wrap": 2, "delta": 5})
        scatter = normalize({"kernel": "Scatter", "pattern": [2, 0, 1],
                             "count": 2, "wrap": 2, "delta": 5})
        source = [payload(gather["payload_tag"], i)
                  for i in range(gather["src_length"])]
        self.assertEqual(list(expected_output(gather, scatter)[0]),
                         stitch_reference([gather, scatter], source))

    def test_materialized_chain_with_ordered_scatter(self):
        gather = normalize({"kernel": "Gather", "pattern": [4, 1, 3],
                            "count": 2, "wrap": 2, "delta": 5})
        scatter_raw = {"kernel": "Scatter", "pattern": [0, 0, 1],
                       "count": 2, "wrap": 2, "delta": 0}
        ordered = normalize({**scatter_raw, "collision-policy": "ordered"})
        source = [payload(gather["payload_tag"], i)
                  for i in range(gather["src_length"])]
        output, overlap = expected_output(gather, ordered)
        self.assertTrue(overlap)
        self.assertEqual(list(output), stitch_reference([gather, ordered], source))
        with self.assertRaisesRegex(ValueError, "ordered collision policy"):
            expected_output(gather, normalize(scatter_raw))

    def test_pattern_generators(self):
        self.assertEqual(parse_pattern("UNIFORM:4:3:NR", 8), ([0, 3, 6, 9], 12))
        self.assertEqual(parse_pattern("MS1:4:2:3", 8), ([0, 1, 4, 5], 8))
        self.assertEqual(parse_pattern("LAPLACIAN:1:1:4", 8), ([0, 1, 2], 1))

    def test_rejects_unsupported_memory_semantics(self):
        for option, value in (("atomic-writes", True), ("boundary", 4),
                              ("compress", True), ("shared-memory", 1024)):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, option):
                normalize({"kernel": "Scatter", "pattern": [0, 1], option: value})
        with self.assertRaisesRegex(ValueError, "needs a scatter"):
            normalize({"kernel": "Gather", "pattern": [0, 1],
                       "collision-policy": "ordered"})


if __name__ == "__main__":
    unittest.main()
