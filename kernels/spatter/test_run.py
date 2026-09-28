"""Small serial checks for the Spatter address mapping and input parser."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import (destination, fnv, normalize, parse_pattern, payload, reference,
                 sample_digest, source_index, source_tag)
from plan import execute_reference, stitch_reference


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

    def test_stitch_gather_and_scatter_as_gs(self):
        gather = normalize({"kernel": "Gather", "pattern": [4, 1, 3],
                            "count": 2, "delta": 5, "wrap": 2})
        scatter = normalize({"kernel": "Scatter", "pattern": [2, 0, 1],
                             "count": 2, "delta": 5, "wrap": 2})
        fused = normalize({"kernel": "GS", "pattern-gather": [4, 1, 3],
                           "pattern-scatter": [2, 0, 1], "count": 2,
                           "delta-gather": 5, "delta-scatter": 5})
        self.assertEqual(gather["dst_length"], scatter["src_length"])
        source = list(range(gather["src_length"]))
        self.assertEqual(stitch_reference([gather, scatter], source),
                         execute_reference(fused, source))
        with self.assertRaisesRegex(ValueError, "stage output length"):
            stitch_reference([gather, gather], source)

    def test_pattern_generators(self):
        self.assertEqual(parse_pattern("UNIFORM:4:3:NR", 8), ([0, 3, 6, 9], 12))
        self.assertEqual(parse_pattern("MS1:4:2:3", 8), ([0, 1, 4, 5], 8))
        self.assertEqual(parse_pattern("LAPLACIAN:1:1:4", 8), ([0, 1, 2], 1))

    def test_rejects_unsupported_memory_semantics(self):
        for option, value in (("atomic-writes", True), ("boundary", 4),
                              ("compress", True), ("shared-memory", 1024)):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, option):
                normalize({"kernel": "Scatter", "pattern": [0, 1], option: value})


if __name__ == "__main__":
    unittest.main()
