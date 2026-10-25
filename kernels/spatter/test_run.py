"""Small serial checks for the Spatter address mapping and input parser."""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as run_module
from run import (GENERATED, destination, fnv, normalize, parse_pattern, payload,
                 prepare, reference, sample_digest, source_index, source_tag,
                 write_destination_groups, write_source,
                 written_destination_count)
from plan import execute_reference, fuse_gather_scatter, stitch_reference
from run_chain import expected_output


class SpatterMappingTest(unittest.TestCase):
    def setUp(self):
        # prepare() materializes generated inputs and data.S. Keep unit tests
        # away from the production build workspace so they cannot replace a
        # full-size payload while another process is linking an ELF.
        self.workspace = tempfile.TemporaryDirectory()
        self.old_root = run_module.ROOT
        self.old_generated = run_module.GENERATED
        run_module.ROOT = Path(self.workspace.name)
        run_module.GENERATED = run_module.ROOT / "generated"
        run_module.GENERATED.mkdir()
        globals()["GENERATED"] = run_module.GENERATED

    def tearDown(self):
        run_module.ROOT = self.old_root
        run_module.GENERATED = self.old_generated
        globals()["GENERATED"] = self.old_generated
        self.workspace.cleanup()

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
        self.assertEqual(written_destination_count(colliding), 1)

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

    def test_final_iteration_for_each_gather_residue(self):
        count, wrap = 11, 4
        finals = [r + ((count - 1 - r) // wrap) * wrap for r in range(wrap)]
        self.assertEqual(finals, [8, 9, 10, 7])
        self.assertEqual(sorted(i % wrap for i in finals), list(range(wrap)))

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

    def test_binary_pattern_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pattern = root / "pattern.u32le"
            pattern.write_bytes(struct.pack("<4I", 7, 0, 11, 3))
            import hashlib
            descriptor = {
                "file": pattern.name, "format": "u32le", "length": 4,
                "bytes": 16, "sha256": hashlib.sha256(pattern.read_bytes()).hexdigest(),
            }
            values, delta = parse_pattern(descriptor, 8, root)
            self.assertEqual(list(values), [7, 0, 11, 3])
            self.assertEqual(delta, 8)
            case = normalize({"kernel": "Gather", "pattern": descriptor,
                              "count": 1}, root)
            self.assertEqual(case["length"], 4)
            self.assertEqual(case["_pattern_inputs"][0]["sha256"], descriptor["sha256"])
            with self.assertRaisesRegex(ValueError, "hash differs"):
                parse_pattern({**descriptor, "sha256": "0" * 64}, 8, root)

    def test_rejects_unsupported_memory_semantics(self):
        for option, value in (("atomic-writes", True), ("boundary", 4),
                              ("compress", True), ("shared-memory", 1024)):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, option):
                normalize({"kernel": "Scatter", "pattern": [0, 1], option: value})
        with self.assertRaisesRegex(ValueError, "needs a scatter"):
            normalize({"kernel": "Gather", "pattern": [0, 1],
                       "collision-policy": "ordered"})

    def test_records_optional_cache_line_padding(self):
        case = normalize({"kernel": "Gather", "pattern": [0, 1], "count": 1})
        suite = Path(__file__).parent / "smoke.json"
        result = prepare(case, suite, 0, sms=2, loader="logical",
                         source_pad_lines=3, output_pad_lines=5)
        assembly = (run_module.ROOT / "data.S").read_text()
        self.assertEqual(result["source_pad_cache_lines"], 3)
        self.assertEqual(result["output_pad_cache_lines"], 5)
        self.assertEqual(result["layout_padding_bytes"], 512)
        self.assertIn(".zero 192\n.globl spatter_sparse", assembly)
        self.assertIn(".zero 320\n.globl spatter_guard_before", assembly)
        self.assertIn(".zero 64\n.globl spatter_dense", assembly)
        self.assertTrue(GENERATED.joinpath("manifest.json").is_file())
        with self.assertRaisesRegex(ValueError, "between 0 and 63"):
            prepare(case, suite, 0, source_pad_lines=64)

    def test_records_optional_warps_per_core(self):
        case = normalize({"kernel": "Gather", "pattern": [0, 1], "count": 1})
        suite = Path(__file__).parent / "smoke.json"
        result = prepare(case, suite, 0, sms=4, warps_per_core=8)
        self.assertEqual(result["muon_warps_per_core"], 8)
        config = GENERATED.joinpath("config.h").read_text()
        self.assertIn("#define NIGHTLY_OCC 8", config)
        with self.assertRaisesRegex(ValueError, "warps_per_core"):
            prepare(case, suite, 0, warps_per_core=3)

    def test_block_cyclic_gather_owns_whole_16_element_chunks(self):
        case = normalize({"kernel": "Gather", "pattern": list(range(73)),
                          "count": 1, "wrap": 1})
        suite = Path(__file__).parent / "smoke.json"
        result = prepare(case, suite, 0, sms=4,
                         gather_block_cyclic_16=True)
        self.assertTrue(result["gather_block_cyclic_16"])
        config = GENERATED.joinpath("config.h").read_text()
        self.assertIn("#define SPATTER_GATHER_BLOCK_CYCLIC_16 1", config)

        chunks = (case["length"] + 15) // 16
        owners_by_sm = [[] for _ in range(4)]
        for chunk in range(chunks):
            sm = chunk % 4
            owners_by_sm[sm].extend(
                range(chunk * 16, min(case["length"], (chunk + 1) * 16)))
        self.assertEqual(sorted(owner for owners in owners_by_sm for owner in owners),
                         list(range(case["length"])))
        self.assertEqual([len(owners) for owners in owners_by_sm], [25, 16, 16, 16])
        for chunk in range(chunks):
            # Sixteen float64 outputs are exactly two whole 64-byte lines.
            self.assertEqual((chunk * 16 * 8) % 64, 0)

        scatter = normalize({"kernel": "Scatter", "pattern": [0, 1],
                             "count": 1})
        with self.assertRaisesRegex(ValueError, "requires Gather"):
            prepare(scatter, suite, 0, sms=4, gather_block_cyclic_16=True)
        repeated = normalize({"kernel": "Gather", "pattern": [0, 1],
                              "count": 5, "wrap": 1})
        with self.assertRaisesRegex(ValueError, "conflicts with iteration"):
            prepare(repeated, suite, 0, sms=4, gather_block_cyclic_16=True)

    def test_records_opt_in_robust_end_drain(self):
        case = normalize({"kernel": "Gather", "pattern": [0, 1], "count": 1})
        suite = Path(__file__).parent / "smoke.json"
        default = prepare(case, suite, 0)
        self.assertFalse(default["robust_end_drain"])
        self.assertFalse(default["robust_spin_until_reset"])
        self.assertFalse(default["robust_serial_fence"])
        self.assertFalse(default["robust_token_fence"])
        self.assertEqual(default["robust_post_spin"], 500)
        self.assertFalse(default["diagnostic_readback"])
        self.assertIn("#define NIGHTLY_ROBUST_GMEM_END 0",
                      GENERATED.joinpath("config.h").read_text())
        robust = prepare(case, suite, 0, robust_end_drain=True)
        self.assertTrue(robust["robust_end_drain"])
        self.assertIn("#define NIGHTLY_ROBUST_GMEM_END 1",
                      GENERATED.joinpath("config.h").read_text())

    def test_records_fpga_spin_handshake_and_requires_drain(self):
        case = normalize({"kernel": "Gather", "pattern": [0, 1], "count": 1})
        suite = Path(__file__).parent / "smoke.json"
        robust = prepare(
            case, suite, 0, robust_end_drain=True,
            robust_spin_until_reset=True, robust_serial_fence=True,
            robust_token_fence=True,
            robust_post_spin=20000,
            diagnostic_readback=True,
        )
        self.assertTrue(robust["robust_spin_until_reset"])
        self.assertTrue(robust["robust_serial_fence"])
        self.assertTrue(robust["robust_token_fence"])
        self.assertEqual(robust["robust_post_spin"], 20000)
        self.assertTrue(robust["diagnostic_readback"])
        self.assertIn("#define NIGHTLY_ROBUST_SPIN_UNTIL_RESET 1",
                      GENERATED.joinpath("config.h").read_text())
        self.assertIn("#define NIGHTLY_ROBUST_SERIAL_FENCE 1",
                      GENERATED.joinpath("config.h").read_text())
        self.assertIn("#define NIGHTLY_ROBUST_TOKEN_FENCE 1",
                      GENERATED.joinpath("config.h").read_text())
        self.assertIn("#define NIGHTLY_ROBUST_POST_SPIN 20000u",
                      GENERATED.joinpath("config.h").read_text())
        with self.assertRaisesRegex(ValueError, "requires robust_end_drain"):
            prepare(case, suite, 0, robust_spin_until_reset=True)
        with self.assertRaisesRegex(ValueError, "requires robust_end_drain"):
            prepare(case, suite, 0, robust_serial_fence=True)
        with self.assertRaisesRegex(ValueError, "requires robust_serial_fence"):
            prepare(case, suite, 0, robust_token_fence=True)
        with self.assertRaisesRegex(ValueError, "robust_post_spin"):
            prepare(case, suite, 0, robust_post_spin=-1)

    def test_ordered_schedule_compacts_empty_destinations(self):
        case = normalize({"kernel": "Scatter", "pattern": [3, 3, 10],
                          "count": 1, "collision-policy": "ordered"})
        self.assertEqual(write_destination_groups(case), 2)
        def words(name):
            data = GENERATED.joinpath(name).read_bytes()
            return list(struct.unpack(f"<{len(data) // 4}I", data))
        self.assertEqual(words("group_destinations.bin"), [3, 10])
        self.assertEqual(words("group_offsets.bin"), [0, 2, 3])
        self.assertEqual(words("group_tasks.bin"), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
