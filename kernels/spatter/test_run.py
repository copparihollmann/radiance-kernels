"""Small serial checks for the Spatter address mapping and input parser."""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run as run_module
from run import (GENERATED, block_cyclic_schedule_receipt, destination, fnv,
                 normalize, parse_pattern, payload, prepare, reference,
                 sample_digest, source_index, source_tag,
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

    def test_parallel_scatter_records_admissible_winner_contract(self):
        raw = {"kernel": "Scatter", "pattern": [0, 0, 1, 0],
               "count": 1, "delta": 8, "wrap": 1,
               "source-tag": 2, "collision-policy": "parallel"}
        case = normalize(raw)
        original_transfers = [
            (source_index(case, i, j), destination(case, i, j))
            for i in range(case["count"])
            for j in range(case["length"])
        ]
        result = prepare(case, Path(__file__).parent / "smoke.json", 0,
                         preset="4sm_pipe")
        self.assertEqual(case["_plan"].schedule, "task")
        self.assertEqual(original_transfers, [(0, 0), (1, 0), (2, 1), (3, 0)])
        self.assertTrue(result["destination_overlap"])
        self.assertTrue(result["admissible_winner_check"])
        self.assertEqual(result["parallel_scatter_output"],
                         "schedule-dependent-non-atomic")
        self.assertEqual(result["expected_nonzero_outputs"], 2)
        config = GENERATED.joinpath("config.h").read_text()
        self.assertIn("#define SPATTER_ADMISSIBLE_WINNER_CHECK 1", config)
        self.assertIn("#define SPATTER_SOURCE_TAG 2u", config)
        self.assertIn("#define SPATTER_SOURCE_INDEX_BASE 0ULL", config)

        ordered = normalize({**raw, "collision-policy": "ordered"})
        owner_result = prepare(
            case, Path(__file__).parent / "smoke.json", 0, preset="4sm_pipe",
            parallel_destination_owner=True)
        self.assertFalse(owner_result["admissible_winner_check"])
        self.assertTrue(owner_result["parallel_destination_owner"])
        self.assertTrue(owner_result["all_original_transfers_executed"])
        self.assertEqual(owner_result["expected_digest"],
                         f"{reference(ordered)[0]:016x}")
        self.assertIsNone(owner_result["expected_nonzero_outputs"])
        self.assertEqual(owner_result["destination_group_count"], 2)
        self.assertEqual(owner_result["address_plan"]["schedule"],
                         "parallel_destination_owner")
        self.assertEqual(owner_result["parallel_scatter_output"],
                         "serial-last-writer-equivalent")
        owner_config = GENERATED.joinpath("config.h").read_text()
        self.assertIn("#define SPATTER_ORDERED_COLLISIONS 0", owner_config)
        self.assertIn("#define SPATTER_DESTINATION_OWNER_GROUPS 1", owner_config)
        self.assertIn("#define SPATTER_ADMISSIBLE_WINNER_CHECK 0", owner_config)
        grouped_tasks = struct.unpack(
            "<4I", GENERATED.joinpath("group_tasks.bin").read_bytes())
        self.assertEqual(sorted(grouped_tasks), list(range(4)))

        with self.assertRaisesRegex(ValueError, "colliding parallel Scatter"):
            prepare(normalize({"kernel": "Gather", "pattern": [0, 1]}),
                    Path(__file__).parent / "smoke.json", 0,
                    parallel_destination_owner=True)

        ordered_result = prepare(
            ordered, Path(__file__).parent / "smoke.json", 0, preset="4sm_pipe")
        self.assertFalse(ordered_result["admissible_winner_check"])
        self.assertIsNone(ordered_result["parallel_scatter_output"])
        self.assertEqual(ordered["_plan"].schedule, "destination_owner")

        gather = normalize({"kernel": "Gather", "pattern": [0, 1],
                            "count": 1})
        gather_result = prepare(
            gather, Path(__file__).parent / "smoke.json", 0, preset="4sm_pipe")
        self.assertFalse(gather_result["admissible_winner_check"])
        self.assertIsNone(gather_result["parallel_scatter_output"])

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
        result = prepare(case, suite, 0, preset="2sm_ddr3_v2", loader="logical",
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
        result = prepare(case, suite, 0, preset="4sm_pipe", warps_per_core=8)
        self.assertEqual(result["muon_warps_per_core"], 8)
        config = GENERATED.joinpath("config.h").read_text()
        with self.assertRaisesRegex(ValueError, "warps_per_core"):
            prepare(case, suite, 0, warps_per_core=3)

    def test_block_cyclic_gather_owns_whole_16_element_chunks(self):
        case = normalize({"kernel": "Gather", "pattern": list(range(73)),
                          "count": 1, "wrap": 1})
        suite = Path(__file__).parent / "smoke.json"
        result = prepare(case, suite, 0, preset="4sm_pipe",
                         gather_block_cyclic_16=True)
        self.assertTrue(result["gather_block_cyclic_16"])
        config = GENERATED.joinpath("config.h").read_text()
        self.assertIn("#define SPATTER_GATHER_BLOCK_CYCLIC_16 1", config)
        self.assertIn("#define SPATTER_GATHER_BLOCK_CYCLIC_ELEMENTS 0u", config)

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
            prepare(scatter, suite, 0, preset="4sm_pipe", gather_block_cyclic_16=True)
        repeated = normalize({"kernel": "Gather", "pattern": [0, 1],
                              "count": 5, "wrap": 1})
        repeated_result = prepare(repeated, suite, 0, preset="4sm_pipe",
                                  gather_block_cyclic_16=True)
        self.assertFalse(repeated_result["gather_iteration_partition"])
        self.assertEqual(repeated_result["gather_scratch_elements"], 0)
        self.assertNotIn("spatter_gather_scratch",
                         (run_module.ROOT / "data.S").read_text())

    def test_full_gather_keeps_original_destination_stream(self):
        case = normalize({"kernel": "Gather", "pattern": list(range(256)),
                          "count": 1024, "wrap": 1})
        suite = Path(__file__).parent / "smoke.json"
        result = prepare(case, suite, 0, preset="4sm_pipe")
        self.assertFalse(result["gather_iteration_partition"])
        self.assertEqual(result["gather_scratch_elements"], 0)
        config = GENERATED.joinpath("config.h").read_text()
        self.assertNotIn("SPATTER_GATHER_ITERATION_PARTITION", config)
        self.assertNotIn("spatter_gather_scratch",
                         (run_module.ROOT / "data.S").read_text())

    def test_records_configurable_block_cyclic_gather_schedule(self):
        case = normalize({"kernel": "Gather", "pattern": list(range(73)),
                          "count": 1, "wrap": 1})
        suite = Path(__file__).parent / "smoke.json"
        result = prepare(case, suite, 0, preset="4sm_pipe",
                         gather_block_cyclic_elements=1024)
        self.assertEqual(result["gather_block_cyclic_elements"], 1024)
        config = GENERATED.joinpath("config.h").read_text()
        self.assertIn("#define SPATTER_GATHER_BLOCK_CYCLIC_ELEMENTS 1024u", config)
        self.assertIn("#define SPATTER_GATHER_BLOCK_CYCLIC_16 0", config)

        with self.assertRaisesRegex(ValueError, "positive multiple of 8"):
            prepare(case, suite, 0, gather_block_cyclic_elements=10)
        with self.assertRaisesRegex(ValueError, "only one"):
            prepare(case, suite, 0, gather_block_cyclic_16=True,
                    gather_block_cyclic_elements=1024)

        scatter = normalize({"kernel": "Scatter", "pattern": [0, 1],
                             "count": 1})
        with self.assertRaisesRegex(ValueError, "requires Gather"):
            prepare(scatter, suite, 0, gather_block_cyclic_elements=1024)

    def test_xrage5_full_tail_coverage_and_three_pass_candidate_metadata(self):
        # Full xRAGE5 has 8,368,968 owners, which is not divisible by 1024.
        # This checks the exact closed-form partition used by kernel.cpp,
        # including the old failure's four indices and the 840-owner tail.
        receipt = block_cyclic_schedule_receipt(8_368_968, 1024, 2)
        self.assertEqual(receipt, {
            "owners": 8_368_968,
            "block_elements": 1024,
            "chunks": 8173,
            "tail_begin": 8_368_128,
            "tail_elements": 840,
            "tail_sm": 0,
            "chunks_by_sm": [4087, 4086],
            "elements_by_sm": [4_184_904, 4_184_064],
            "covered_elements": 8_368_968,
            "coverage": "complete-disjoint-chunks",
        })
        for owner in range(5_632_092, 5_632_096):
            chunk = owner // receipt["block_elements"]
            self.assertEqual(chunk, 5500)
            self.assertEqual(chunk % 2, 0)
            self.assertLess(owner, min((chunk + 1) * 1024,
                                       receipt["owners"]))

        case = normalize({"kernel": "Gather", "pattern": list(range(1041)),
                          "count": 1, "wrap": 1})
        suite = Path(__file__).parent / "smoke.json"
        result = prepare(
            case, suite, 0, preset="2sm_ddr3_v2", loader="direct-hashed",
            warps_per_core=8, gather_block_cyclic_elements=1024, diagnostic_readback=True)
        self.assertEqual(result["gather_block_cyclic_schedule"]["tail_elements"],
                         17)
        self.assertEqual(result["gather_block_cyclic_schedule"]["covered_elements"],
                         1041)
        expected = {
            "sm_count": 2,
            "loader": "direct-hashed",
            "muon_warps_per_core": 8,
            "gather_block_cyclic_elements": 1024,
            "diagnostic_readback": True,
        }
        for key, value in expected.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result["drain_protocol"],
                         {"NIGHTLY_ROBUST_GMEM_END": 1,
                          "NIGHTLY_ROBUST_SPIN_UNTIL_RESET": 1,
                          "NIGHTLY_ROBUST_SERIAL_FENCE": 1,
                          "NIGHTLY_ROBUST_POST_SPIN": 20000})
        config = GENERATED.joinpath("config.h").read_text()
        for define in (
                "#define HPC_SM_COUNT 2u",
                "#define SPATTER_GATHER_BLOCK_CYCLIC_ELEMENTS 1024u"):
            self.assertIn(define, config)

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
