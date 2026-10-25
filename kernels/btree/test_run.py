"""Focused deterministic tests for the faithful Vortex B+Tree port."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
RUN_PATH = ROOT / "run.py"
RUN_SPEC = importlib.util.spec_from_file_location("btree_run", RUN_PATH)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
run = importlib.util.module_from_spec(RUN_SPEC)
sys.modules[RUN_SPEC.name] = run
RUN_SPEC.loader.exec_module(run)

MODEL_PATH = ROOT / "model_check.py"
MODEL_SPEC = importlib.util.spec_from_file_location("btree_model_check", MODEL_PATH)
assert MODEL_SPEC is not None and MODEL_SPEC.loader is not None
model = importlib.util.module_from_spec(MODEL_SPEC)
MODEL_SPEC.loader.exec_module(model)


class BTreeRunTest(unittest.TestCase):
    def test_pinned_reference_hashes_and_git_objects(self):
        reference = ROOT.parents[3] / "reference" / "vortex"
        if not reference.joinpath(".git").exists():
            self.skipTest("pinned Vortex object store is unavailable")
        self.assertEqual(
            subprocess.check_output(
                ["git", "-C", str(reference), "rev-parse", "HEAD"],
                text=True).strip(), run.REFERENCE_REVISION)
        for path, blob, expected_hash in (
            (run.REFERENCE_MAIN, run.REFERENCE_MAIN_GIT_BLOB,
             run.REFERENCE_MAIN_SHA256),
            (run.REFERENCE_KERNEL, run.REFERENCE_KERNEL_GIT_BLOB,
             run.REFERENCE_KERNEL_SHA256),
            (run.REFERENCE_MAKEFILE, run.REFERENCE_MAKEFILE_GIT_BLOB,
             run.REFERENCE_MAKEFILE_SHA256),
        ):
            observed_blob = subprocess.check_output(
                ["git", "-C", str(reference), "rev-parse",
                 f"{run.REFERENCE_REVISION}:{path}"], text=True).strip()
            contents = subprocess.check_output(
                ["git", "-C", str(reference), "show",
                 f"{run.REFERENCE_REVISION}:{path}"])
            self.assertEqual(observed_blob, blob)
            self.assertEqual(hashlib.sha256(contents).hexdigest(), expected_hash)

    def test_default_fixture_matches_pinned_libc_rand_contract(self):
        points, ranges = run.fixture_queries(256, 32, 16, 8)
        self.assertEqual(points[:8], [14, 158, 286, 146, 408, 222, 32, 83])
        self.assertEqual(points[-4:], [58, 91, 311, 72])
        self.assertEqual(ranges[:4],
                         [(498, 508), (438, 440), (104, 110), (456, 462)])
        self.assertEqual(ranges[-2:], [(354, 368), (390, 396)])

    def test_lower_bound_oracle_handles_hits_misses_and_inclusive_ranges(self):
        keys, records = run.fixture_keys(8)
        self.assertEqual(run.point_oracle(keys, records, 6), 100006)
        self.assertEqual(run.point_oracle(keys, records, 7), -1)
        self.assertEqual(run.point_oracle(keys, records, -1), -1)
        self.assertEqual(run.range_oracle(keys, 4, 10), (2, 4))
        self.assertEqual(run.range_oracle(keys, 5, 10), (-1, 0))

    def test_flattened_tree_matches_independent_oracle(self):
        keys, records = run.fixture_keys(256)
        pool, root = run.build_tree(keys)
        nodes = run.flatten_tree(pool, root)
        height = run.tree_height(pool, root)
        self.assertEqual(height, 2)
        self.assertEqual(len(nodes), 21)
        for query in range(-1, 513):
            self.assertEqual(
                run.model_point(nodes, records, height, query),
                run.point_oracle(keys, records, query))
        self.assertEqual(len(nodes[0].pack()), 148)
        self.assertEqual(run.NODE_STRUCT.size, 148)

    def test_one_two_four_sm_partitions_are_exact_and_line_exclusive(self):
        for sms in (1, 2, 4):
            for total in (32, 16):
                owners, boundaries = run.ownership(total, sms)
                self.assertEqual(len(owners), total)
                self.assertEqual(boundaries[0], 0)
                self.assertEqual(boundaries[-1], total)
                self.assertTrue(all(boundary % 4 == 0
                                    for boundary in boundaries))
                for line in range(total // 4):
                    self.assertEqual(
                        len(set(owners[line * 4:(line + 1) * 4])), 1)
                self.assertEqual(set(owners), set(range(sms)))

    def test_prepare_is_deterministic_and_records_full_metadata(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run.prepare(256, 32, 16, 8, Path(first), sms=4)
            hashes_a = dict(a["generated_sha256"])
            b = run.prepare(256, 32, 16, 8, Path(second), sms=4)
            self.assertEqual(hashes_a, b["generated_sha256"])
            self.assertEqual(a["point_output_digest"], b["point_output_digest"])
            self.assertEqual(a["range_output_digest"], b["range_output_digest"])
            self.assertEqual(a["generator_model_check"], "passed")
            self.assertEqual(a["correctness"],
                             "complete-exact-point-and-range-output-check")
            self.assertTrue(a["partition_complete"])
            self.assertTrue(a["partition_cache_line_safe"])
            self.assertEqual(a["partition_overlap"], 0)
            self.assertTrue(a["is_official_default_canary"])
            self.assertTrue(a["scalar_only"])
            self.assertTrue(a["no_vpu"])
            self.assertFalse(a["vpu"])
            self.assertFalse(a["gemmini"])
            self.assertEqual(a["abi_mapping"]["knode_size_bytes"], 148)
            config = (ROOT / "generated/config.h").read_text()
            self.assertIn("#define NIGHTLY_CLUSTERS 4", config)
            assembly = (ROOT / "data.S").read_text()
            self.assertIn(".globl btree_guard_middle", assembly)
            self.assertIn(".zero 512", assembly)
            self.assertIn(".zero 256", assembly)

    def test_output_record_digests_cover_values_ownership_and_tags(self):
        keys, records = run.fixture_keys(256)
        points, ranges = run.fixture_queries(256, 32, 16, 8)
        expected_points = [run.point_oracle(keys, records, query)
                           for query in points]
        expected_ranges = [run.range_oracle(keys, start, end)
                           for start, end in ranges]
        one_point, one_range, _ = run.output_records(
            expected_points, expected_ranges, 1)
        four_point, four_range, _ = run.output_records(
            expected_points, expected_ranges, 4)
        self.assertNotEqual(run.fnv64(word for row in one_point for word in row),
                            run.fnv64(word for row in four_point for word in row))
        self.assertNotEqual(run.fnv64(word for row in one_range for word in row),
                            run.fnv64(word for row in four_range for word in row))

    def test_model_config_checks_two_full_outputs_and_three_guards(self):
        fake = {
            "sm_count": 4, "muon_warps_per_core": 8,
            "point_output_records": 32, "range_output_records": 16,
            "point_output_digest": "0123456789abcdef",
            "range_output_digest": "fedcba9876543210",
        }
        config = model.make_config(fake, Path("/tmp/btree.elf"),
                                   0x10200000, 0x10201000,
                                   0x101fff00, 0x10200f00, 0x10202000, 99)
        self.assertIn("timing = false", config)
        self.assertIn("num_clusters = 4", config)
        self.assertIn("words = 128", config)
        self.assertIn("words = 64", config)
        self.assertEqual(config.count("expected_fnv64"), 5)
        self.assertIn("btree-guard-middle", config)

    def test_device_is_scalar_and_preserves_point_and_range_traversal(self):
        source = (ROOT / "kernel.cpp").read_text().lower()
        for required in ("point_lookup", "leaf_record_index",
                         "btree_point_outputs", "btree_range_outputs",
                         "mu_contiguous_partition"):
            self.assertIn(required, source)
        for forbidden in ("gemmini", "mx::", "vsetvli", "mvin", "mvout"):
            self.assertNotIn(forbidden, source)

    def test_parameter_validation_requires_a_line_per_sm(self):
        with self.assertRaisesRegex(ValueError, "one cache line per SM"):
            run.validate_parameters(256, 8, 16, 8, 4, "logical", 8)
        with self.assertRaisesRegex(ValueError, "multiple of four"):
            run.validate_parameters(256, 18, 16, 8, 2, "logical", 8)


if __name__ == "__main__":
    unittest.main()
