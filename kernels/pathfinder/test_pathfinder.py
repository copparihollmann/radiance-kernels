"""Focused deterministic tests for the faithful scalar Pathfinder port."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
RUN_SPEC = importlib.util.spec_from_file_location("pathfinder_run", ROOT / "run.py")
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
run = importlib.util.module_from_spec(RUN_SPEC)
sys.modules[RUN_SPEC.name] = run
RUN_SPEC.loader.exec_module(run)

MODEL_SPEC = importlib.util.spec_from_file_location(
    "pathfinder_model_check", ROOT / "model_check.py")
assert MODEL_SPEC is not None and MODEL_SPEC.loader is not None
model = importlib.util.module_from_spec(MODEL_SPEC)
MODEL_SPEC.loader.exec_module(model)


class PathfinderTest(unittest.TestCase):
    def test_pinned_reference_hashes_and_git_objects(self):
        reference = ROOT.parents[3] / "reference" / "vortex"
        if not reference.joinpath(".git").exists():
            self.skipTest("pinned Vortex object store is unavailable")
        self.assertEqual(subprocess.check_output(
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

    def test_upstream_fixture_uses_exact_libc_rand_stream(self):
        wall = run.fixture_wall(2, 16)
        self.assertEqual(wall, [
            5, 4, 5, 7, 0, 3, 0, 8, 2, 2, 6, 3, 8, 9, 7, 5,
            9, 0, 6, 9, 5, 0, 1, 5, 9, 4, 4, 3, 9, 9, 0, 4,
        ])

    def test_cpu_oracle_three_neighbor_and_edge_semantics(self):
        wall = [5, 1, 9, 3, 4, 8, 2, 7, 6, 0, 5, 1]
        # row 1: [5, 9, 3, 10]; row 2: [11, 3, 8, 4]
        self.assertEqual(run.cpu_oracle(wall, 3, 4), [11, 3, 8, 4])

    def test_partitioned_model_matches_independent_oracle(self):
        for rows, cols in ((2, 64), (7, 64), (16, 64), (17, 128)):
            wall = run.fixture_wall(rows, cols)
            expected = run.cpu_oracle(wall, rows, cols)
            for sms in (1, 2, 4):
                actual, owners = run.partitioned_model(wall, rows, cols, sms)
                self.assertEqual(actual, expected)
                self.assertEqual(len(owners), cols)

    def test_one_two_four_sm_partitions_are_exact_and_line_exclusive(self):
        for cols in (64, 128, 4096):
            for sms in (1, 2, 4):
                owners, boundaries = run.ownership(cols, sms)
                self.assertEqual(boundaries[0], 0)
                self.assertEqual(boundaries[-1], cols)
                self.assertEqual(set(owners), set(range(sms)))
                for boundary in boundaries:
                    self.assertEqual(boundary % 16, 0)
                for begin in range(0, cols, 16):
                    self.assertEqual(len(set(owners[begin:begin + 16])), 1)

    def test_exact_work_counts_and_redundant_halo_overlap(self):
        expected = {
            1: (4096, 0, 1_044_480, 0, 1_052_672),
            2: (4606, 510, 1_109_250, 64_770, 1_117_952),
            4: (5626, 1530, 1_238_790, 194_310, 1_248_512),
        }
        for sms, values in expected.items():
            accounting = run.work_accounting(256, 4096, sms, 8)
            self.assertEqual(
                (accounting["seed_loads"], accounting["redundant_seed_loads"],
                 accounting["dp_updates"], accounting["redundant_dp_updates"],
                 accounting["total_useful_iterations"]), values)
            self.assertEqual(accounting["unique_seed_columns"], 4096)
            self.assertEqual(accounting["unique_dp_updates"], 255 * 4096)
            self.assertEqual(accounting["final_writes"], 4096)
            self.assertEqual(accounting["redundant_final_writes"], 0)
            self.assertEqual(
                sum(item["total_useful_iterations"]
                    for item in accounting["per_sm"]),
                accounting["total_useful_iterations"])
            self.assertTrue(accounting["all_threads_have_useful_work_every_phase"])

    def test_phase_overlap_matches_explicit_column_multiplicity(self):
        rows, cols, sms = 16, 64, 4
        accounting = run.work_accounting(rows, cols, sms, 8)
        for phase in accounting["rows"]:
            visits = [0] * cols
            for interval in phase["intervals"]:
                for column in range(interval["begin"], interval["end"]):
                    visits[column] += 1
            self.assertTrue(all(count >= 1 for count in visits))
            self.assertEqual(sum(visits), phase["total_work"])
            self.assertEqual(sum(count - 1 for count in visits),
                             phase["redundant_work"])
            self.assertEqual(max(visits), phase["max_multiplicity"])

    def test_strided_thread_counts_are_exact_and_balanced(self):
        for begin, end, threads in ((0, 4096, 256), (17, 1551, 256),
                                    (33, 64, 256)):
            counts = run.strided_thread_counts(begin, end, threads)
            self.assertEqual(sum(counts), end - begin)
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_prepare_canary_and_meaningful_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            case = run.prepare(16, 64, Path(temp) / "canary", sms=4,
                               preset="upstream-canary")
            self.assertTrue(case["is_upstream_sized_canary"])
            self.assertFalse(case["is_meaningful_workload"])
            self.assertEqual(case["correctness"],
                             "complete-exact-final-row-and-owner-check")
            self.assertTrue(case["partition_complete"])
            self.assertTrue(case["partition_cache_line_safe"])
            self.assertTrue(case["scalar_only"])
            self.assertTrue(case["no_vpu"])
            self.assertFalse(case["vpu"])
            config = (ROOT / "generated/config.h").read_text()
            self.assertIn("#define NIGHTLY_CLUSTERS 4", config)
            assembly = (ROOT / "data.S").read_text()
            self.assertIn(".globl pathfinder_scratch", assembly)
            self.assertIn(".globl pathfinder_guard_middle", assembly)
            meaningful = run.prepare(256, 4096, Path(temp) / "meaningful",
                                     sms=4, preset="meaningful",
                                     loader="direct-hashed",
                                     robust_end_drain=True,
                                     robust_spin_until_reset=True,
                                     robust_serial_fence=True,
                                     robust_post_spin=20000)
            self.assertTrue(meaningful["is_meaningful_workload"])
            self.assertEqual(meaningful["cells"], 1_048_576)
            self.assertTrue(meaningful["work_accounting"]
                            ["all_threads_have_useful_work_every_phase"])
            self.assertEqual(meaningful["work_accounting"]
                             ["uniform_barrier_count_per_sm"], 257)
            self.assertTrue(meaningful["gpu_address_hash"])
            self.assertTrue(meaningful["robust_end_drain"])
            self.assertTrue(meaningful["robust_spin_until_reset"])
            self.assertTrue(meaningful["robust_serial_fence"])
            self.assertEqual(meaningful["robust_post_spin"], 20000)

    def test_prepare_is_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run.prepare(16, 64, Path(first), sms=2,
                            preset="upstream-canary")
            hashes = dict(a["generated_sha256"])
            b = run.prepare(16, 64, Path(second), sms=2,
                            preset="upstream-canary")
            self.assertEqual(hashes, b["generated_sha256"])
            self.assertEqual(a["output_digest"], b["output_digest"])
            self.assertEqual(a["owner_digest"], b["owner_digest"])

    def test_model_config_checks_complete_outputs_and_guards(self):
        fake = {
            "sm_count": 4, "muon_warps_per_core": 8,
            "output_words": 64, "owner_words": 64,
            "output_digest": "0123456789abcdef",
            "owner_digest": "fedcba9876543210",
        }
        config = model.make_config(fake, Path("/tmp/pathfinder.elf"),
                                   0x10200000, 0x10201000, 0x101fff00,
                                   0x10200f00, 0x10202000, 99)
        self.assertIn("timing = false", config)
        self.assertIn("num_clusters = 4", config)
        self.assertEqual(config.count("expected_fnv64"), 5)
        self.assertIn("pathfinder-final-row", config)
        self.assertIn("pathfinder-owners", config)

    def test_device_is_scalar_and_preserves_dp_structure(self):
        source = (ROOT / "kernel.cpp").read_text().lower()
        for required in ("minimum3", "pathfinder_rows", "pathfinder_wall",
                         "pathfinder_scratch", "mu_contiguous_partition"):
            self.assertIn(required, source)
        for forbidden in ("gemmini", "mx::", "vsetvli", "mvin", "mvout"):
            self.assertNotIn(forbidden, source)

    def test_device_has_no_tid_zero_only_work_and_uniform_phase_sync(self):
        source = (ROOT / "kernel.cpp").read_text()
        compact = "".join(source.split())
        self.assertNotIn("if(tid==0", compact)
        self.assertNotIn("if(tid==0u", compact)
        self.assertEqual(source.count("column += threads_per_block"), 3)
        self.assertEqual(source.count("pathfinder_phase_sync(warps);"), 3)
        helper_begin = source.index("static void pathfinder_phase_sync")
        helper_end = source.index("static void kernel_body", helper_begin)
        helper = source[helper_begin:helper_end]
        self.assertIn("mu_fence();", helper)
        self.assertIn("mu_barrier(kPathfinderPhaseBarrier, warps);", helper)
        self.assertNotIn("if (", helper)
        self.assertIn("kPathfinderPhaseBarrier = 12u", source)
        self.assertNotIn("kPathfinderPhaseBarrier = 13u", source)
        self.assertNotIn("kPathfinderPhaseBarrier = 14u", source)
        self.assertNotIn("kPathfinderPhaseBarrier = 15u", source)

    def test_parameter_validation(self):
        with self.assertRaisesRegex(ValueError, "one cache line per SM"):
            run.validate_parameters(16, 32, 4, "logical", 8)
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            run.validate_parameters(16, 70, 2, "logical", 8)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "requires robust_end_drain"):
                run.prepare(16, 64, Path(temp),
                            robust_serial_fence=True)


if __name__ == "__main__":
    unittest.main()
