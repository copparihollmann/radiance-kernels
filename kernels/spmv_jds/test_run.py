"""Focused tests for the scalar pinned Vortex/Parboil JDS SpMV port."""

import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


RUN_PATH = Path(__file__).resolve().parent / "run.py"
RUN_SPEC = importlib.util.spec_from_file_location("spmv_jds_run", RUN_PATH)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
run = importlib.util.module_from_spec(RUN_SPEC)
sys.modules[RUN_SPEC.name] = run
RUN_SPEC.loader.exec_module(run)


class SpmvJdsRunTest(unittest.TestCase):
    def test_device_source_is_scalar_and_preserves_equation(self):
        source = RUN_PATH.with_name("kernel.cpp").read_text()
        self.assertIn("sum = sum + d * t;", source)
        self.assertIn("spmv_jds_ptr[k] + jds_row", source)
        for forbidden in ("gemmini", "mx::", "mvin", "mvout", "vsetvli"):
            self.assertNotIn(forbidden, source.lower())

    def test_fixture_is_reproducible_and_records_provenance(self):
        dataset_a = run.fixture_dataset(257, 7)
        dataset_b = run.fixture_dataset(257, 7)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run.prepare(dataset_a, Path(first), sms=4)
            hashes_a = dict(a["generated_sha256"])
            b = run.prepare(dataset_b, Path(second), sms=4)
            self.assertEqual(hashes_a, b["generated_sha256"])
            self.assertEqual(a["expected_digest"], b["expected_digest"])
            self.assertEqual(a["reference_revision"], run.REFERENCE_REVISION)
            self.assertEqual(a["reference_kernel_sha256"],
                             run.REFERENCE_KERNEL_SHA256)
            self.assertEqual(a["reference_kernel_git_blob"],
                             run.REFERENCE_KERNEL_GIT_BLOB)
            self.assertTrue(a["scalar_only"])
            self.assertTrue(a["no_vpu"])
            self.assertFalse(a["vpu"])
            self.assertFalse(a["vector_isa"])
            self.assertFalse(a["gemmini"])
            self.assertEqual(a["generator_model_check"], "passed")
            self.assertEqual(a["partition_nnz"], [340, 335, 336, 342])
            assembly = RUN_PATH.with_name("data.S").read_text()
            self.assertIn(
                '.zero 1028\n.balign 64\n.globl spmv_jds_guard_after', assembly)

    def test_exact_jds_equation_retains_per_row_order(self):
        rows = [
            [(0, run.f32(1.0e20)), (1, run.f32(1.0)),
             (2, run.f32(-1.0e20))],
            [(1, run.f32(2.0))],
            [(2, run.f32(3.0))],
        ] + [[(row, run.f32(1.0))] for row in range(3, 64)]
        jds = run.build_jds(rows)
        output = run.reference_output(jds, [run.f32(1.0)] * 64)
        self.assertEqual(run.float_word(output[0]), run.float_word(0.0))
        self.assertEqual(run.float_word(output[1]), run.float_word(2.0))
        ix = jds.row_to_jds[0]
        self.assertEqual(jds.group_bounds[ix // run.JDS_WARP_SIZE], 3)

    def test_nonzero_balanced_boundaries_are_line_safe_and_complete(self):
        costs = [1] * 128
        costs[:32] = [20] * 32
        boundaries = run.balanced_boundaries(costs, 4)
        run.validate_boundaries(boundaries, len(costs), 4)
        self.assertEqual(boundaries[0], 0)
        self.assertEqual(boundaries[-1], len(costs))
        self.assertTrue(all(value % 16 == 0 for value in boundaries[1:-1]))
        work = run.partition_work(costs, boundaries)
        self.assertEqual(sum(work), sum(costs))
        self.assertLess(max(work), sum(costs[:32]))

    def test_matrix_market_import_mirrors_and_validates_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "tiny.mtx"
            vector = root / "tiny.vec"
            gold = root / "tiny.gold"
            matrix.write_text(
                "%%MatrixMarket matrix coordinate real symmetric\n"
                "% tiny\n"
                "64 64 66\n" +
                "".join(f"{i} {i} 2.0\n" for i in range(1, 65)) +
                "1 2 0.5\n2 3 -0.25\n"
            )
            vector.write_bytes(struct.pack("<64f", *([1.0] * 64)))
            rows, stored = run.read_matrix_market(matrix)
            jds = run.build_jds(rows)
            expected = run.reference_output(jds, [1.0] * 64)
            gold.write_bytes(struct.pack("<I64f", 64, *expected))
            dataset = run.imported_dataset(matrix, vector, gold, "tiny")
            result = run.prepare(dataset, root / "out", sms=2)
            self.assertEqual(stored, 66)
            self.assertEqual(result["expanded_nnz"], 68)
            self.assertEqual(result["upstream_gold_validation"]["bad"], 0)

    def test_pinned_1138_bus_matches_upstream_gold_bit_exactly(self):
        reference = RUN_PATH.parents[4] / "reference" / "vortex"
        if not reference.joinpath(".git").exists():
            self.skipTest("pinned Vortex checkout is not available")
        runs = RUN_PATH.parent / "runs"
        runs.mkdir(exist_ok=True)
        base = "tests/opencl/spmv/"
        with tempfile.TemporaryDirectory(dir=runs) as directory:
            directory = Path(directory)
            paths = {}
            for kind, name in (
                ("matrix", "1138_bus.mtx"),
                ("vector", "1138_bus.vec"),
                ("gold", "1138_bus.gold"),
            ):
                data = subprocess.check_output([
                    "git", "-c", "gc.auto=0", "-C", str(reference),
                    "show", f"{run.REFERENCE_REVISION}:{base}{name}",
                ])
                paths[kind] = directory / name
                paths[kind].write_bytes(data)
            dataset = run.imported_dataset(
                paths["matrix"], paths["vector"], paths["gold"], "1138_bus")
            jds = run.build_jds(dataset.rows)
            expected = run.reference_output(jds, dataset.vector)
            validation = run.compare_upstream(expected, dataset.upstream_gold)
            self.assertEqual(validation["bad"], 0)
            self.assertEqual(validation["max_abs_diff"], 0.0)
            self.assertEqual(sum(jds.row_nnz), 4054)
            self.assertEqual(len(jds.data), 4448)
            self.assertEqual(
                run.digest(map(run.float_word, expected)), 0xE532B7DE1538E91C)

    def test_robust_completion_validation(self):
        dataset = run.fixture_dataset(64, 3)
        with tempfile.TemporaryDirectory() as directory:
            result = run.prepare(
                dataset, Path(directory), robust_end_drain=True,
                robust_spin_until_reset=True, robust_serial_fence=True,
                robust_post_spin=20000,
            )
            self.assertTrue(result["robust_spin_until_reset"])
            self.assertTrue(result["robust_serial_fence"])
            with self.assertRaisesRegex(ValueError, "requires robust_end_drain"):
                run.prepare(dataset, Path(directory), robust_serial_fence=True)


if __name__ == "__main__":
    unittest.main()
