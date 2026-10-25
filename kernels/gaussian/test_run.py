"""Focused tests for the full scalar Vortex/Rodinia Gaussian port."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("gaussian_run", ROOT / "run.py")
assert SPEC is not None and SPEC.loader is not None
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


class GaussianRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "runs").mkdir(exist_ok=True)

    def tempdir(self):
        return tempfile.TemporaryDirectory(dir=ROOT / "runs")

    def test_pinned_reference_objects(self):
        verified = run.reference_objects()
        self.assertEqual(set(verified), set(run.REFERENCE_SOURCES))
        for source, expected in run.REFERENCE_SOURCES.items():
            self.assertEqual(verified[source]["git_blob"], expected["blob"])
            self.assertEqual(verified[source]["sha256"], expected["sha256"])

    def test_canary_is_exact_upstream_matrix4(self):
        a, b = run.fixture("canary")
        self.assertEqual(len(a), 16)
        self.assertEqual([run.float_word(v) for v in a[:4]],
                         [0xBF19999A, 0xBF000000, 0x3F333333, 0x3E99999A])
        self.assertEqual([run.float_word(v) for v in b],
                         [0xBF59999A, 0xBF2E147B, 0x3E75C28F, 0xBF07AE14])

    def test_known_two_by_two_oracle(self):
        a = [run.f32(4), run.f32(2), run.f32(2), run.f32(3)]
        b = [run.f32(6), run.f32(5)]
        result = run.oracle(a, b, 2)
        self.assertEqual(result["m"][2], run.f32(0.5))
        self.assertEqual(result["a"], [run.f32(4), run.f32(2),
                                       run.f32(0), run.f32(2)])
        self.assertEqual(result["b"], [run.f32(6), run.f32(2)])
        self.assertEqual(run.back_substitution(result["a"], result["b"], 2),
                         [run.f32(1), run.f32(1)])

    def test_every_pivot_has_fan1_then_fan2(self):
        a, b = run.fixture("canary")
        result = run.oracle(a, b, 4)
        self.assertEqual(len(result["stage_digests"]), 6)
        self.assertEqual(len(result["model_a"]), 6 * 16)
        self.assertEqual(len(result["model_b"]), 6 * 4)
        self.assertEqual(len(result["model_m"]), 6 * 16)

    def test_ownership_is_exact_once_and_cache_line_safe(self):
        for size in (4, 64):
            for sms in (1, 2, 4):
                ownership = run.validate_ownership(size, sms)
                self.assertEqual(ownership["stride"] % 16, 0)
                self.assertEqual(len(ownership["all"]),
                                 2 * (size - 1) * ownership["stride"])

    def test_meaningful_contract_and_residual(self):
        with self.tempdir() as directory:
            case = run.prepare(
                "meaningful", 4, Path(directory), loader="direct-hashed",
                robust_end_drain=True, robust_spin_until_reset=True,
                robust_serial_fence=True, robust_post_spin=20000)
            self.assertEqual(case["size"], 64)
            self.assertEqual(case["stages"], 126)
            self.assertEqual(case["meaningful_vs_upstream_default_cubic_work_ratio"], 8.0)
            self.assertLessEqual(case["relative_backward_error"],
                                 case["residual_tolerance"])
            self.assertEqual(case["fan1_stages"], 63)
            self.assertEqual(case["fan2_stages"], 63)
            self.assertEqual(case["loader"], "direct-hashed")
            self.assertTrue(case["gpu_address_hash"])
            self.assertTrue(case["robust_end_drain"])
            self.assertTrue(case["robust_spin_until_reset"])
            self.assertTrue(case["robust_serial_fence"])
            self.assertEqual(case["robust_post_spin"], 20000)
            config = (run.GENERATED / "config.h").read_text()
            self.assertIn("#define NIGHTLY_ROBUST_GMEM_END 1", config)
            self.assertIn("#define NIGHTLY_ROBUST_SPIN_UNTIL_RESET 1", config)

    def test_robust_completion_requires_end_drain(self):
        with self.tempdir() as directory:
            with self.assertRaisesRegex(ValueError, "requires robust_end_drain"):
                run.prepare("canary", 1, Path(directory),
                            robust_spin_until_reset=True)

    def test_reproducible_generation(self):
        with self.tempdir() as first, self.tempdir() as second:
            a = run.prepare("canary", 2, Path(first))
            b = run.prepare("canary", 2, Path(second))
            self.assertEqual(a["generated_sha256"], b["generated_sha256"])
            self.assertEqual(a["final_digests"], b["final_digests"])
            self.assertEqual(a["owner_digest"], b["owner_digest"])

    def test_generated_model_snapshots_are_complete(self):
        with self.tempdir() as directory:
            case = run.prepare("canary", 1, Path(directory))
            size, stages = case["size"], case["stages"]
            self.assertEqual((run.GENERATED / "model_a.bin").stat().st_size,
                             stages * size * size * 4)
            self.assertEqual((run.GENERATED / "model_b.bin").stat().st_size,
                             stages * size * 4)
            self.assertEqual((run.GENERATED / "model_m.bin").stat().st_size,
                             stages * size * size * 4)

    def test_final_binary_digests_match_manifest(self):
        with self.tempdir() as directory:
            case = run.prepare("canary", 1, Path(directory))
            for key, filename in (("a", "expected_a.bin"),
                                  ("b", "expected_b.bin"),
                                  ("m", "expected_m.bin")):
                raw = (run.GENERATED / filename).read_bytes()
                words = struct.unpack(f"<{len(raw) // 4}I", raw)
                self.assertEqual(f"{run.fnv64(words):016x}",
                                 case["final_digests"][key])

    def test_scalar_source_has_no_accelerator_path(self):
        kernel = (ROOT / "kernel.cpp").read_text().lower()
        makefile = (ROOT / "Makefile").read_text().lower()
        self.assertIn("-fno-vectorize", makefile)
        for forbidden in ("gemmini", "mxgemm", "riscv_vector.h"):
            self.assertNotIn(forbidden, kernel)


if __name__ == "__main__":
    unittest.main()
