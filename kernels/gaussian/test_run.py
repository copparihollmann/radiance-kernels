"""Focused tests for the full scalar Vortex/Rodinia Gaussian port."""

from __future__ import annotations

import importlib.util
from fractions import Fraction
from pathlib import Path
import random
import struct
import subprocess
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
        cls.native_dir = ROOT / "runs" / "native-checker-unittest"
        cls.native_dir.mkdir(exist_ok=True)
        cls.native_checker = cls.native_dir / "native_checker_test"
        subprocess.run([
            "g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
            "-fno-fast-math", "-ffp-contract=off", "-frounding-math",
            str(ROOT / "native_checker_test.cpp"),
            "-o", str(cls.native_checker),
        ], check=True)

    def tempdir(self):
        return tempfile.TemporaryDirectory(dir=ROOT / "runs")

    @staticmethod
    def parse_probe(stdout: str) -> dict[str, str]:
        return dict(field.split("=", 1) for field in stdout.strip().split())

    @staticmethod
    def words(path: Path) -> list[int]:
        raw = path.read_bytes()
        return list(struct.unpack(f"<{len(raw) // 4}I", raw))

    @staticmethod
    def fraction(word: int) -> Fraction:
        sign = -1 if word >> 31 else 1
        exponent = (word >> 23) & 0xff
        significand = word & 0x7fffff
        if exponent == 0xff:
            raise ValueError("nonfinite binary32 word")
        if exponent == 0:
            power = -149
        else:
            significand |= 0x800000
            power = exponent - 150
        value = (Fraction(significand << power, 1) if power >= 0 else
                 Fraction(significand, 1 << -power))
        return sign * value

    def run_probe(self, size: int, tolerance_shift: int,
                  original_a: Path, original_b: Path,
                  transformed_a: Path, transformed_b: Path,
                  *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run([
            str(self.native_checker), "check", str(size),
            str(tolerance_shift), str(original_a), str(original_b),
            str(transformed_a), str(transformed_b),
        ], text=True, capture_output=True, check=check)

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

    def test_integer_softfloat_focused_exhaustive_cross_product(self):
        completed = subprocess.run(
            [str(self.native_checker), "selftest"], text=True,
            capture_output=True, check=True)
        self.assertEqual(completed.stdout.strip(), "softfloat_checks=252704")

    def test_actual_checker_accepts_full_meaningful_oracle(self):
        with self.tempdir() as directory:
            case = run.prepare("meaningful", 1, Path(directory))
            completed = self.run_probe(
                64, case["residual_tolerance_power2_shift"],
                run.GENERATED / "original_a.bin",
                run.GENERATED / "original_b.bin",
                run.GENERATED / "expected_a.bin",
                run.GENERATED / "expected_b.bin")
            fields = self.parse_probe(completed.stdout)
            self.assertEqual(fields["back_valid"], "1")
            self.assertEqual(fields["certificate_valid"], "1")
            self.assertEqual(fields["certified"], "1")
            self.assertEqual(fields["solution_digest"],
                             case["expected_solution_digest"])

            # Independently prove that the checker's outward fixed-point
            # coefficient encloses the exact rational residual of the same
            # binary32 solution and original system.
            original_a = self.words(run.GENERATED / "original_a.bin")
            original_b = self.words(run.GENERATED / "original_b.bin")
            solution = self.words(run.GENERATED / "expected_solution.bin")
            exact_residual = Fraction(0)
            for row in range(64):
                residual = sum(
                    self.fraction(original_a[row * 64 + column]) *
                    self.fraction(solution[column])
                    for column in range(64)) - self.fraction(original_b[row])
                exact_residual = max(exact_residual, abs(residual))
            coefficient = int(fields["residual_coeff"])
            exponent = int(fields["residual_exp"])
            bound = (Fraction(coefficient << exponent, 1) if exponent >= 0
                     else Fraction(coefficient, 1 << -exponent))
            self.assertGreaterEqual(bound, exact_residual)

    def test_actual_checker_rejects_finite_semantic_corruption(self):
        with self.tempdir() as directory:
            out = Path(directory)
            case = run.prepare("canary", 1, out)
            corrupted = bytearray(
                (run.GENERATED / "expected_b.bin").read_bytes())
            # Add exactly 1.0 to the final transformed RHS word while keeping
            # every output finite.  This must not pass merely because A/m or
            # the generated CPU-oracle residual remain intact.
            final = struct.unpack_from("<f", corrupted, 12)[0]
            struct.pack_into("<f", corrupted, 12, run.f32(final + 1.0))
            corrupted_path = out / "corrupted_b.bin"
            corrupted_path.write_bytes(corrupted)
            completed = self.run_probe(
                4, case["residual_tolerance_power2_shift"],
                run.GENERATED / "original_a.bin",
                run.GENERATED / "original_b.bin",
                run.GENERATED / "expected_a.bin", corrupted_path,
                check=False)
            self.assertEqual(completed.returncode, 1)
            fields = self.parse_probe(completed.stdout)
            self.assertEqual(fields["back_valid"], "1")
            self.assertEqual(fields["certified"], "0")
            self.assertEqual(fields["certificate_failure"], "7")

    def test_actual_checker_rejects_nonfinite_and_zero_pivot(self):
        with self.tempdir() as directory:
            out = Path(directory)
            case = run.prepare("canary", 1, out)
            for name, replacement, expected_failure in (
                    ("nan", 0x7fc00000, 2), ("zero", 0x00000000, 3)):
                transformed = bytearray(
                    (run.GENERATED / "expected_a.bin").read_bytes())
                struct.pack_into("<I", transformed, 15 * 4, replacement)
                path = out / f"{name}_a.bin"
                path.write_bytes(transformed)
                completed = self.run_probe(
                    4, case["residual_tolerance_power2_shift"],
                    run.GENERATED / "original_a.bin",
                    run.GENERATED / "original_b.bin", path,
                    run.GENERATED / "expected_b.bin", check=False)
                self.assertEqual(completed.returncode, 1)
                fields = self.parse_probe(completed.stdout)
                self.assertEqual(fields["back_valid"], "0")
                self.assertEqual(int(fields["back_failure"]), expected_failure)

    def test_residual_outward_bound_on_python_exact_edge_cases(self):
        finite_edges = (
            0x00000000, 0x80000000, 0x00000001, 0x80000001,
            0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
            0x3e800000, 0xbe800000, 0x3f000000, 0xbf000000,
            0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
            0x7f7fffff, 0xff7fffff,
        )
        generator = random.Random(0x6A55126)
        with self.tempdir() as directory:
            root = Path(directory)
            for case_index in range(64):
                size = 4
                a = [generator.choice(finite_edges)
                     for _ in range(size * size)]
                b = [generator.choice(finite_edges) for _ in range(size)]
                x = [generator.choice(finite_edges) for _ in range(size)]
                # The denominator lower bound must be nonzero by contract.
                b[case_index % size] = 0x3f800000
                paths = []
                for name, words in (("a", a), ("b", b), ("x", x)):
                    path = root / f"{name}.bin"
                    path.write_bytes(struct.pack(f"<{len(words)}I", *words))
                    paths.append(path)
                completed = subprocess.run([
                    str(self.native_checker), "certify", str(size), "16",
                    *(str(path) for path in paths),
                ], text=True, capture_output=True, check=True)
                fields = self.parse_probe(completed.stdout)
                self.assertEqual(fields["certificate_valid"], "1")
                exact = Fraction(0)
                for row in range(size):
                    residual = sum(
                        self.fraction(a[row * size + column]) *
                        self.fraction(x[column]) for column in range(size))
                    residual -= self.fraction(b[row])
                    exact = max(exact, abs(residual))
                coefficient = int(fields["residual_coeff"])
                exponent = int(fields["residual_exp"])
                bound = (Fraction(coefficient << exponent, 1)
                         if exponent >= 0 else
                         Fraction(coefficient, 1 << -exponent))
                self.assertGreaterEqual(bound, exact)
                denominator = max(abs(self.fraction(word)) for word in b)
                certified_from_report = bound * (1 << 16) <= denominator
                self.assertEqual(fields["certified"] == "1",
                                 certified_from_report)

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
            self.assertEqual(case["residual_tolerance_power2_shift"], 12)
            self.assertEqual(
                case["residual_certificate_denominator_lower_bound"],
                "||b||inf")
            self.assertTrue(case["no_vpu"])
            self.assertFalse(case["vpu"])
            self.assertFalse(case["vector_isa"])
            self.assertFalse(case["gemmini"])
            self.assertFalse(case["mx"])
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

    def test_host_acceptance_uses_actual_arrays_not_oracle_constant(self):
        host = (ROOT / "host.cpp").read_text()
        self.assertIn("gaussian_back_substitute_actual", host)
        self.assertIn("gaussian_certify_actual_solution", host)
        self.assertIn("GAUSSIAN_ORIGINAL_A_ADDR", host)
        self.assertIn("residual.certified && owners_ok && guards_ok", host)
        self.assertNotIn("GAUSSIAN_ORACLE_RESIDUAL_E18 <=", host)
        self.assertNotIn("GAUSSIAN_RESIDUAL_E18 <=", host)

    def test_multilaunch_host_emits_one_aggregate_timing_record(self):
        host = (ROOT / "host.cpp").read_text()
        self.assertNotIn("nightly_host_report(", host)
        self.assertEqual(host.count('nh_puts("kernel_cycles=")'), 1)
        self.assertIn("accumulate_stage_cycles(per_core_cycles)", host)
        self.assertIn("report_aggregate_cycles(per_core_cycles, total_cycles)",
                      host)


if __name__ == "__main__":
    unittest.main()
