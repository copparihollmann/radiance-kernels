"""Focused tests for the faithful Vortex/Rodinia Hotspot3D port."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
RUN_SPEC = importlib.util.spec_from_file_location("hotspot3d_run", ROOT / "run.py")
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
run = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(run)


class Hotspot3dRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "runs").mkdir(exist_ok=True)

    def temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=ROOT / "runs")

    def test_pinned_reference_objects(self):
        verified = run.reference_objects()
        self.assertEqual(set(verified), set(run.REFERENCE_SOURCES))
        for source, expected in run.REFERENCE_SOURCES.items():
            self.assertEqual(verified[source]["git_blob"], expected["blob"])
            self.assertEqual(verified[source]["sha256"], expected["sha256"])

    def test_upstream_canary_contract_and_reproducibility(self):
        with self.temporary_directory() as first, self.temporary_directory() as second:
            a = run.prepare("canary", 4, Path(first))
            b = run.prepare("canary", 4, Path(second))
            self.assertEqual((a["nx"], a["ny"], a["nz"], a["iterations"]),
                             (16, 16, 4, 2))
            self.assertEqual(a["updates"], 2048)
            self.assertEqual(a["power_sha256"], b["power_sha256"])
            self.assertEqual(a["initial_sha256"], b["initial_sha256"])
            self.assertEqual(a["expected_sha256"], b["expected_sha256"])
            self.assertEqual(a["expected_digest"], b["expected_digest"])
            self.assertEqual(len(a["stage_expected_digests"]), 2)
            self.assertEqual(a["max_ulp"], 32)

    def test_meaningful_contract(self):
        with self.temporary_directory() as directory:
            case = run.prepare("meaningful", 4, Path(directory))
            self.assertEqual((case["nx"], case["ny"], case["nz"]),
                             (64, 64, 8))
            self.assertEqual(case["iterations"], 4)
            self.assertEqual(case["output_element_count"], 32768)
            self.assertEqual(case["updates"], 131072)
            self.assertEqual(case["max_ulp"], 64)
            self.assertEqual(len(case["stage_expected_digests"]), 4)

    def test_constant_field_oracle_with_zero_power(self):
        nx = ny = nz = 2
        coeff = run.coefficients(nx, ny, nz)
        initial = [run.f32(300.0)] * 8
        output = run.reference_output(
            [run.f32(0.0)] * 8, initial, nx, ny, nz, 1, coeff)
        expected = run.f32(0.0)
        for weight in ("cc", "cw", "ce", "cs", "cn", "cb", "ct"):
            expected = run.fadd(expected, run.fmul(coeff[weight], run.f32(300.0)))
        expected = run.fadd(expected,
                            run.fmul(coeff["step_div_cap"], run.f32(0.0)))
        expected = run.fadd(expected,
                            run.fmul(coeff["ct"], run.f32(run.AMBIENT)))
        self.assertEqual([run.float_word(value) for value in output],
                         [run.float_word(expected)] * 8)

    def test_inputs_follow_one_continuous_srand7_stream(self):
        power, temperature = run.reference_inputs(3)
        self.assertEqual(
            [run.float_word(value) for value in temperature],
            [0x439F56A8, 0x43987312, 0x439BA99A],
        )
        self.assertEqual(
            [run.float_word(value) for value in power],
            [0x3E991687, 0x3F5EF9DB, 0x3EE2D0E5],
        )

    def test_partitions_are_exact_once_and_cache_line_exclusive(self):
        for columns in (16 * 16, 64 * 64):
            for sms in (1, 2, 4):
                owners, boundaries = run.ownership(columns, sms)
                self.assertEqual(len(owners), columns)
                self.assertEqual(boundaries[0], 0)
                self.assertEqual(boundaries[-1], columns)
                self.assertTrue(all(boundary % 16 == 0
                                    for boundary in boundaries))
                for begin in range(0, columns, 16):
                    self.assertEqual(len(set(owners[begin:begin + 16])), 1)

    def test_generated_float_constants_are_exact_binary32_hex(self):
        with self.temporary_directory() as directory:
            run.prepare("canary", 2, Path(directory))
            config = (run.GENERATED / "config.h").read_text()
            coeff = run.coefficients(16, 16, 4)
            for name, value in coeff.items():
                macro = "STEP_DIV_CAP" if name == "step_div_cap" else name.upper()
                self.assertIn(
                    f"#define HOTSPOT3D_{macro} {value.hex()}f", config)

    def test_scalar_source_has_no_accelerator_path(self):
        kernel = (ROOT / "kernel.cpp").read_text().lower()
        makefile = (ROOT / "Makefile").read_text().lower()
        self.assertIn("-fno-vectorize", makefile)
        for forbidden in ("gemmini", "mxgemm", "vpu", "riscv_vector.h"):
            self.assertNotIn(forbidden, kernel)

    def test_private_toolchain_fallback_is_resolved(self):
        environment = run.build_environment()
        selected = environment.get("RISCV64_TOOLCHAIN_PATH",
                                   environment.get("RISCV"))
        self.assertIsNotNone(selected)
        self.assertTrue(
            Path(selected, "bin/riscv64-unknown-elf-g++").is_file())

    def test_oracle_binary_is_complete(self):
        with self.temporary_directory() as directory:
            case = run.prepare("canary", 1, Path(directory))
            data = (run.GENERATED / "expected.bin").read_bytes()
            self.assertEqual(len(data), case["output_element_count"] * 4)
            words = struct.unpack(f"<{case['output_element_count']}I", data)
            self.assertEqual(f"{run.fnv64(words):016x}",
                             case["expected_digest"])


if __name__ == "__main__":
    unittest.main()
