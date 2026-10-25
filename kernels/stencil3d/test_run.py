"""Focused tests for the faithful Vortex stencil3d generator."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

RUN_PATH = Path(__file__).resolve().parent / "run.py"
RUN_SPEC = importlib.util.spec_from_file_location("stencil3d_run", RUN_PATH)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
run = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(run)
AUDIT_PATH = RUN_PATH.parents[1] / "evaluation/fpga_1a/audit_stencil3d_artifact.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("stencil3d_audit", AUDIT_PATH)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
audit = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(audit)


class Stencil3dRunTest(unittest.TestCase):
    def test_reference_generation_is_reproducible(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run.prepare(4, Path(first), sms=4, loader="logical")
            b = run.prepare(4, Path(second), sms=4, loader="logical")
            self.assertEqual(a["input_sha256"], b["input_sha256"])
            self.assertEqual(a["expected_sha256"], b["expected_sha256"])
            self.assertEqual(a["expected_digest"], b["expected_digest"])
            self.assertEqual(a["output_element_count"], 64)
            self.assertEqual(a["reference_revision"], run.REFERENCE_REVISION)
            self.assertEqual(a["max_ulp"], 6)
            config = run.GENERATED.joinpath("config.h").read_text()
            self.assertIn("#define NIGHTLY_CLUSTERS 4", config)
            self.assertIn(
                f"#define STENCIL3D_EXPECTED_DIGEST 0x{a['expected_digest']}ULL",
                config,
            )
            self.assertEqual(
                a["correctness"],
                "complete-output-six-ulp-reference-digest-poison-and-guards",
            )
            self.assertEqual(
                a["kernel_source_sha256"],
                audit.current_source_sha256(RUN_PATH.parents[2]),
            )

    def test_host_poison_and_reference_digest_gates_are_present(self):
        host = (RUN_PATH.parent / "host.cpp").read_text()
        self.assertIn("got[i] = 0x7fc00000u", host)
        self.assertIn("reference_digest == STENCIL3D_EXPECTED_DIGEST", host)
        self.assertIn("reference_digest_ok &&", host)
        self.assertIn("poison == 0", host)

    def test_boundary_replication_matches_constant_volume(self):
        output = run.reference_output([run.f32(0.25)] * 8, 2)
        self.assertEqual([run.float_word(value) for value in output],
                         [run.float_word(run.f32(0.25))] * 8)

    def test_robust_completion_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run.prepare(
                2, Path(directory), robust_end_drain=True,
                robust_spin_until_reset=True, robust_serial_fence=True,
                robust_post_spin=20000,
            )
            self.assertTrue(result["robust_spin_until_reset"])
            self.assertTrue(result["robust_serial_fence"])
            with self.assertRaisesRegex(ValueError, "requires robust_end_drain"):
                run.prepare(2, Path(directory), robust_spin_until_reset=True)


if __name__ == "__main__":
    unittest.main()
