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

class Stencil3dRunTest(unittest.TestCase):
    def test_reference_generation_is_reproducible(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run.prepare(4, Path(first), preset="4sm_pipe", loader="logical")
            b = run.prepare(4, Path(second), preset="4sm_pipe", loader="logical")
            self.assertEqual(a["input_sha256"], b["input_sha256"])
            self.assertEqual(a["expected_sha256"], b["expected_sha256"])
            self.assertEqual(a["expected_digest"], b["expected_digest"])
            self.assertEqual(a["output_element_count"], 64)
            self.assertEqual(a["reference_revision"], run.REFERENCE_REVISION)
            self.assertEqual(a["max_ulp"], 6)
            config = run.GENERATED.joinpath("config.h").read_text()
            self.assertIn(
                f"#define STENCIL3D_EXPECTED_DIGEST 0x{a['expected_digest']}ULL",
                config,
            )
            self.assertEqual(
                a["correctness"],
                "complete-output-six-ulp-reference-digest-poison-and-guards",
            )
            self.assertEqual(len(a["kernel_source_sha256"]), 64)

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

if __name__ == "__main__":
    unittest.main()
