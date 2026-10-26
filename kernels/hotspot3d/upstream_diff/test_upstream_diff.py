"""Focused tests for the exact-fixture Vortex Hotspot3D differential."""

from __future__ import annotations

from array import array
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
MODULE_PATH = ROOT / "upstream_diff.py"
SPEC = importlib.util.spec_from_file_location("hotspot3d_upstream_diff", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
diff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diff)


class UpstreamDiffTest(unittest.TestCase):
    def test_byte_exact_vortex_pin(self):
        kernel = diff.pinned_kernel()
        expected = diff.VORTEX_OBJECTS[diff.VORTEX_KERNEL_PATH]
        self.assertEqual(len(kernel), 1686)
        self.assertEqual(diff.sha256_bytes(kernel), expected["sha256"])
        self.assertEqual(diff.git_blob_sha1(kernel), expected["blob"])

    def test_local_checkout_verifies_all_pinned_objects(self):
        root = diff.default_vortex_root()
        if root is None:
            self.skipTest("local Vortex checkout is optional")
        result = diff.verify_vortex_checkout(root, diff.pinned_kernel())
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["kernel_byte_exact"])
        self.assertEqual(set(result["objects"]), set(diff.VORTEX_OBJECTS))

    def test_wrapper_embeds_unmodified_kernel_exactly_once(self):
        kernel = diff.pinned_kernel()
        wrapper = diff.native_source(kernel)
        self.assertEqual(wrapper.count(kernel), 1)
        self.assertNotIn(b"fmaf", wrapper)
        self.assertIn(b"#define get_global_id(dimension)", wrapper)
        self.assertIn(
            b"cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]",
            kernel,
        )

    def test_radiance_fixture_identity(self):
        radiance = diff.load_radiance_run()
        dimensions = radiance.PROFILES["meaningful"]
        self.assertEqual(
            dimensions,
            {"nx": 64, "ny": 64, "nz": 8, "iterations": 4},
        )
        coefficients = radiance.coefficients(64, 64, 8)
        words = {
            name: f"{diff.float_to_word(value):08x}"
            for name, value in coefficients.items()
        }
        self.assertEqual(words, diff.COEFFICIENT_WORDS)

    def test_strict_comparator_rejects_one_ulp(self):
        expected = array("I", [0x43A00000] * 8)
        actual = array("I", expected)
        actual[3] += 1
        result = diff.compare_words(actual, expected, 1, exact=True)
        self.assertFalse(result["passed"])
        self.assertEqual(result["over_limit"], 1)
        self.assertEqual(result["maximum_ulp_distance"], 1)

    def test_semantic_comparator_rejects_limit_plus_one_and_nonfinite(self):
        expected = array("I", [0x43A00000] * 8)
        actual = array("I", expected)
        actual[2] += diff.stage_ulp_limit(4) + 1
        actual[5] = 0x7FC00000
        result = diff.compare_words(actual, expected, 4, exact=False)
        self.assertFalse(result["passed"])
        self.assertEqual(result["over_limit"], 2)
        self.assertEqual(result["nonfinite"], 1)

    def test_full_differential_reproduces_every_step(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            evidence = diff.generate_evidence(
                Path(directory) / "build",
                diff.default_vortex_root(),
                "gcc",
                "clang",
            )
            diff.validate_evidence(evidence)
            executions = {item["name"]: item for item in evidence["executions"]}
            strict = executions["cpu-strict"]
            self.assertEqual(
                [stage["exact_word_matches"] for stage in strict["steps"]],
                [diff.ELEMENTS] * diff.ITERATIONS,
            )
            self.assertEqual(
                [stage["maximum_ulp_distance"] for stage in strict["steps"]],
                [0] * diff.ITERATIONS,
            )
            self.assertEqual(
                evidence["radiance_fixture"]["state_sha256_initial_through_step4"],
                diff.STATE_SHA256,
            )

    def test_committed_machine_evidence(self):
        path = ROOT / "evidence" / "hotspot3d-upstream-differential.json"
        if not path.exists():
            self.skipTest("machine evidence is generated after source tests")
        evidence = json.loads(path.read_text())
        diff.validate_evidence(evidence)
        self.assertLessEqual(
            evidence["resource_policy"]["hard_bytes"],
            4 * 1024**3,
        )


if __name__ == "__main__":
    unittest.main()
