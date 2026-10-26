"""Focused tests for the exact-fixture Vortex Nearn differential."""

from __future__ import annotations

from array import array
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
MODULE_PATH = ROOT / "upstream_diff.py"
SPEC = importlib.util.spec_from_file_location("nearn_upstream_diff", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
diff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diff)


class UpstreamDiffTest(unittest.TestCase):
    def test_byte_exact_vortex_pin(self):
        kernel = diff.pinned_kernel()
        self.assertEqual(len(kernel), 662)
        self.assertEqual(diff.sha256_bytes(kernel), diff.VORTEX_KERNEL_SHA256)
        self.assertIn(
            b"*dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+",
            kernel,
        )

    def test_wrapper_embeds_unmodified_kernel_exactly_once(self):
        kernel = diff.pinned_kernel()
        source = diff.native_source(kernel)
        self.assertEqual(source.count(kernel), 1)
        self.assertNotIn(b"fmaf", source)
        self.assertIn(b"#define sqrt sqrtf", source)
        self.assertIn(b"#define get_global_id", source)

    def test_small_fixture_runs_both_cpu_evaluation_modes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            evidence = diff.generate_evidence(
                4096, base / "build", diff.default_vortex_root(), "gcc", "clang"
            )
            diff.validate_evidence(evidence, require_full=False)
            if evidence["source_execution"]["opencl_attempt"]["status"] == "unavailable":
                self.assertEqual(
                    [item["name"] for item in evidence["executions"]],
                    ["cpu-strict", "cpu-contracted"],
                )
            for execution in evidence["executions"]:
                self.assertEqual(
                    execution["comparison"]["records_compared"], 4096
                )
                self.assertEqual(execution["comparison"]["over_one_ulp"], 0)
                self.assertTrue(
                    execution["comparison"]["stable_top_five_exact"]
                )

    def test_comparator_rejects_two_ulp_and_top_order_changes(self):
        expected = array("I", [0x3F800000, 0x40000000, 0x40400000,
                               0x40800000, 0x40A00000])
        actual = array("I", expected)
        actual[0] += 2
        result = diff.compare_words(actual, expected)
        self.assertFalse(result["passed"])
        self.assertEqual(result["over_one_ulp"], 1)
        self.assertEqual(result["maximum_ulp_distance"], 2)

    def test_committed_full_evidence(self):
        path = ROOT / "evidence" / "nearn-upstream-1m-differential.json"
        if not path.exists():
            self.skipTest("full evidence is generated after source tests")
        evidence = json.loads(path.read_text())
        diff.validate_evidence(evidence, require_full=True)


if __name__ == "__main__":
    unittest.main()
