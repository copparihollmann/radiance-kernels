"""Focused tests for STREAM input generation and occupancy selection."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run


class StreamRunTest(unittest.TestCase):
    def test_records_selected_warps_per_core(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run.prepare(
                "add", 32, Path(directory), full_host_check=True,
                sms=4, loader="logical", warps_per_core=2,
            )
            self.assertEqual(result["muon_warps_per_core"], 2)
            self.assertEqual(result["sm_count"], 4)
            self.assertIn("#define NIGHTLY_OCC 2", run.GENERATED.joinpath("config.h").read_text())

    def test_rejects_unsupported_warps_per_core(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "warps_per_core"):
                run.prepare("copy", 32, Path(directory), warps_per_core=3)


if __name__ == "__main__":
    unittest.main()
