"""Focused tests for STREAM input generation and occupancy selection."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

RUN_PATH = Path(__file__).resolve().parent / "run.py"
RUN_SPEC = importlib.util.spec_from_file_location("stream_run", RUN_PATH)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
run = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(run)


class StreamRunTest(unittest.TestCase):
    def test_records_selected_warps_per_core(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run.prepare(
                "add", 32, Path(directory), full_host_check=True,
                sms=4, loader="logical", warps_per_core=2,
            )
            self.assertEqual(result["muon_warps_per_core"], 2)
            self.assertEqual(result["sm_count"], 4)
            self.assertEqual(result["stream_scalar"], 3.0)
            self.assertEqual(result["benchmark_contract"],
                             "stream-v5.10-elemental-equation")
            self.assertIn("#define NIGHTLY_OCC 2", run.GENERATED.joinpath("config.h").read_text())
            self.assertIn("#define STREAM_SCALAR 0x1.8000000000000p+1f",
                          run.GENERATED.joinpath("config.h").read_text())
            self.assertIn("#define NIGHTLY_ROBUST_GMEM_END 0",
                          run.GENERATED.joinpath("config.h").read_text())
            self.assertFalse(result["robust_end_drain"])
            self.assertFalse(result["robust_spin_until_reset"])
            self.assertFalse(result["robust_serial_fence"])
            self.assertEqual(result["robust_post_spin"], 500)
            self.assertFalse(result["diagnostic_readback"])
            self.assertFalse(result["stream_block_cyclic_lines"])
            self.assertEqual(result["stream_block_cyclic_chunk_lines"], 0)

    def test_records_opt_in_cache_line_block_cyclic_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = run.prepare(
                "copy", 64, Path(directory), full_host_check=True,
                sms=4,
            )
            result = run.prepare(
                "copy", 64, Path(directory), full_host_check=True,
                sms=4, block_cyclic_lines=True,
            )
            self.assertTrue(result["stream_block_cyclic_lines"])
            self.assertNotEqual(result["suite_sha256"], baseline["suite_sha256"])
            self.assertIn("#define STREAM_BLOCK_CYCLIC_LINES 1",
                          run.GENERATED.joinpath("config.h").read_text())

    def test_records_chunked_cache_line_block_cyclic_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run.prepare(
                "scale", 1024, Path(directory), full_host_check=True,
                sms=4, block_cyclic_chunk_lines=64,
            )
            self.assertEqual(result["stream_block_cyclic_chunk_lines"], 64)
            config = run.GENERATED.joinpath("config.h").read_text()
            self.assertIn("#define STREAM_BLOCK_CYCLIC_CHUNK_LINES 64u", config)
            self.assertIn("#define STREAM_BLOCK_CYCLIC_LINES 0", config)
            with self.assertRaisesRegex(ValueError, "only one"):
                run.prepare(
                    "scale", 1024, Path(directory), block_cyclic_lines=True,
                    block_cyclic_chunk_lines=64,
                )
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                run.prepare(
                    "scale", 1024, Path(directory),
                    block_cyclic_chunk_lines=-1,
                )

    def test_nonstandard_scalar_is_explicitly_stream_like(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run.prepare("triad", 32, Path(directory), scalar=2.0)
            self.assertEqual(result["stream_scalar"], 2.0)
            self.assertEqual(result["benchmark_contract"], "stream-like")

    def test_rejects_unsupported_warps_per_core(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "warps_per_core"):
                run.prepare("copy", 32, Path(directory), warps_per_core=3)

    def test_records_robust_end_and_diagnostic_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run.prepare(
                "triad", 32, Path(directory), sms=4,
                robust_end_drain=True, diagnostic_readback=True,
            )
            self.assertTrue(result["robust_end_drain"])
            self.assertTrue(result["diagnostic_readback"])
            self.assertIn("#define NIGHTLY_ROBUST_GMEM_END 1",
                          run.GENERATED.joinpath("config.h").read_text())

    def test_records_fpga_spin_handshake_and_requires_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run.prepare(
                "triad", 32, Path(directory), sms=4,
                robust_end_drain=True, robust_spin_until_reset=True,
                robust_serial_fence=True,
                robust_post_spin=20000,
            )
            self.assertTrue(result["robust_spin_until_reset"])
            self.assertTrue(result["robust_serial_fence"])
            self.assertEqual(result["robust_post_spin"], 20000)
            self.assertIn("#define NIGHTLY_ROBUST_SPIN_UNTIL_RESET 1",
                          run.GENERATED.joinpath("config.h").read_text())
            self.assertIn("#define NIGHTLY_ROBUST_SERIAL_FENCE 1",
                          run.GENERATED.joinpath("config.h").read_text())
            self.assertIn("#define NIGHTLY_ROBUST_POST_SPIN 20000u",
                          run.GENERATED.joinpath("config.h").read_text())
            with self.assertRaisesRegex(ValueError, "requires robust_end_drain"):
                run.prepare(
                    "triad", 32, Path(directory),
                    robust_spin_until_reset=True,
                )
            with self.assertRaisesRegex(ValueError, "requires robust_end_drain"):
                run.prepare(
                    "triad", 32, Path(directory),
                    robust_serial_fence=True,
                )
            with self.assertRaisesRegex(ValueError, "robust_post_spin"):
                run.prepare("triad", 32, Path(directory), robust_post_spin=-1)


if __name__ == "__main__":
    unittest.main()
