"""Focused tests for STREAM input generation and occupancy selection."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
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
            self.assertIn("#define NIGHTLY_ROBUST_CLUSTER_L1_MANAGER 0",
                          run.GENERATED.joinpath("config.h").read_text())
            self.assertFalse(result["robust_end_drain"])
            self.assertFalse(result["robust_spin_until_reset"])
            self.assertFalse(result["robust_serial_fence"])
            self.assertFalse(result["robust_cluster_l1_manager"])
            self.assertEqual(result["robust_completion_mode"], "disabled")
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

    def test_records_opt_in_cluster_l1_manager_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = run.prepare(
                "copy", 32, Path(directory), robust_end_drain=True,
            )
            result = run.prepare(
                "copy", 32, Path(directory), robust_end_drain=True,
                robust_spin_until_reset=True,
                robust_cluster_l1_manager=True,
            )
            self.assertTrue(result["robust_cluster_l1_manager"])
            self.assertTrue(result["robust_spin_until_reset"])
            self.assertEqual(
                result["robust_completion_mode"],
                "cluster-l1-manager-candidate",
            )
            self.assertNotEqual(result["suite_sha256"], baseline["suite_sha256"])
            config = run.GENERATED.joinpath("config.h").read_text()
            self.assertIn("#define NIGHTLY_ROBUST_CLUSTER_L1_MANAGER 1", config)
            self.assertIn("#define NIGHTLY_ROBUST_SPIN_UNTIL_RESET 1", config)

    def test_cluster_l1_manager_requires_robust_end_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                    ValueError,
                    "robust_cluster_l1_manager requires robust_end_drain"):
                run.prepare(
                    "copy", 32, Path(directory),
                    robust_cluster_l1_manager=True,
                )

    def test_cluster_l1_manager_barriers_are_uniform_and_three_passes_remain(self):
        device = RUN_PATH.parents[2] / "lib/include/nightly/device.h"
        source = device.read_text()
        stream_kernel = RUN_PATH.with_name("kernel.cpp").read_text()
        self.assertNotIn("mu_barrier(", stream_kernel)
        self.assertNotIn("NIGHTLY_BAR_ROBUST_DRAIN", stream_kernel)
        candidate = source.split(
            "#if NIGHTLY_ROBUST_CLUSTER_L1_MANAGER\n", 1)[1].split(
                "#else\n  if (warp < MU_NUM_CORES && lane == 0u) {\n"
                "    rad_pause(RAD_EPI_QUEUE_SPIN);", 1)[0]
        self.assertEqual(candidate.count("nightly_l0d_drain();"), 3)
        self.assertEqual(candidate.count("nightly_l1_drain();"), 3)
        self.assertEqual(
            candidate.count("mu_barrier(NIGHTLY_BAR_ROBUST_DRAIN, warps);"),
            6,
        )
        self.assertNotRegex(
            candidate,
            r"(?s)if \([^)]*\) \{[^}]*mu_barrier\(",
        )
        default = source.split(
            "#else\n  if (warp < MU_NUM_CORES && lane == 0u) {\n"
            "    rad_pause(RAD_EPI_QUEUE_SPIN);", 1)[1].split(
                "#endif\n  mu_barrier(NIGHTLY_BAR_END - 1u, warps);", 1)[0]
        self.assertEqual(default.count("nightly_l0d_drain();"), 3)
        self.assertEqual(default.count("nightly_l1_drain();"), 3)
        self.assertEqual(default.count("rad_pause(RAD_EPI_SETTLE);"), 2)
        self.assertEqual(default.count("rad_pause(NIGHTLY_ROBUST_POST_SPIN);"), 1)

    def test_cli_threads_cluster_l1_manager_and_rejects_missing_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "candidate"
            command = [
                sys.executable, str(RUN_PATH), "--kind", "copy",
                "--elements", "32", "--out", str(out), "--prepare-only",
                "--robust-end-drain", "--robust-cluster-l1-manager",
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads(out.joinpath("result.json").read_text())
            self.assertTrue(manifest["robust_cluster_l1_manager"])
            self.assertEqual(
                manifest["robust_completion_mode"],
                "cluster-l1-manager-candidate",
            )

            invalid = subprocess.run(
                [
                    sys.executable, str(RUN_PATH), "--kind", "copy",
                    "--elements", "32", "--out", str(out), "--prepare-only",
                    "--robust-cluster-l1-manager",
                ],
                capture_output=True, text=True,
            )
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn(
                "--robust-cluster-l1-manager requires --robust-end-drain",
                invalid.stderr,
            )


if __name__ == "__main__":
    unittest.main()
