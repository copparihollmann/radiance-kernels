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
                preset="4sm_pipe", loader="logical", warps_per_core=2,
            )
            self.assertEqual(result["muon_warps_per_core"], 2)
            self.assertEqual(result["sm_count"], 4)
            self.assertEqual(result["preset"], "4sm_pipe")
            self.assertEqual(result["stream_scalar"], 3.0)
            self.assertEqual(result["benchmark_contract"],
                             "stream-v5.10-elemental-equation")
            self.assertIn("#define STREAM_SCALAR 0x1.8000000000000p+1f",
                          run.GENERATED.joinpath("config.h").read_text())
            self.assertFalse(result["diagnostic_readback"])
            self.assertFalse(result["stream_block_cyclic_lines"])
            self.assertEqual(result["stream_block_cyclic_chunk_lines"], 0)

    def test_array_offset_pads_between_arrays_without_changing_arithmetic(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = run.prepare(
                "triad", 64, Path(directory), full_host_check=True, preset="2sm_pipe",
            )
            plain = (run.ROOT / "data.S").read_text()
            offset = run.prepare(
                "triad", 64, Path(directory), full_host_check=True, preset="2sm_pipe",
                array_offset_bytes=32768,
            )
            padded = (run.ROOT / "data.S").read_text()
            # Moving a base address must not change what the kernel computes.
            self.assertEqual(offset["expected_digest"], baseline["expected_digest"])
            self.assertEqual(offset["stream_array_offset_bytes"], 32768)
            # The GPU address hash divides a client-space shift by the slice count.
            self.assertEqual(offset["stream_array_offset_slice_local_bytes"], 8192)
            self.assertNotEqual(offset["suite_sha256"], baseline["suite_sha256"])
            self.assertEqual(baseline["stream_array_offset_bytes"], 0)
            self.assertNotIn(".zero 32768", plain)
            # One pad between the two .data inputs, one ahead of the .bss guard,
            # so array k of three is pushed k * 32768 bytes.
            self.assertEqual(padded.count(".zero 32768"), 2)
            # The guard band still sits immediately before the output array.
            self.assertIn("stream_guard_before:\n.zero 64\n.globl stream_a",
                          padded)

    def test_array_offset_rejects_unaligned_values(self):
        with tempfile.TemporaryDirectory() as directory:
            for bad in (-128, 100, 32769):
                with self.assertRaises(ValueError):
                    run.prepare("copy", 64, Path(directory), preset="1sm_pipe",
                                array_offset_bytes=bad)

    def test_records_opt_in_cache_line_block_cyclic_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = run.prepare(
                "copy", 64, Path(directory), full_host_check=True,
                preset="4sm_pipe",
            )
            result = run.prepare(
                "copy", 64, Path(directory), full_host_check=True,
                preset="4sm_pipe", block_cyclic_lines=True,
            )
            self.assertTrue(result["stream_block_cyclic_lines"])
            self.assertNotEqual(result["suite_sha256"], baseline["suite_sha256"])
            self.assertIn("#define STREAM_BLOCK_CYCLIC_LINES 1",
                          run.GENERATED.joinpath("config.h").read_text())

    def test_records_chunked_cache_line_block_cyclic_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run.prepare(
                "scale", 1024, Path(directory), full_host_check=True,
                preset="4sm_pipe", block_cyclic_chunk_lines=64,
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

    def test_split_l1_preserves_three_passes_without_manager_barriers(self):
        device = RUN_PATH.parents[2] / "lib/include/nightly/device.h"
        source = device.read_text()
        split = source.split("#if NIGHTLY_ROBUST_SPLIT_L1_DRAIN\n", 1)[1].split(
            "#else\n  const uint32_t global_core", 1)[0]
        self.assertIn("RAD_L1_DRAIN_MULT / MU_NUM_CORES", split)
        self.assertIn("nightly_cluster_id()", split)
        self.assertIn("vx_core_id()", split)
        default_epilogue = source.split(
            "#else\n  if (warp < MU_NUM_CORES && lane == 0u) {\n"
            "    rad_pause(RAD_EPI_QUEUE_SPIN);", 1)[1].split(
                "#endif\n  mu_barrier(NIGHTLY_BAR_END - 1u, warps);", 1)[0]
        self.assertEqual(default_epilogue.count("nightly_l1_drain();"), 3)
        self.assertNotIn("NIGHTLY_BAR_ROBUST_DRAIN", default_epilogue)

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

    def test_cli_accepts_a_preset_and_records_it(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "candidate"
            completed = subprocess.run(
                [sys.executable, str(RUN_PATH), "--kind", "copy",
                 "--elements", "32", "--out", str(out), "--prepare-only",
                 "--preset", "2sm_ddr3_v3"],
                capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads(out.joinpath("result.json").read_text())
            self.assertEqual(manifest["preset"], "2sm_ddr3_v3")
            self.assertEqual(manifest["sm_count"], 2)
            self.assertEqual(manifest["l1_banks"], 4)
            self.assertFalse(manifest["l1_coherent"])

    def test_cli_rejects_an_unknown_preset(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = subprocess.run(
                [sys.executable, str(RUN_PATH), "--kind", "copy",
                 "--elements", "32", "--out", str(Path(directory) / "x"),
                 "--prepare-only", "--preset", "9sm_imaginary"],
                capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)


if __name__ == "__main__":
    unittest.main()
