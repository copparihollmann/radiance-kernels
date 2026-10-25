#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).with_name("assemble_manifest.py")
SPEC = importlib.util.spec_from_file_location("assemble_manifest", MODULE)
AM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AM)


class AssembleManifestTests(unittest.TestCase):
    def test_hashes_and_records_existing_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "xrage--scatter-ordered--occ8--4sm"
            build.mkdir()
            (build / "kernel.soc.elf").write_bytes(b"elf")
            result = {
                "status": "built", "sm_count": 4, "loader": "logical",
                "kind": "scatter", "collision_policy": "ordered", "case": 1,
                "destination_group_count": 17, "muon_warps_per_core": 8,
                "robust_end_drain": True, "robust_spin_until_reset": True,
                "robust_serial_fence": True,
                "robust_token_fence": False,
                "robust_post_spin": 20000,
                "correctness": "digest-checked",
                "kernel_revision": "abc", "kernel_source_sha256": "def",
            }
            (build / "result.json").write_text(json.dumps(result))
            manifest = AM.assemble(root, "xrage")
            self.assertFalse(manifest["hardware_rebuilt"])
            self.assertTrue(manifest["no_vpu"])
            self.assertEqual(manifest["artifact_base_revisions"], ["abc"])
            row = manifest["workloads"][0]
            self.assertEqual(row["operation"], "scatter-ordered")
            self.assertEqual(row["destination_group_count"], 17)
            self.assertEqual(row["muon_warps_per_core"], 8)
            self.assertTrue(row["robust_end_drain"])
            self.assertTrue(row["robust_spin_until_reset"])
            self.assertTrue(row["robust_serial_fence"])
            self.assertFalse(row["robust_token_fence"])
            self.assertEqual(row["robust_post_spin"], 20000)
            self.assertEqual(row["elf_sha256"], AM.sha256(build / "kernel.soc.elf"))

    def test_parallel_scatter_uses_canonical_operation(self):
        self.assertEqual(AM.operation({"kind": "scatter",
                                      "collision_policy": "parallel"}),
                         "scatter-parallel")

    def test_stream_completion_candidate_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "stream-copy--4sm"
            build.mkdir()
            (build / "kernel.soc.elf").write_bytes(b"elf")
            result = {
                "status": "built", "sm_count": 4, "loader": "logical",
                "kind": "stream-copy", "correctness": "digest-checked",
                "robust_end_drain": True,
                "robust_cluster_l1_manager": True,
                "robust_completion_mode": "cluster-l1-manager-candidate",
            }
            (build / "result.json").write_text(json.dumps(result))
            row = AM.assemble(root, "stream")["workloads"][0]
            self.assertTrue(row["robust_cluster_l1_manager"])
            self.assertEqual(
                row["robust_completion_mode"],
                "cluster-l1-manager-candidate",
            )

    def test_gather_block_cyclic_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "xrage-gather-blockcyclic16--4sm"
            build.mkdir()
            (build / "kernel.soc.elf").write_bytes(b"elf")
            result = {
                "status": "built", "sm_count": 4, "loader": "logical",
                "kind": "gather", "gather_block_cyclic_16": True,
                "pattern_length": 8368968, "host_full_readback": True,
                "expected_pattern_digest": "pattern",
                "expected_source_digest": "source",
                "correctness": "digest-checked",
            }
            (build / "result.json").write_text(json.dumps(result))
            row = AM.assemble(root, "xrage")["workloads"][0]
            self.assertEqual(row["operation"], "gather")
            self.assertTrue(row["gather_block_cyclic_16"])
            self.assertEqual(row["pattern_length"], 8368968)
            self.assertTrue(row["host_full_readback"])
            self.assertEqual(row["expected_pattern_digest"], "pattern")
            self.assertEqual(row["expected_source_digest"], "source")

    def test_gemv_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "gemv-bf16--4sm"
            build.mkdir()
            (build / "kernel.soc.elf").write_bytes(b"elf")
            result = {
                "status": "built", "sm_count": 4, "loader": "logical",
                "kind": "gemv", "rows": 16, "cols": 4096,
                "ilp": 1, "unroll": 8, "output_words": 8,
                "element_format": "bf16", "no_vpu": True,
                "correctness": "full-output-exact-device-order-digest",
            }
            (build / "result.json").write_text(json.dumps(result))
            row = AM.assemble(root, "gemv_bf16")["workloads"][0]
            self.assertEqual(row["operation"], "gemv")
            self.assertEqual(row["rows"], 16)
            self.assertEqual(row["output_words"], 8)
            self.assertTrue(row["no_vpu"])

    def test_stencil_reference_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "stencil3d--size-64--4sm"
            build.mkdir()
            (build / "kernel.soc.elf").write_bytes(b"elf")
            result = {
                "status": "built", "sm_count": 4, "loader": "logical",
                "size": 64, "output_element_count": 64 ** 3,
                "expected_digest": "1234", "max_ulp": 6,
                "reference_repository": "vortexgpgpu/vortex",
                "reference_revision": "abc", "reference_kernel": "kernel.cpp",
                "reference_seed": 50,
                "correctness": "complete-output-reference-check",
            }
            (build / "result.json").write_text(json.dumps(result))
            row = AM.assemble(root, "stencil3d")["workloads"][0]
            self.assertEqual(row["size"], 64)
            self.assertEqual(row["output_element_count"], 64 ** 3)
            self.assertEqual(row["max_ulp"], 6)
            self.assertEqual(row["reference_seed"], 50)

    def test_sfilter_partition_and_dataset_metadata_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "sfilter--size-1024--4sm"
            build.mkdir()
            (build / "kernel.soc.elf").write_bytes(b"elf")
            result = {
                "status": "built", "sm_count": 4, "loader": "logical",
                "kind": "2d-3x3-spatial-filter", "size": 1024,
                "output_element_count": 1024 * 1024,
                "expected_digest": "abcd", "max_ulp": 4,
                "partition_row_boundaries": [1, 257, 513, 768, 1023],
                "partition_ownership": [{"physical_threadblock_id": 0}],
                "partition_ownership_sha256": "ownership",
                "partition_complete": True, "partition_overlap": False,
                "partition_cache_line_safe": True, "cache_line_bytes": 64,
                "dataset_contract": "derived-1024x1024",
                "dataset_contract_preserved": False,
                "benchmark_contract": "radiance-scaled-sfilter-vortex-equation-derived-input",
                "correctness": "complete-output-four-ulp-zero-boundaries-guards-and-digest",
            }
            (build / "result.json").write_text(json.dumps(result))
            row = AM.assemble(root, "sfilter")["workloads"][0]
            self.assertEqual(row["partition_row_boundaries"],
                             [1, 257, 513, 768, 1023])
            self.assertTrue(row["partition_cache_line_safe"])
            self.assertFalse(row["dataset_contract_preserved"])
            self.assertEqual(row["max_ulp"], 4)

    def test_stream_dot_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "stream-dot--4sm"
            build.mkdir()
            (build / "kernel.soc.elf").write_bytes(b"elf")
            result = {
                "status": "built", "sm_count": 4, "loader": "logical",
                "kind": "stream-dot", "stream_elements": 1048576,
                "output_element_count": 4, "expected_digest": "1234",
                "reference_repository": "UoB-HPC/BabelStream",
                "reference_revision": "abc", "reference_kernel": "dot_kernel",
                "correctness": "exact-device-order-partials-and-guards",
            }
            (build / "result.json").write_text(json.dumps(result))
            row = AM.assemble(root, "stream_dot")["workloads"][0]
            self.assertEqual(row["operation"], "stream-dot")
            self.assertEqual(row["stream_elements"], 1048576)
            self.assertEqual(row["output_element_count"], 4)
            self.assertEqual(row["reference_repository"], "UoB-HPC/BabelStream")

    def test_streamcluster_dual_output_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "streamcluster-pgain--n128-d32--4sm"
            build.mkdir()
            (build / "kernel.soc.elf").write_bytes(b"elf")
            result = {
                "status": "built", "sm_count": 4, "loader": "logical",
                "kind": "rodinia-pgain", "num": 128, "dim": 32, "k": 16,
                "output_work_float_count": 2176,
                "output_switch_logical_byte_count": 128,
                "work_expected_digest": "work",
                "switch_expected_digest": "logical-switch",
                "switch_functional_expected_digest": "physical-switch",
                "switch_storage_bytes": 256,
                "partition_work_cache_line_safe": True,
                "partition_switch_cache_line_safe": True,
                "official_default_contract": True,
                "timing_contract": "fused-memset-plus-pgain-full-sequence",
                "correctness": "complete-dual-output-check",
            }
            (build / "result.json").write_text(json.dumps(result))
            row = AM.assemble(root, "streamcluster")["workloads"][0]
            self.assertEqual(row["operation"], "rodinia-pgain")
            self.assertEqual(row["output_work_float_count"], 2176)
            self.assertEqual(row["switch_storage_bytes"], 256)
            self.assertTrue(row["partition_switch_cache_line_safe"])
            self.assertEqual(row["timing_contract"],
                             "fused-memset-plus-pgain-full-sequence")

    def test_lbm_full_oracle_and_partition_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "lbm--official-32x32x8-one-step-gather--4sm"
            build.mkdir()
            (build / "kernel.soc.elf").write_bytes(b"elf")
            result = {
                "status": "built", "sm_count": 4, "loader": "logical",
                "kind": "d3q19-gather-stream-collide",
                "dimensions": [32, 32, 8],
                "padded_dimensions": [40, 32, 12], "timesteps": 1,
                "active_cells": 8192,
                "destination_populations_checked": 155648,
                "velocity_words_checked": 24576,
                "expected_population_digest": "pop",
                "expected_output_raw_digest": "raw",
                "expected_output_raw_words": 307200,
                "gold_velocity_digest": "velocity",
                "partition": {"cache_line_safe": True, "complete": True},
                "reference_verified": True,
                "official_default_one_timestep_verified": True,
                "correctness": "complete-output-and-upstream-gold",
            }
            device = build / "kernel.radiance.elf"
            device.write_bytes(b"device")
            result["device_elf"] = str(device)
            result["device_elf_sha256"] = AM.sha256(device)
            (build / "result.json").write_text(json.dumps(result))
            row = AM.assemble(root, "lbm")["workloads"][0]
            self.assertEqual(row["operation"],
                             "d3q19-gather-stream-collide")
            self.assertEqual(row["dimensions"], [32, 32, 8])
            self.assertEqual(row["expected_output_raw_words"], 307200)
            self.assertEqual(row["expected_output_raw_digest"], "raw")
            self.assertTrue(row["partition"]["cache_line_safe"])
            self.assertTrue(row["reference_verified"])
            self.assertTrue(row["official_default_one_timestep_verified"])
            self.assertEqual(row["device_elf"], str(device.resolve()))
            self.assertEqual(row["device_elf_sha256"], AM.sha256(device))

    def test_nearn_reference_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "nearn--records-1048576--4sm"
            build.mkdir()
            (build / "kernel.soc.elf").write_bytes(b"elf")
            result = {
                "status": "built", "sm_count": 4, "loader": "logical",
                "kind": "nearest-neighbor-distance", "records": 1048576,
                "output_element_count": 1048576, "expected_digest": "1234",
                "expected_top_indices": [4, 3, 2, 1, 0], "max_ulp": 1,
                "reference_repository": "vortexgpgpu/vortex",
                "reference_revision": "abc", "reference_kernel": "kernel.cl",
                "reference_kernel_sha256": "deadbeef",
                "correctness": "complete-output-ulp-and-top-k-check",
            }
            (build / "result.json").write_text(json.dumps(result))
            row = AM.assemble(root, "nearn")["workloads"][0]
            self.assertEqual(row["records"], 1048576)
            self.assertEqual(row["expected_top_indices"], [4, 3, 2, 1, 0])
            self.assertEqual(row["max_ulp"], 1)
            self.assertEqual(row["reference_kernel_sha256"], "deadbeef")

    def test_spmv_jds_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "spmv-jds--dubcova3--4sm"
            build.mkdir()
            (build / "kernel.soc.elf").write_bytes(b"elf")
            result = {
                "status": "built", "sm_count": 4, "loader": "logical",
                "kind": "sparse-matrix-vector-product",
                "dimension": 146689, "expanded_nnz": 3636649,
                "jds_padded_entries": 3637664,
                "partition_boundaries": [0, 20544, 53184, 89456, 146689],
                "partition_nnz": [909329, 909362, 908874, 909084],
                "partition_complete": True, "partition_overlap": False,
                "accumulation_order": "strict-increasing-k-no-contraction",
                "scalar_only": True, "vpu": False, "vector_isa": False,
                "gemmini": False, "mx": False,
                "reference_repository": "vortexgpgpu/vortex",
                "reference_revision": "abc", "reference_kernel": "kernel.cl",
                "reference_kernel_sha256": "deadbeef",
                "reference_main": "main.cc", "reference_main_sha256": "feedface",
                "reference_converter": "convert_dataset.c",
                "reference_converter_sha256": "cafebabe",
                "generated_sha256": {"data.bin": "1234"},
                "correctness": "complete-output-upstream-tolerance-exact-digest-and-guards",
            }
            (build / "result.json").write_text(json.dumps(result))
            row = AM.assemble(root, "spmv_jds")["workloads"][0]
            self.assertEqual(row["operation"], "sparse-matrix-vector-product")
            self.assertEqual(row["dimension"], 146689)
            self.assertEqual(row["partition_nnz"],
                             [909329, 909362, 908874, 909084])
            self.assertTrue(row["partition_complete"])
            self.assertFalse(row["vpu"])
            self.assertFalse(row["vector_isa"])
            self.assertEqual(row["reference_main_sha256"], "feedface")
            self.assertEqual(row["reference_converter_sha256"], "cafebabe")
            self.assertEqual(row["generated_sha256"], {"data.bin": "1234"})


if __name__ == "__main__":
    unittest.main()
