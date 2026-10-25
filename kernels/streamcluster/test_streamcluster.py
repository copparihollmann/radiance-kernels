"""Focused lineage, numerical, ownership, and checker tests for pgain."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ORACLE_SPEC = importlib.util.spec_from_file_location(
    "streamcluster_oracle", ROOT / "oracle.py")
assert ORACLE_SPEC is not None and ORACLE_SPEC.loader is not None
oracle = importlib.util.module_from_spec(ORACLE_SPEC)
sys.modules[ORACLE_SPEC.name] = oracle
ORACLE_SPEC.loader.exec_module(oracle)

PORT_SPEC = importlib.util.spec_from_file_location(
    "radiance_streamcluster", ROOT / "streamcluster.py")
assert PORT_SPEC is not None and PORT_SPEC.loader is not None
port = importlib.util.module_from_spec(PORT_SPEC)
PORT_SPEC.loader.exec_module(port)


class StreamClusterTest(unittest.TestCase):
    def test_pinned_reference_commit_blobs_and_content(self):
        verified = port.verify_reference(port.DEFAULT_REFERENCE)
        self.assertEqual(verified["head"], port.REFERENCE_REVISION)
        self.assertEqual(
            verified["files"][port.REFERENCE_KERNEL]["git_blob"],
            port.REFERENCE_KERNEL_GIT_BLOB,
        )
        self.assertEqual(
            verified["files"][port.REFERENCE_MAIN]["git_blob"],
            port.REFERENCE_MAIN_GIT_BLOB,
        )

    def test_glibc_srand9_sequence_is_exact(self):
        generator = oracle.GlibcRand(9)
        self.assertEqual(
            [generator.rand() for _ in range(12)],
            [444454915, 1502197874, 436864165, 770686807,
             1899516530, 994794653, 1552099820, 1351221478,
             1547895242, 1568662052, 116124856, 1982508163],
        )

    def test_official_oracle_hashes_every_output(self):
        dataset = oracle.make_dataset()
        result = oracle.pgain(dataset)
        work = b"".join(struct.pack("<f", value) for value in result.work)
        self.assertEqual(dataset.num, 128)
        self.assertEqual(dataset.dim, 32)
        self.assertEqual(dataset.k, 16)
        self.assertEqual(dataset.candidate, 64)
        self.assertEqual(len(result.work), 128 * 17)
        self.assertEqual(len(result.switch), 128)
        self.assertEqual(hashlib.sha256(work).hexdigest(),
                         port.OFFICIAL_HASHES["work_expected.bin"])
        self.assertEqual(hashlib.sha256(result.switch).hexdigest(),
                         port.OFFICIAL_HASHES["switch_expected.bin"])
        self.assertEqual(result.switch, bytes(128))
        self.assertEqual(
            oracle.fnv_words(oracle.float_word(value) for value in result.work),
            port.OFFICIAL_WORK_DIGEST,
        )
        self.assertEqual(oracle.fnv_bytes(result.switch),
                         port.OFFICIAL_SWITCH_DIGEST)

    def test_oracle_exercises_both_exact_pgain_branches(self):
        dataset = oracle.Dataset(
            num=2, dim=1, seed=9, center_stride=1, candidate=1,
            coordinates=(oracle.f32(0.0), oracle.f32(2.0)),
            points=(
                oracle.Point(oracle.f32(1.0), 0, oracle.f32(5.0)),
                oracle.Point(oracle.f32(1.0), 1, oracle.f32(0.0)),
            ),
            center_table=(0, 1), centers=(0, 1),
        )
        result = oracle.pgain(dataset)
        self.assertEqual(result.switch, b"1\0")
        self.assertEqual(result.work[2], oracle.f32(-1.0))
        self.assertEqual(result.work[4], oracle.f32(0.0))

    def test_all_sm_manifests_are_complete_and_line_safe(self):
        manifests = []
        with tempfile.TemporaryDirectory() as directory:
            for sms in (1, 2, 4):
                manifests.append(port.prepare(
                    128, 32, Path(directory) / f"{sms}sm", sms=sms))
        self.assertEqual([m["partition_boundaries"] for m in manifests],
                         [[0, 128], [0, 64, 128], [0, 32, 64, 96, 128]])
        self.assertEqual([m["switch_segment_stride"] for m in manifests],
                         [128, 64, 64])
        for manifest in manifests:
            self.assertTrue(manifest["partition_complete"])
            self.assertFalse(manifest["partition_overlap"])
            self.assertTrue(manifest["partition_work_cache_line_safe"])
            self.assertTrue(manifest["partition_switch_cache_line_safe"])
            self.assertEqual(manifest["output_work_float_count"], 128 * 17)
            self.assertEqual(manifest["output_switch_logical_byte_count"], 128)
            self.assertEqual(manifest["dataset_hashes"], port.OFFICIAL_HASHES)
            self.assertTrue(manifest["official_default_contract"])
            self.assertEqual(manifest["timing_contract"],
                             "fused-memset-plus-pgain-full-sequence")

    def test_four_sm_cache_lines_are_pairwise_disjoint(self):
        boundaries, owners, stride = port.ownership_metadata(128, 16, 4)
        self.assertEqual(boundaries, [0, 32, 64, 96, 128])
        self.assertEqual(stride, 64)
        work_sets = []
        switch_sets = []
        for owner in owners:
            work_sets.append(set(range(owner["work_byte_begin"] // 64,
                                       owner["work_byte_end"] // 64)))
            sb = owner["switch_storage_byte_begin"]
            se = owner["switch_segment_byte_end"]
            switch_sets.append(set(range(sb // 64, se // 64)))
        for lhs in range(4):
            for rhs in range(lhs + 1, 4):
                self.assertFalse(work_sets[lhs] & work_sets[rhs])
                self.assertFalse(switch_sets[lhs] & switch_sets[rhs])

    def test_derived_size_is_labeled_and_preserves_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = port.prepare(256, 8, Path(directory), sms=4)
        self.assertTrue(manifest["derived"])
        self.assertFalse(manifest["official_default_contract"])
        self.assertIn("derived-size", manifest["benchmark_contract"])
        self.assertEqual(manifest["seed"], 9)
        self.assertEqual(manifest["center_stride"], 8)
        self.assertEqual(manifest["candidate"], 128)

    def test_invalid_or_unpinned_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing"):
                port.verify_reference(Path(directory) / "not-a-repository")
            with self.assertRaisesRegex(ValueError, "divisible"):
                port.prepare(127, 32, Path(directory) / "bad")
            with self.assertRaisesRegex(ValueError, "sms must"):
                port.prepare(128, 32, Path(directory) / "bad-sm", sms=3)

    def test_source_enforces_scalar_complete_checker_contract(self):
        kernel = (ROOT / "kernel_impl.hpp").read_text().lower()
        host = (ROOT / "host.cpp").read_text()
        makefile = (ROOT / "Makefile").read_text()
        for forbidden in ("gemmini", "mxgemm", "vsetvli"):
            self.assertNotIn(forbidden, kernel)
        self.assertIn("#pragma clang fp contract(off)", kernel)
        self.assertIn("physical_threadblock_id", (ROOT / "kernel.cpp").read_text())
        self.assertIn("STREAMCLUSTER_WORK_FLOATS", host)
        self.assertIn("STREAMCLUSTER_NUM", host)
        self.assertIn("switch_padding_bad", host)
        self.assertIn("work_digest_ok", host)
        self.assertIn("guards_intact", host)
        self.assertIn("-fno-vectorize -fno-slp-vectorize", makefile)


if __name__ == "__main__":
    unittest.main()
