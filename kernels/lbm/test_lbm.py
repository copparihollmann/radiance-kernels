"""Focused lineage, numerical, ownership, and checker tests for LBM."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("radiance_lbm", ROOT / "lbm.py")
assert SPEC is not None and SPEC.loader is not None
lbm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lbm)


def words_bytes(words: list[int]) -> bytes:
    return struct.pack(f"<{len(words)}I", *words)


class LbmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blobs = lbm.verify_reference(lbm.DEFAULT_REFERENCE)
        cls.contract = lbm.official_contract(cls.blobs)

    def test_pinned_reference_and_official_default(self):
        self.assertEqual(
            lbm.git_output(lbm.DEFAULT_REFERENCE, "rev-parse", "HEAD").decode().strip(),
            lbm.REFERENCE_REVISION,
        )
        self.assertEqual(len(self.blobs), len(lbm.REFERENCE_FILES))
        self.assertEqual(
            hashlib.sha256(
                self.blobs["tests/opencl/lbm/32_32_8_ldc.of"]
            ).hexdigest(),
            "3d9454aaf7abb4d4b4d9866a1b11d0024e770ad25887b43d0bbe8d951517f2c6",
        )
        self.assertIn(
            b"OPTS ?= 1 -i $(SRC_DIR)/32_32_8_ldc.of",
            self.blobs["tests/opencl/lbm/Makefile"],
        )

    def test_ordered_model_matches_every_bundled_gold_word(self):
        destination, populations = lbm.reference_timestep(
            self.contract["source"]
        )
        velocity = lbm.velocity_field(destination)
        self.assertEqual(velocity, self.contract["gold_velocity"])
        self.assertEqual(len(velocity), 3 * 8192)
        self.assertEqual(len(populations), 19 * 8192)
        self.assertEqual(
            hashlib.sha256(words_bytes(velocity)).hexdigest(),
            "727feb99cd3db0f6951ab7a2bbd8fd850edb007837e774590448dff652923f52",
        )
        self.assertEqual(lbm.digest(velocity), 0xDAC47AA72F907F9D)

    def test_input_and_expected_population_hashes_are_stable(self):
        input_words = lbm.source_words(self.contract["source"])
        expected = self.contract["expected_populations"]
        self.assertEqual(
            hashlib.sha256(words_bytes(input_words)).hexdigest(),
            "7f0227d3967bf0c390f2131466b457453bb4eb6ce91f8680774a302b0b3bae46",
        )
        self.assertEqual(
            hashlib.sha256(words_bytes(expected)).hexdigest(),
            "b1e5e7c2dcf81b2650d37e9dd2e78948f05e58b29c6c4936c1e5c56ca926b2b1",
        )
        self.assertEqual(lbm.digest(expected), 0x5F16D36BF8D711BD)

    def test_model_destination_poisons_every_active_population(self):
        initial = self.contract["output_initial"]
        active = []
        for z in range(lbm.SIZE_Z):
            for y in range(lbm.SIZE_Y):
                for x in range(lbm.SIZE_X):
                    for entry in range(lbm.POPULATIONS):
                        active.append(lbm.float_word(
                            initial[lbm.grid_index(x, y, z, entry)]
                        ))
        self.assertEqual(len(active), 155648)
        self.assertTrue(all(word == 0x7FC00000 for word in active))

    def test_partition_is_complete_disjoint_and_five_line_safe(self):
        expected_groups = {
            1: [0, 2048],
            2: [0, 1024, 2048],
            4: [0, 512, 1024, 1536, 2048],
        }
        for sms in (1, 2, 4):
            audit = lbm.partition_audit(sms)
            self.assertEqual(audit["group_boundaries"], expected_groups[sms])
            self.assertTrue(audit["complete"])
            self.assertFalse(audit["overlap"])
            self.assertTrue(audit["cache_line_safe"])
            self.assertEqual(audit["group_cells"], 4)
            self.assertEqual(audit["group_floats_including_flags"], 80)
            self.assertEqual(audit["group_bytes"], 320)
            self.assertEqual(audit["cache_lines_per_group"], 5)
            self.assertEqual(
                sum(owner["cell_count"] for owner in audit["owners"]), 8192
            )
            self.assertEqual(
                sum(owner["population_writes"] for owner in audit["owners"]),
                19 * 8192,
            )

    def test_prepare_embeds_complete_contract_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = lbm.prepare(out, sms=4)
            persisted = json.loads((out / "result.json").read_text())
            audit = json.loads((out / "partition-audit.json").read_text())
        self.assertEqual(manifest, persisted)
        self.assertEqual(manifest["dimensions"], [32, 32, 8])
        self.assertEqual(manifest["timesteps"], 1)
        self.assertEqual(manifest["destination_populations_checked"], 155648)
        self.assertEqual(manifest["velocity_words_checked"], 24576)
        self.assertEqual(manifest["expected_population_digest"], "5f16d36bf8d711bd")
        self.assertEqual(manifest["expected_output_raw_digest"], "e1fa359562105965")
        self.assertEqual(manifest["expected_output_raw_words"], 307200)
        self.assertEqual(manifest["gold_velocity_digest"], "dac47aa72f907f9d")
        self.assertTrue(manifest["reference_verified"])
        self.assertTrue(manifest["model"]["bundled_gold_exact_word_match"])
        self.assertTrue(manifest["scalar_only"])
        self.assertTrue(manifest["no_vpu"])
        self.assertFalse(manifest["hardware_run"])
        self.assertEqual(audit, manifest["partition"])
        self.assertEqual(
            set(manifest["reference_files"]), set(lbm.REFERENCE_FILES)
        )

    def test_invalid_reference_and_sm_count_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "sms must"):
            lbm.partition_audit(3)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "reference repository"):
                lbm.verify_reference(Path(directory) / "missing")
        changed = "#define SIZE_X (33)\n"
        self.assertEqual(lbm.parse_define(changed, "SIZE_X"), 33)
        with self.assertRaisesRegex(ValueError, "missing integer definition"):
            lbm.parse_define(changed, "SIZE_Z")

    def test_kernel_is_physical_sm_scalar_and_ordered(self):
        kernel = (ROOT / "kernel.cpp").read_text()
        self.assertIn("kGroupBoundaries[physical_threadblock_id]", kernel)
        self.assertIn("LBM_GROUP_CELLS", kernel)
        self.assertIn("#pragma clang fp contract(off)", kernel)
        self.assertIn("tempNT = collide_diagonal", kernel)
        self.assertIn("nightly_kernel_end", kernel)
        for forbidden in ("gemmini", "mxgemm", "vsetvli", "__riscv_v"):
            self.assertNotIn(forbidden, kernel.lower())

    def test_host_checks_all_outputs_poison_and_guards(self):
        host = (ROOT / "host.cpp").read_text()
        self.assertIn("output[grid_index(x, y, z, entry)] = 0x7fc00000u", host)
        self.assertIn("LBM_EXPECTED_POPULATION_WORDS", host)
        self.assertIn("LBM_GOLD_VELOCITY_WORDS", host)
        self.assertIn("velocity(output", host)
        self.assertIn("difference <= absolute_tolerance", host)
        self.assertIn("difference < 0.002f * abs_float(reference)", host)
        self.assertIn("guards_intact", host)
        self.assertIn("population_digest", host)
        self.assertIn("velocity_digest", host)


if __name__ == "__main__":
    unittest.main()
