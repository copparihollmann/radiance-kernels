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
            self.assertEqual(row["elf_sha256"], AM.sha256(build / "kernel.soc.elf"))

    def test_parallel_scatter_uses_canonical_operation(self):
        self.assertEqual(AM.operation({"kind": "scatter",
                                      "collision_policy": "parallel"}),
                         "scatter-parallel")


if __name__ == "__main__":
    unittest.main()
