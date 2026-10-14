"""Controls for adapting generated model chains to the HBM launch protocol."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from prepare_hbm import prepare, transform_kernel


KERNEL = '''#include "kernel_verify.h"
int main() {
  if (mu_hart_id() != 0) { for (;;) {} }
  mu_schedule(stage_0, nullptr, 1);
  if (!verify_chunk_000()) return 0;
  mu_tohost(errors ? ((errors << 1) | 1u) : 0u); return 0;
}
'''


class PrepareHbmTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_source_stages_are_copied_and_completion_is_adapted(self):
        source = self.root / "source"
        source.mkdir()
        (source / "manifest.json").write_text(json.dumps({
            "device_source_files": ["kernel.cpp", "stage_chunk_000.cpp",
                                    "verify_chunk_000.cpp"],
            "radiance_elf_sha256": "old-device-hash",
        }))
        (source / "kernel.cpp").write_text(KERNEL)
        stage = b"void stage_0() { /* original arithmetic */ }\n"
        (source / "stage_chunk_000.cpp").write_bytes(stage)
        (source / "verify_chunk_000.cpp").write_text(
            '#include "kernel_verify.h"\n'
            'bool verify_chunk_000() { mu_tohost(3); return false; }\n')
        target = prepare(source, self.root / "hbm")
        self.assertEqual((target / "stage_chunk_000.cpp").read_bytes(), stage)
        kernel = (target / "kernel.cpp").read_text()
        self.assertIn("model_hbm_wait_launch();", kernel)
        self.assertIn("model_hbm_status(errors", kernel)
        self.assertIn("model_hbm_end(); return 0;", kernel)
        self.assertNotIn("mu_tohost(", kernel)
        self.assertIn("model_hbm_status(3)",
                      (target / "verify_chunk_000.cpp").read_text())
        self.assertIn("mu_link_nightly.ld", (target / "Makefile").read_text())
        self.assertIn("mcycleh", (target / "model_hbm_bridge.h").read_text())
        self.assertIn("MODEL_HBM_PB_END_HIGH", (target / "model_hbm_pb.h").read_text())
        self.assertIn("kernel_cycles64=", (target / "host.cpp").read_text())
        manifest = json.loads((target / "manifest.json").read_text())
        self.assertEqual(manifest["source_device_elf_sha256"], "old-device-hash")
        self.assertEqual(manifest["hbm_transform"]["scheduled_calls"], 1)

    def test_rejects_an_unrecognized_completion_path(self):
        with self.assertRaisesRegex(ValueError, "completion status"):
            transform_kernel(KERNEL.replace("mu_tohost(", "other_status("))

    def test_two_cluster_launch_and_status_contract(self):
        source = self.root / "source-two"
        source.mkdir()
        (source / "manifest.json").write_text(json.dumps({
            "device_source_files": ["kernel.cpp"], "batch_clusters": 2,
        }))
        kernel = KERNEL.replace("mu_hart_id()", "vx_core_id()")
        kernel = kernel.replace("nullptr", "(void*)(uintptr_t)replica")
        kernel = kernel.replace("  if (!verify_chunk_000()) return 0;\n", "")
        (source / "kernel.cpp").write_text(kernel)
        target = prepare(source, self.root / "hbm-two")
        manifest = json.loads((target / "manifest.json").read_text())
        self.assertEqual(manifest["hbm_launch_clusters"], 2)
        self.assertIn("MODEL_HBM_CLUSTERS 2u", (target / "host.cpp").read_text())
        self.assertIn("model_hbm_end(); return 0;", (target / "kernel.cpp").read_text())


if __name__ == "__main__":
    unittest.main()
