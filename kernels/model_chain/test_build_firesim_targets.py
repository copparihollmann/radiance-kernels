"""Small disk-capacity controls for target-specific FireSim packaging."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from build_firesim_targets import DISK_RESERVE_BYTES, preflight_disk_space


class DiskPreflightTest(unittest.TestCase):
    def test_accounts_for_all_outputs_and_preserves_reserve(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as name:
            bundle = Path(name)
            (bundle / "one.elf").write_bytes(b"a" * 17)
            (bundle / "two.elf").write_bytes(b"b" * 29)
            models = [{"source_elf": "one.elf"}, {"source_elf": "two.elf"}]
            required = 17 + 29 + 2 * (64 + (1 << 20)) + DISK_RESERVE_BYTES
            with patch("build_firesim_targets.shutil.disk_usage",
                       return_value=SimpleNamespace(free=required - 1)):
                with self.assertRaisesRegex(OSError, "insufficient free space"):
                    preflight_disk_space(bundle, bundle / "new" / "out", models, 64)
            with patch("build_firesim_targets.shutil.disk_usage",
                       return_value=SimpleNamespace(free=required)):
                result = preflight_disk_space(
                    bundle, bundle / "new" / "out", models, 64)
            self.assertEqual(result["conservative_required_bytes_with_reserve"], required)


if __name__ == "__main__":
    unittest.main()
