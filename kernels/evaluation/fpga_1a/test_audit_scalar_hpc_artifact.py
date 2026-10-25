#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest


MODULE = Path(__file__).with_name("audit_scalar_hpc_artifact.py")
SPEC = importlib.util.spec_from_file_location("audit_scalar_hpc_artifact", MODULE)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


class ScalarHpcArtifactAuditTest(unittest.TestCase):
    def test_expected_matrix_is_exactly_logical_and_direct_1_2_4_sm(self):
        expected = {(sm, loader) for sm in (1, 2, 4)
                    for loader in ("logical", "direct-hashed")}
        for family in ("hotspot3d", "pathfinder", "gaussian"):
            self.assertEqual(AUDIT.expected_identities(family), expected)

    def test_unknown_family_fails_closed(self):
        with self.assertRaisesRegex(SystemExit, "unsupported family"):
            AUDIT.expected_identities("unknown")


if __name__ == "__main__":
    unittest.main()
