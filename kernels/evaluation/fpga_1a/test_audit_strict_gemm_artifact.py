#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "audit_strict_gemm_artifact", HERE / "audit_strict_gemm_artifact.py")
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class StrictGemmArtifactAuditTest(unittest.TestCase):
    def test_parse_generated_macros(self):
        values = MOD.parse_macros("""
#define GEMM_M 512u
#define GEMM_B_ADDR 0x40000000u // B
#define ignored_lower 3
""")
        self.assertEqual(values, {"GEMM_M": 512, "GEMM_B_ADDR": 0x40000000})

    def test_extracts_function_without_neighbor(self):
        dump = """1000 <first>:
1000: andi a4, a3, 0x1

1010 <second>:
1010: ret
"""
        self.assertIn("andi", MOD.function_block(dump, "first"))
        self.assertNotIn("ret", MOD.function_block(dump, "first"))
        with self.assertRaisesRegex(RuntimeError, "lacks missing"):
            MOD.function_block(dump, "missing")

    def test_vector_isa_detection(self):
        self.assertFalse(MOD.isa_has_vector(
            "rv32i2p1_m2p0_zicsr2p0_zfinx1p0_zhinx1p0"))
        self.assertFalse(MOD.isa_has_vector(
            "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0"))
        self.assertTrue(MOD.isa_has_vector("rv64imafdcv_zicsr"))
        self.assertTrue(MOD.isa_has_vector("rv64imafdc_zve32x"))


if __name__ == "__main__":
    unittest.main()
