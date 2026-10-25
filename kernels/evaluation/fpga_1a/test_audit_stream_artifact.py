#!/usr/bin/env python3
"""Focused tests for the STREAM scalar/no-VPU artifact audit."""

import unittest

from kernels.evaluation.fpga_1a.audit_stream_artifact import (
    has_vector_isa,
    parse_architecture,
)


class StreamArtifactAuditTest(unittest.TestCase):
    def test_parse_gnu_readelf_architecture(self) -> None:
        self.assertEqual(
            parse_architecture(
                '  Tag_RISCV_arch: "rv32i2p1_m2p0_zicsr2p0_zfinx1p0"\n'
            ),
            "rv32i2p1_m2p0_zicsr2p0_zfinx1p0",
        )

    def test_parse_llvm_readelf_architecture(self) -> None:
        self.assertEqual(
            parse_architecture(
                "      Tag: 5\n"
                "      TagName: arch\n"
                "      Value: rv32i2p1_m2p0_zicsr2p0_zfinx1p0\n"
            ),
            "rv32i2p1_m2p0_zicsr2p0_zfinx1p0",
        )

    def test_scalar_isa_is_not_vector(self) -> None:
        self.assertFalse(
            has_vector_isa("rv32i2p1_m2p0_zicsr2p0_zfinx1p0_zhinx1p0")
        )

    def test_vector_extensions_are_rejected(self) -> None:
        for architecture in (
            "rv64imafdcv",
            "rv32i2p1_v1p0",
            "rv32i2p1_zve32x1p0",
        ):
            with self.subTest(architecture=architecture):
                self.assertTrue(has_vector_isa(architecture))


if __name__ == "__main__":
    unittest.main()
