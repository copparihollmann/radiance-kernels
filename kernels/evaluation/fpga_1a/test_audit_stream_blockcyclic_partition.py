#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import unittest


MODULE = Path(__file__).with_name("audit_stream_blockcyclic_partition.py")
SPEC = importlib.util.spec_from_file_location("stream_partition_audit", MODULE)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class StreamBlockCyclicPartitionAuditTests(unittest.TestCase):
    def test_exact_ownership_for_full_and_partial_lines(self):
        result = AUDIT.audit(67, (1, 2, 4))
        self.assertEqual(result["status"], "pass")
        for check in result["checks"]:
            self.assertTrue(check["coverage_exactly_once"])
            self.assertTrue(check["cache_line_exclusive"])
            self.assertEqual(check["total_elements"], 67)

    def test_rejects_invalid_sizes(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            AUDIT.audit(0, (1,))
        with self.assertRaisesRegex(ValueError, "positive"):
            AUDIT.audit(16, (0,))
        with self.assertRaisesRegex(ValueError, "chunk_lines"):
            AUDIT.audit(16, (1,), 0)

    def test_exact_ownership_for_multi_line_chunks(self):
        result = AUDIT.audit(1049, (1, 2, 4), chunk_lines=8)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["chunk_lines"], 8)
        for check in result["checks"]:
            self.assertTrue(check["coverage_exactly_once"])
            self.assertTrue(check["cache_line_exclusive"])
            self.assertEqual(check["total_elements"], 1049)


if __name__ == "__main__":
    unittest.main()
