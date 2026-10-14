"""Reject incomplete and inconsistent nightly Spatter UART reports."""

import unittest

from .collect_result import parse_uart


CASE = {"kind": "scatter", "pattern_length": 256, "count": 1024,
        "output_elements": 262144, "logical_payload_bytes": 4194304}
UART = """spatter: kind=1 length=256 count=1024 clusters=2
cluster 0 core 0: start=100 end=450 cycles=350
cluster 0 core 1: start=101 end=449 cycles=348
cluster 1 core 0: start=102 end=451 cycles=349
cluster 1 core 1: start=103 end=450 cycles=347
kernel_cycles=351
spatter: cluster=0 exact=262144/262144
spatter: cluster=1 exact=262144/262144
spatter: host_samples=64/64
logical_bytes=4194304
bytes_per_gpu_cycle_x100=1194958
PASS
"""


class CollectorTest(unittest.TestCase):
    def parse(self, text):
        return parse_uart(text, CASE, 2, 2)

    def test_valid_report(self):
        result = self.parse(UART)
        self.assertEqual(result["kernel_cycles"], 351)
        self.assertEqual(len(result["per_core"]), 4)

    def test_rejects_false_or_mismatched_report(self):
        for bad in (
            UART.replace("count=1024", "count=1025"),
            UART.replace("kernel_cycles=351", "kernel_cycles=350"),
            UART.replace("exact=262144/262144", "exact=262143/262144", 1),
            UART.replace("spatter: cluster=1 exact=262144/262144\n", ""),
            UART.replace("logical_bytes=4194304", "logical_bytes=4194303"),
            UART.replace("PASS\n", "FAIL\n"),
            UART + "PASS\n",
        ):
            with self.subTest(bad=bad[-50:]):
                with self.assertRaises(ValueError):
                    self.parse(bad)


if __name__ == "__main__":
    unittest.main()
