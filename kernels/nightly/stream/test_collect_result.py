"""Synthetic UART controls for the FireSim result collector."""

import unittest

from .collect_result import parse_uart


UART = """stream: kind=3 N=4102 clusters=2
cluster 0 core 0: start=100 end=450 cycles=350
cluster 0 core 1: start=101 end=449 cycles=348
cluster 1 core 0: start=102 end=451 cycles=349
cluster 1 core 1: start=103 end=450 cycles=347
kernel_cycles=351
stream: cluster=0 exact=4102/4102 nonfinite=0
stream: cluster=1 exact=4102/4102 nonfinite=0
stream: host_samples=64/64
logical_bytes=49224
bytes_per_gpu_cycle_x100=14023
PASS
"""


class CollectorTest(unittest.TestCase):
    def parse(self, text):
        return parse_uart(text, "triad", 4102, 2, 2, 49224)

    def test_valid_report(self):
        result = self.parse(UART)
        self.assertEqual(result["kernel_cycles"], 351)
        self.assertEqual(len(result["per_core"]), 4)

    def test_rejects_incomplete_or_false_result(self):
        for bad in (
            UART.replace("kernel_cycles=351", "kernel_cycles=350"),
            UART.replace("exact=4102/4102", "exact=4101/4102", 1),
            UART.replace("stream: cluster=1 exact=4102/4102 nonfinite=0\n", ""),
            UART.replace("PASS\n", "FAIL\n"),
            UART + "PASS\n",
        ):
            with self.subTest(bad=bad[-35:]):
                with self.assertRaises(ValueError):
                    self.parse(bad)


if __name__ == "__main__":
    unittest.main()
