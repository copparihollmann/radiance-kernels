#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEMV = load("gemv")
RUN = load("run")


class GemvReferenceTests(unittest.TestCase):
    def test_upstream_payload_matches_observed_device_order(self):
        # This is the complete 16-row output reconstructed from the upstream
        # Cyclotron trace.  It guards the distinction between one final FP32
        # rounding (the old, incorrect oracle) and the kernel's BF16 FMAs/tree.
        expected = [
            0x4275, 0xC1C4, 0x40B8, 0xC198, 0xC0E0, 0xC296, 0xC090, 0xC269,
            0x41EF, 0xC2C0, 0x418E, 0x416C, 0x41E8, 0xC020, 0xC2B2, 0x40F8,
        ]
        import re
        text = (ROOT / "data").read_text()
        def values(name):
            body = re.search(rf"{name}\[\]\s*=\s*\{{([^}}]*)\}}", text, re.S).group(1)
            return [int(token.strip(), 0) for token in body.split(",") if token.strip()]
        actual = GEMV.device_gemv_reference(
            values("A_raw"), values("x_raw"), 16, 4096,
        )
        self.assertEqual(actual, expected)
        self.assertEqual(RUN.digest_u32(RUN.output_words(actual)),
                         0x3F7991587C820AAB)

    def test_reference_rejects_dropped_tail(self):
        with self.assertRaises(ValueError):
            GEMV.device_gemv_reference([0] * 16, [0] * 16, 1, 16,
                                       lanes=16, ilp=1, unroll=8)


if __name__ == "__main__":
    unittest.main()
