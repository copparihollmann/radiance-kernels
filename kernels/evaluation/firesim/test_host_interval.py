import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from host_interval import parse_uart


class ParseUartTest(unittest.TestCase):
    def test_pass_with_interval(self):
        uart = ("HOST_RELEASE_TO_DONE_CYCLES=1234\nSimulation complete.\n"
                "*** PASSED *** after 5678 cycles\nCOMMAND_EXIT_CODE=\"0\"\n")
        self.assertEqual(parse_uart(uart), (5678, 1234))

    def test_reject_missing_or_duplicate_interval(self):
        base = "Simulation complete.\n*** PASSED *** after 5678 cycles\nCOMMAND_EXIT_CODE=\"0\"\n"
        for prefix in ("", "HOST_RELEASE_TO_DONE_CYCLES=1234\n" * 2,
                       "HOST_RELEASE_TO_DONE_CYCLES=0\n"):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                parse_uart(prefix + base)

    def test_reject_failed_guest(self):
        uart = ("HOST_RELEASE_TO_DONE_CYCLES=1234\nSimulation complete.\n"
                "*** FAILED *** (code = 1) after 5678 cycles\n"
                "COMMAND_EXIT_CODE=\"1\"\n")
        with self.assertRaises(ValueError):
            parse_uart(uart)


if __name__ == "__main__":
    unittest.main()
