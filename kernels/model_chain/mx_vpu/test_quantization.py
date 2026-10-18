"""Check the actual packing helpers against an independent nearest-value oracle."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class QuantizationChecks(unittest.TestCase):
    def test_rounding_boundaries_random_values_and_normalization(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            binary = str(Path(directory) / 'quantization')
            source = Path(__file__).with_suffix('.cpp')
            subprocess.run(['g++', '-std=c++17', '-O2', '-ffp-contract=off', str(source),
                            '-o', binary], check=True, capture_output=True)
            run = subprocess.run([binary], check=True, capture_output=True, text=True)
            counts = json.loads(run.stdout)
            self.assertGreater(counts['finite_fp8_checks'], 1000000)
            self.assertGreater(counts['normalization_checks'], 990000)


if __name__ == '__main__':
    unittest.main()
