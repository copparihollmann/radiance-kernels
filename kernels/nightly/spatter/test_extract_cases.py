"""Bounded-memory extraction of original-size Spatter cases."""

import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from kernels.nightly.spatter.extract_cases import case_spans, extract, sha256


class ExtractCasesTest(unittest.TestCase):
    def test_nested_arrays_quotes_and_chunk_boundaries(self):
        raw = b'[{"kernel":"Gather","pattern":[1,2],"label":"a\\\"},["},{"kernel":"Scatter","pattern":[3]}]'
        spans = case_spans(BytesIO(raw), chunk_bytes=3)
        self.assertEqual(len(spans), 2)
        self.assertEqual(json.loads(raw[spans[0][0]:spans[0][1]]),
                         {"kernel": "Gather", "pattern": [1, 2],
                          "label": 'a"},['})
        self.assertEqual(json.loads(raw[spans[1][0]:spans[1][1]])["kernel"],
                         "Scatter")

    def test_extract_checks_source_and_preserves_selected_objects(self):
        raw = b'[{"case":0},{"case":1,"pattern":[2,3]},{"case":2}]'
        temporary_root = Path(__file__).resolve().parents[3] / ".tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            source = Path(directory) / "source.json"
            output = Path(directory) / "selected.json"
            source.write_bytes(raw)
            expected = hashlib.sha256(raw).hexdigest()
            with self.assertRaisesRegex(ValueError, "source SHA-256"):
                extract(source, output, [2, 0], "0" * 64)
            self.assertFalse(output.exists())
            record = extract(source, output, [2, 0], expected)
            self.assertEqual(json.loads(output.read_text()),
                             [{"case": 2}, {"case": 0}])
            self.assertEqual(record["output_sha256"], sha256(output))


if __name__ == "__main__":
    unittest.main()
