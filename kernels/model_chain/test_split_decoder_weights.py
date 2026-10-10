"""Check that segmented checkpoint images preserve every parameter byte."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from split_decoder_weights import verify_image, write_split


class SplitDecoderWeightsTest(unittest.TestCase):
    def test_split_preserves_parameters_and_detects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            tensors = [bytes([value]) * 64 for value in (1, 2, 3)]
            blob = b"".join(tensors)
            (source / "weights.bin").write_bytes(blob)
            manifest = {
                "image_file": "weights.bin",
                "image_size_bytes": len(blob),
                "image_sha256": hashlib.sha256(blob).hexdigest(),
                "gpu_base_address": 0x30000000,
                "parameters": [
                    {"logical_name": f"parameter.{index}",
                     "offset_bytes": 64 * index,
                     "gpu_address": 0x30000000 + 64 * index,
                     "size_bytes": 64,
                     "packed_sha256": hashlib.sha256(tensor).hexdigest()}
                    for index, tensor in enumerate(tensors)
                ],
            }
            source_manifest = source / "weights-image.json"
            source_manifest.write_text(json.dumps(manifest))
            destination = Path(temporary) / "split"
            result = write_split(source_manifest, destination,
                                 first_base=0x30000000,
                                 second_base=0x80000000,
                                 first_limit=0x30000080)
            self.assertEqual((destination / "weights-0.bin").read_bytes(),
                             tensors[0] + tensors[1])
            self.assertEqual((destination / "weights-1.bin").read_bytes(),
                             tensors[2])
            self.assertEqual(result["image_sha256"], manifest["image_sha256"])
            self.assertEqual([item["gpu_address"] for item in result["parameters"]],
                             [0x30000000, 0x30000040, 0x80000000])
            verify_image(destination / "weights-image.json")
            (destination / "weights-1.bin").write_bytes(bytes(64))
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                verify_image(destination / "weights-image.json")


if __name__ == "__main__":
    unittest.main()
