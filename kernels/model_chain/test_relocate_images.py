"""Placement and byte-preservation checks for checkpoint image relocation."""

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from relocate_images import relocate
from split_decoder_weights import verify_image


class RelocateImagesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.elf = self.root / "device.elf"
        header = bytearray(52)
        header[:16] = b"\x7fELF\x01\x01\x01" + b"\0" * 9
        struct.pack_into("<I", header, 28, 52)
        struct.pack_into("<HH", header, 42, 32, 1)
        program = struct.pack("<8I", 1, 0, 0x10000000, 0, 64, 4096, 0, 0)
        self.elf.write_bytes(header + program)
        self.manifests = []
        for index in range(2):
            data = bytes([index + 1]) * 64
            source = self.root / f"source-{index}.bin"
            source.write_bytes(data)
            base = 0x20000000 + index * 0x1000
            manifest = self.root / f"source-{index}.json"
            manifest.write_text(json.dumps({
                "image_file": source.name, "gpu_base_address": base,
                "gpu_end_address_exclusive": base + len(data),
                "image_size_bytes": len(data),
                "image_sha256": hashlib.sha256(data).hexdigest(),
                "parameters": [{"logical_name": f"value-{index}",
                                "offset_bytes": 0, "size_bytes": len(data),
                                "gpu_address": base}],
            }))
            self.manifests.append(manifest)

    def test_relocated_manifest_preserves_payload_and_updates_addresses(self):
        outputs = relocate(self.manifests, [0x1000, 0x2000], self.elf,
                           self.root / "out", 0x80000000)
        for index, output in enumerate(outputs):
            result, paths = verify_image(output)
            self.assertEqual(result["parameters"][0]["gpu_address"],
                             (index + 1) * 0x1000)
            self.assertEqual(paths[0].read_bytes(),
                             (self.root / f"source-{index}.bin").read_bytes())

    def test_rejects_overlap_and_gpu_overflow(self):
        with self.assertRaisesRegex(ValueError, "overlaps another preload"):
            relocate(self.manifests, [0x1000, 0x1000], self.elf,
                     self.root / "overlap", 0x80000000)
        with self.assertRaisesRegex(ValueError, "exceeds the GPU region"):
            relocate(self.manifests, [0x1000, 0x80000000], self.elf,
                     self.root / "overflow", 0x80000000)


if __name__ == "__main__":
    unittest.main()
