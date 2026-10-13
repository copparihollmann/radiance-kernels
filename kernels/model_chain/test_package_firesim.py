"""Small ELF controls for the checkpoint-image FireSim packer."""

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from package_firesim import package, verify_package


class PackageFireSimTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.elf = self.root / "source.elf"
        header = bytearray(64)
        header[:16] = b"\x7fELF\x02\x01\x01" + b"\0" * 9
        struct.pack_into("<HHIQQQIHHHHHH", header, 16, 2, 243, 1,
                         0x80000000, 64, 0, 0, 64, 56, 1, 0, 0, 0)
        phdr = struct.pack("<IIQQQQQQ", 1, 5, 4096, 0x80000000,
                           0x80000000, 64, 64, 4096)
        with self.elf.open("wb") as output:
            output.write(header)
            output.write(phdr)
            output.write(b"\0" * (4096 - output.tell()))
            output.write(b"\x13\0\0\0" * 16)
        self.image = self.root / "image.bin"
        self.image.write_bytes(bytes(range(64)))
        self.manifest = self.root / "image.json"
        self.manifest.write_text(json.dumps({
            "image_file": self.image.name,
            "gpu_base_address": 0x20000000,
            "image_size_bytes": 64,
            "image_sha256": hashlib.sha256(self.image.read_bytes()).hexdigest(),
        }))
        self.output = self.root / "firesim.elf"

    def test_embedded_image_and_corruption_detection(self):
        package(self.elf, [self.manifest], self.output)
        receipt = self.output.with_suffix(".elf.json")
        self.assertEqual(verify_package(receipt)["embedded_images_verified"], 1)
        metadata = json.loads(receipt.read_text())
        image_offset = metadata["images"][0]["file_offset"]
        with self.output.open("r+b") as stream:
            stream.seek(image_offset)
            stream.write(b"!")
        with self.assertRaisesRegex(ValueError, "ELF hash differs"):
            verify_package(receipt)

    def test_rejects_overlapping_image(self):
        with self.assertRaisesRegex(ValueError, "overlapping load regions"):
            package(self.elf, [self.manifest, self.manifest], self.output)

    def test_rejects_bad_image_hash(self):
        self.image.write_bytes(b"!" * 64)
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            package(self.elf, [self.manifest], self.output)


if __name__ == "__main__":
    unittest.main()
