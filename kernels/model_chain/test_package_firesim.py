"""Small ELF controls for the checkpoint-image FireSim packer."""

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from package_firesim import package, package_hashed, read_elf, verify_package
from soc.scramble_gpu_elf import AddressHash


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

    def test_hashed_layout_contains_image_and_device_code(self):
        contents = bytearray(self.elf.read_bytes())
        struct.pack_into("<H", contents, 0x38, 2)
        struct.pack_into("<IIQQQQQQ", contents, 120, 1, 5, 8192,
                         0x100002000, 0x100002000, 64, 128, 4096)
        contents.extend(b"\0" * (8192 - len(contents)))
        code = bytes(range(64, 128))
        contents.extend(code)
        self.elf.write_bytes(contents)
        self.image.write_bytes(bytes(range(256)))
        image_manifest = json.loads(self.manifest.read_text())
        image_manifest["gpu_base_address"] = 0x1000
        image_manifest["image_size_bytes"] = self.image.stat().st_size
        image_manifest["image_sha256"] = hashlib.sha256(
            self.image.read_bytes()).hexdigest()
        self.manifest.write_text(json.dumps(image_manifest))
        result = package_hashed(self.elf, [self.manifest], self.output,
                                0x40000, 32, 4)
        self.assertEqual(result["load_method"],
                         "direct LoadMem, physically hashed GPU region")
        self.assertEqual(verify_package(
            self.output.with_suffix(".elf.json"))["hashed_images_present"], 1)
        _, _, segments = read_elf(self.output)
        loaded_gpu_bytes = sum(seg[5] for seg in segments
                               if seg[0] == 1 and seg[3] >= 0x100000000)
        self.assertLess(loaded_gpu_bytes, 0x40000)
        self.assertEqual(loaded_gpu_bytes, sum(
            x["size_bytes"] for x in result["hash_layout"]["load_regions"]))
        address_hash = AddressHash(0, 0x40000, 32, 4)
        with self.output.open("rb") as source:
            physical_base = result["hash_layout"]["file_offset"]
            for gpu_address, data in ((0x1000, self.image.read_bytes()),
                                      (0x2000, code)):
                for offset, expected in enumerate(data):
                    source.seek(physical_base +
                                address_hash.forward(gpu_address + offset))
                    self.assertEqual(source.read(1), bytes((expected,)))
            source.seek(physical_base + 0x3000)
            self.assertEqual(source.read(32), b"\0" * 32)

    def test_compact_hashed_layout_omits_unused_physical_gaps(self):
        self.image.write_bytes(bytes(range(64)))
        manifest = json.loads(self.manifest.read_text())
        manifest["gpu_base_address"] = 0x1000
        self.manifest.write_text(json.dumps(manifest))
        result = package_hashed(self.elf, [self.manifest], self.output,
                                0x40000, 32, 4, compact=True)
        self.assertLess(self.output.stat().st_size, 0x40000)
        self.assertEqual(verify_package(
            self.output.with_suffix(".elf.json"))["hashed_images_present"], 1)
        layout = result["hash_layout"]
        address_hash = AddressHash(0, 0x40000, 32, 4)
        with self.output.open("rb") as source:
            for logical, expected in enumerate(self.image.read_bytes(), 0x1000):
                physical = address_hash.forward(logical)
                region = next(item for item in layout["load_regions"]
                              if item["physical_offset"] <= physical <
                              item["physical_offset"] + item["size_bytes"])
                source.seek(region["file_offset"] + physical -
                            region["physical_offset"])
                self.assertEqual(source.read(1), bytes((expected,)))
        with self.output.open("r+b") as output:
            output.seek(layout["load_regions"][0]["file_offset"])
            output.write(b"!")
        with self.assertRaisesRegex(ValueError, "ELF hash differs"):
            verify_package(self.output.with_suffix(".elf.json"))

    def test_hashed_layout_rejects_image_past_capacity(self):
        image_manifest = json.loads(self.manifest.read_text())
        image_manifest["gpu_base_address"] = 0x4000
        self.manifest.write_text(json.dumps(image_manifest))
        with self.assertRaisesRegex(ValueError, "exceed the configured hashed GPU memory"):
            package_hashed(self.elf, [self.manifest], self.output,
                           0x4000, 32, 4)

    def test_hashed_layout_can_reuse_portable_handoff_receipt(self):
        image_manifest = json.loads(self.manifest.read_text())
        image_manifest["gpu_base_address"] = 0x1000
        self.manifest.write_text(json.dumps(image_manifest))
        package(self.elf, [self.manifest], self.output)
        receipt = self.output.with_suffix(".elf.json")
        hashed = self.root / "hashed.elf"
        package_hashed(self.elf, [], hashed, 0x4000, 32, 4,
                       receipts=[receipt])
        self.assertEqual(verify_package(hashed.with_suffix(".elf.json"))[
            "hashed_images_present"], 1)

    def test_hashed_layout_can_package_a_device_only_smoke_elf(self):
        contents = bytearray(self.elf.read_bytes())
        struct.pack_into("<H", contents, 0x38, 2)
        struct.pack_into("<IIQQQQQQ", contents, 120, 1, 5, 8192,
                         0x100002000, 0x100002000, 64, 64, 4096)
        contents.extend(b"\0" * (8192 - len(contents)))
        contents.extend(bytes(range(64)))
        self.elf.write_bytes(contents)
        result = package_hashed(self.elf, [], self.output, 0x40000, 32, 4)
        self.assertEqual(result["images"], [])
        self.assertEqual(len(result["source_gpu_segments"]), 1)
        self.assertEqual(verify_package(
            self.output.with_suffix(".elf.json"))["embedded_images_verified"], 0)


if __name__ == "__main__":
    unittest.main()
