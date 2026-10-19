"""Metadata changes are allowed; changed instructions or loaded state are not."""
from pathlib import Path
import struct
import tempfile
import unittest

from debug_symbols import load_signature, verify_pair


class DebugSymbols(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.original = self.root / 'qualified.elf'
        self.debug = self.root / 'debug.elf'
        data = bytearray(132)
        data[:6] = b'\x7fELF\x01\x01'
        struct.pack_into('<II', data, 24, 0x10000000, 52)
        struct.pack_into('<HH', data, 42, 32, 1)
        struct.pack_into('<8I', data, 52, 1, 128, 0x10000000, 0x10000000, 4, 64, 5, 4096)
        data[128:] = b'code'
        self.data = data
        self.original.write_bytes(data)

    def test_added_metadata_and_different_file_offsets_preserve_loaded_memory(self):
        data = bytearray(self.data[:128]) + b'padding!' + self.data[128:] + b'debug line tables'
        struct.pack_into('<I', data, 56, 136)
        self.debug.write_bytes(data)
        self.assertTrue(verify_pair(self.original, self.debug)['passed'])

    def test_changed_instructions_bss_flags_and_entry_are_rejected(self):
        for offset in (128, 24, 52 + 20, 52 + 24):
            with self.subTest(offset=offset):
                data = bytearray(self.data)
                data[offset] ^= 1
                self.debug.write_bytes(data)
                self.assertFalse(verify_pair(self.original, self.debug)['passed'])

    def test_truncated_loaded_segment_is_rejected(self):
        self.debug.write_bytes(self.data[:-1])
        with self.assertRaises(ValueError):
            load_signature(self.debug)


if __name__ == '__main__':
    unittest.main()
