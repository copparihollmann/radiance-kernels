import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from pinned_inputs import PinnedInputs


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


class PreservedSourceInputs(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.live=self.root/'source.h';self.live.write_text('original')
        self.blob=self.root/'blob';self.blob.write_bytes(self.live.read_bytes())
        self.expected=sha(self.live)
        self.manifest=self.root/'manifest.json'
        self.manifest.write_text(json.dumps(dict(sources={str(self.live):{self.expected:dict(path='blob')}})))
        self.pin=sha(self.manifest)

    def test_original_live_input(self):
        self.assertEqual(PinnedInputs(self.manifest,self.pin).resolve(self.live,self.expected),self.live)

    def test_edited_source_keeps_original_receipt(self):
        self.live.write_text('upstream update')
        checked=PinnedInputs(self.manifest,self.pin).validate({str(self.live):self.expected})
        self.assertEqual(checked[str(self.live)],str(self.blob))
        self.assertEqual(self.live.read_text(),'upstream update')

    def test_missing_source_uses_preserved_bytes(self):
        self.live.unlink()
        self.assertEqual(PinnedInputs(self.manifest,self.pin).resolve(self.live,self.expected),self.blob)

    def test_reject_modified_manifest_or_preserved_source(self):
        self.live.write_text('changed')
        self.blob.write_text('modified snapshot')
        with self.assertRaises(ValueError): PinnedInputs(self.manifest,self.pin).resolve(self.live,self.expected)
        self.manifest.write_text('{}')
        with self.assertRaises(ValueError): PinnedInputs(self.manifest,self.pin)

    def test_reject_missing_executable_without_snapshot(self):
        with self.assertRaises(ValueError): PinnedInputs(self.manifest,self.pin).resolve(self.root/'engine',self.expected)


if __name__=='__main__': unittest.main()
