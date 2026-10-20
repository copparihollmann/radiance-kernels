"""Recheck recorded run inputs against live files or exact preserved source bytes.

Historical run receipts remain unchanged. A source edit must not invalidate an
old receipt when its original bytes are preserved and still hash identically.
Executables, configurations, checkpoints and outputs have no implicit fallback.
"""
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()


class PinnedInputs:
    def __init__(self,manifest,expected_manifest_sha256):
        self.path=Path(manifest)
        if digest(self.path)!=expected_manifest_sha256:
            raise ValueError('source snapshot manifest differs from its pin')
        self.data=json.loads(self.path.read_text())

    def resolve(self,path,expected):
        original=Path(path)
        if original.is_file() and digest(original)==expected:
            return original
        saved=self.data['sources'].get(str(original),{}).get(expected)
        if not saved:
            raise ValueError(f'changed/missing input without a preserved source: {original}')
        blob=(self.path.parent/saved['path']).resolve()
        if not blob.is_relative_to(self.path.parent.resolve()) or digest(blob)!=expected:
            raise ValueError(f'preserved source differs: {original}')
        return blob

    def validate(self,expected_inputs):
        return {path:str(self.resolve(path,expected)) for path,expected in expected_inputs.items()}
