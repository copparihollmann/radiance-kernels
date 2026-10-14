#!/usr/bin/env python3
"""Extract selected cases from a large Spatter JSON array without loading it."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


LANL_XRAGE_SHA256 = "7325525ada0dacb6e1206717d242f6721b6d8da77718506fd909444c388f7733"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_spans(stream, chunk_bytes=1 << 20):
    """Return byte ranges for top-level objects in a JSON array."""
    spans = []
    depth = 0
    in_string = False
    escaped = False
    start = None
    offset = 0
    first_token = None
    last_token = None
    for chunk in iter(lambda: stream.read(chunk_bytes), b""):
        for byte in chunk:
            if byte > 32:
                first_token = byte if first_token is None else first_token
                last_token = byte
            if in_string:
                if escaped:
                    escaped = False
                elif byte == 92:
                    escaped = True
                elif byte == 34:
                    in_string = False
            elif byte == 34:
                in_string = True
            elif byte in (91, 123):
                if depth == 1 and byte == 123:
                    start = offset
                depth += 1
            elif byte in (93, 125):
                depth -= 1
                if depth < 0:
                    raise ValueError("unbalanced JSON brackets")
                if byte == 125 and depth == 1:
                    if start is None:
                        raise ValueError("top-level object has no start")
                    spans.append((start, offset + 1))
                    start = None
            offset += 1
    if (first_token != 91 or last_token != 93 or depth != 0 or
            in_string or start is not None):
        raise ValueError("expected one complete JSON array")
    return spans


def extract(source, output, cases, expected_sha256):
    source = Path(source)
    output = Path(output)
    actual_sha256 = sha256(source)
    if actual_sha256 != expected_sha256:
        raise ValueError("source SHA-256 differs from the pinned trace")
    with source.open("rb") as stream:
        spans = case_spans(stream)
    if not cases or len(set(cases)) != len(cases) or any(
            index < 0 or index >= len(spans) for index in cases):
        raise ValueError(f"case indices must be distinct and inside 0..{len(spans) - 1}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(b"[")
            with source.open("rb") as stream:
                for position, index in enumerate(cases):
                    if position:
                        temporary.write(b",")
                    start, stop = spans[index]
                    stream.seek(start)
                    remaining = stop - start
                    while remaining:
                        chunk = stream.read(min(remaining, 1 << 20))
                        if not chunk:
                            raise ValueError("source ended inside a case")
                        temporary.write(chunk)
                        remaining -= len(chunk)
            temporary.write(b"]\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, output)
    return {
        "source_sha256": actual_sha256,
        "source_bytes": source.stat().st_size,
        "case_count": len(spans),
        "selected_cases": cases,
        "selected_byte_ranges": [list(spans[index]) for index in cases],
        "output_sha256": sha256(output),
        "output_bytes": output.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    parser.add_argument("--source-sha256", default=LANL_XRAGE_SHA256)
    args = parser.parse_args()
    record = extract(args.source, args.out, args.case, args.source_sha256)
    record_path = args.out.with_suffix(args.out.suffix + ".json")
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
