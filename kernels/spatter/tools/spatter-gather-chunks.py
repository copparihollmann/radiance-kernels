#!/usr/bin/env python3
"""Plan contiguous count chunks for a wrap-1 Spatter Gather."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import fnv, normalize, payload, reference  # noqa: E402


def plan(suite: Path, case_id: int, chunk_count: int) -> dict:
    data = suite.read_bytes()
    cases = json.loads(data)
    if not isinstance(cases, list) or not 0 <= case_id < len(cases):
        raise ValueError("case must select a configuration in the JSON list")
    raw = cases[case_id]
    original_count = raw.get("count", 1024)
    if not isinstance(original_count, int) or original_count <= 0:
        raise ValueError("original count must be a positive integer")
    if not 0 < chunk_count < original_count:
        raise ValueError("chunk count must be positive and smaller than the original count")
    probe = normalize({**raw, "count": 1})
    if probe["kind"] != "gather" or probe["wrap"] != 1:
        raise ValueError("only wrap-1 Gather is currently supported")
    last_iteration = original_count - 1
    full_digest = fnv(payload(probe["payload_tag"], index + probe["delta"] * last_iteration)
                      for index in probe["pattern"])
    chunks = []
    for start in range(0, original_count, chunk_count):
        count = min(chunk_count, original_count - start)
        case = normalize({**raw, "count": count})
        case["iteration_start"] = start
        case["source_index_base"] = start * case["delta"]
        digest, overlap = reference(case)
        assert digest is not None and not overlap
        chunks.append({
            "iteration_start": start,
            "count": count,
            "iteration_end_exclusive": start + count,
            "source_index_base": case["source_index_base"],
            "source_elements": case["src_length"],
            "source_bytes": 8 * case["src_length"],
            "logical_payload_bytes": 16 * count * case["length"],
            "expected_digest": f"{digest:016x}",
        })
    if chunks[-1]["expected_digest"] != f"{full_digest:016x}":
        raise AssertionError("the final chunk differs from the full serial result")
    if sum(chunk["count"] for chunk in chunks) != original_count:
        raise AssertionError("count chunks do not cover the full input")
    original_source = max(probe["pattern"]) + probe["delta"] * last_iteration + 1
    return {
        "suite": str(suite.relative_to(ROOT) if suite.is_relative_to(ROOT) else suite),
        "suite_sha256": hashlib.sha256(data).hexdigest(),
        "case": case_id,
        "kind": "gather",
        "wrap": 1,
        "pattern_length": probe["length"],
        "delta": probe["delta"],
        "original_count": original_count,
        "original_source_elements": original_source,
        "original_source_bytes": 8 * original_source,
        "logical_payload_bytes": 16 * original_count * probe["length"],
        "expected_full_digest": f"{full_digest:016x}",
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--chunk-count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = plan(args.suite.resolve(), args.case, args.chunk_count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(result['chunks'])} chunks; complete expected digest "
          f"{result['expected_full_digest']}")


if __name__ == "__main__":
    main()
