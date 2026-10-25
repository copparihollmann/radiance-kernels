#!/usr/bin/env python3
"""Audit a cache-line-aligned block-cyclic owner map for full xRAGE Gather."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SPATTER = REPO / "kernels" / "spatter"
sys.path.insert(0, str(SPATTER))

from run import normalize  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--block-elements", type=int, default=16)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.block_elements < 8 or args.block_elements % 8 != 0:
        parser.error("--block-elements must be a positive multiple of 8")

    raw = json.loads(args.suite.read_text())
    case = normalize(raw[args.case], args.suite.parent)
    assert case["kind"] in ("gather", "multigather")
    assert case["count"] <= case["wrap"], (
        "candidate is only for the direct output-owner path")

    owners = case["length"] * case["wrap"]
    block_elements = args.block_elements
    chunks = (owners + block_elements - 1) // block_elements
    checks = []
    for sms in (1, 2, 4):
        owners_per_sm = [0] * sms
        chunks_per_sm = [0] * sms
        for chunk in range(chunks):
            sm = chunk % sms
            begin = chunk * block_elements
            end = min(owners, begin + block_elements)
            assert begin % 8 == 0
            assert end == owners or end % 8 == 0
            chunks_per_sm[sm] += 1
            owners_per_sm[sm] += end - begin
        assert sum(owners_per_sm) == owners
        assert max(owners_per_sm) - min(owners_per_sm) <= block_elements
        checks.append({
            "sm_count": sms,
            "owners": owners,
            "total_transfers": case["count"] * case["length"],
            "block_elements": block_elements,
            "output_bytes_per_block": block_elements * 8,
            "chunks_per_sm": chunks_per_sm,
            "owners_per_sm": owners_per_sm,
            "max_owner_imbalance": max(owners_per_sm) - min(owners_per_sm),
            "cache_line_exclusive": True,
            "coverage_exactly_once": True,
            "status": "pass",
        })

    pattern = raw[args.case]["pattern"]
    pattern_path = Path(pattern["file"])
    if not pattern_path.is_absolute():
        pattern_path = (args.suite.parent / pattern_path).resolve()
    report = {
        "status": "pass",
        "suite": str(args.suite.resolve()),
        "suite_sha256": sha256(args.suite),
        "case": args.case,
        "kind": case["kind"],
        "pattern_path": str(pattern_path),
        "pattern_sha256": sha256(pattern_path),
        "kernel_sha256": sha256(SPATTER / "kernel.cpp"),
        "checks": checks,
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
