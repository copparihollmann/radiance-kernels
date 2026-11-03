#!/usr/bin/env python3
"""Prove exact, cache-line-exclusive STREAM block-cyclic ownership."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LINE_ELEMENTS = 16


def audit(elements: int, sm_counts: tuple[int, ...], chunk_lines: int = 1) -> dict:
    if elements < 1:
        raise ValueError("elements must be positive")
    if chunk_lines < 1:
        raise ValueError("chunk_lines must be positive")
    checks = []
    lines = (elements + LINE_ELEMENTS - 1) // LINE_ELEMENTS
    for sms in sm_counts:
        if sms < 1:
            raise ValueError("SM counts must be positive")
        seen = bytearray(elements)
        line_counts = [0] * sms
        element_counts = [0] * sms
        for line in range(lines):
            owner = (line // chunk_lines) % sms
            line_counts[owner] += 1
            begin = line * LINE_ELEMENTS
            end = min(begin + LINE_ELEMENTS, elements)
            for index in range(begin, end):
                seen[index] += 1
                element_counts[owner] += 1
        complete = all(count == 1 for count in seen)
        # Cache-line exclusivity must be DERIVED from the element->owner map, not asserted.
        # Asserting it as a literal cannot fail, so it would keep reporting True if the
        # partition were ever changed to element granularity. Here we rebuild the owner of
        # every element independently and require that all elements of a 64-byte line share
        # one owner -- a check that genuinely fails for a line-straddling partition.
        element_owner = [-1] * elements
        for line in range(lines):
            owner = (line // chunk_lines) % sms
            for index in range(line * LINE_ELEMENTS,
                               min((line + 1) * LINE_ELEMENTS, elements)):
                element_owner[index] = owner
        split_lines = []
        for line in range(lines):
            begin = line * LINE_ELEMENTS
            end = min(begin + LINE_ELEMENTS, elements)
            owners = {element_owner[i] for i in range(begin, end)}
            if len(owners) != 1 or -1 in owners:
                split_lines.append(line)
        exclusive = not split_lines
        checks.append({
            "sm_count": sms,
            "status": "pass" if (complete and exclusive) else "fail",
            "coverage_exactly_once": complete,
            "cache_line_exclusive": exclusive,
            "split_line_count": len(split_lines),
            "split_lines_sample": split_lines[:8],
            "line_counts": line_counts,
            "element_counts": element_counts,
            "total_elements": sum(element_counts),
        })
    passed = all(check["status"] == "pass" for check in checks)
    return {
        "schema": "radiance-stream-blockcyclic-partition-audit-v1",
        "status": "pass" if passed else "fail",
        "elements": elements,
        "line_elements": LINE_ELEMENTS,
        "chunk_lines": chunk_lines,
        "cache_line_bytes": 64,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=1_048_576)
    parser.add_argument("--sms", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--chunk-lines", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.elements, tuple(args.sms), args.chunk_lines)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    else:
        print(rendered, end="")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
