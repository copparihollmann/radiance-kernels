#!/usr/bin/env python3
"""Fuse two Spatter JSON cases into one Radiance GS case when safe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


KERNEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KERNEL))
from plan import fuse_gather_scatter  # noqa: E402
from run import normalize  # noqa: E402


def selected(path: Path, index: int) -> dict:
    suite = json.loads(path.read_text())
    if not isinstance(suite, list) or not 0 <= index < len(suite):
        raise ValueError(f"{path}: case {index} is outside the suite")
    return normalize(suite[index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gather_suite", type=Path)
    parser.add_argument("gather_case", type=int)
    parser.add_argument("scatter_suite", type=Path)
    parser.add_argument("scatter_case", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fused = fuse_gather_scatter(selected(args.gather_suite, args.gather_case),
                                selected(args.scatter_suite, args.scatter_case))
    normalize(fused)  # Check the fused address window before writing.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([fused], indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
