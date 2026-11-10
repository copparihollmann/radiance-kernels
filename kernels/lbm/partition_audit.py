#!/usr/bin/env python3
"""Audit the official LBM physical-SM ownership without running hardware."""

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("radiance_lbm", ROOT / "lbm.py")
assert SPEC is not None and SPEC.loader is not None
lbm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lbm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sms", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    audit = lbm.partition_audit(args.sms)
    text = json.dumps(audit, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
