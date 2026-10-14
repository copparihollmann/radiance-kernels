#!/usr/bin/env python3
"""Generate the four independent FP32 STREAM cases used by kernels/stream."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE.parents[1] / "stream"))
from nightly_data import Layout  # noqa: E402
from run import KINDS, OUTPUT, input_value, result_value  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=KINDS, default="copy")
    parser.add_argument("--n", type=int, default=65536)
    args = parser.parse_args()
    if args.n < 2 or args.n % 2:
        parser.error("n must be an even integer of at least 2")
    if args.n * 8 + 256 > 0x20000000:
        parser.error("two FP32 vectors do not fit in one nightly memory region")

    layout = Layout("stream")
    layout.const("STREAM_N", f"{args.n}u")
    layout.const("STREAM_KIND", KINDS[args.kind])
    layout.const("STREAM_LOGICAL_BYTES", f"{args.n * (8 if args.kind in ('copy', 'scale') else 12)}ull")
    output = OUTPUT[args.kind]
    needed = {"copy": "a", "scale": "c", "add": "ab", "triad": "bc"}[args.kind]
    input_hashes = {}
    for name, region in (("a", 1), ("b", 2), ("c", 3)):
        if name == output:
            words = np.full(args.n, 0x7FC00000, dtype="<u4")
        elif name in needed:
            words = np.fromiter((input_value(name, i) for i in range(args.n)),
                                dtype="<f4", count=args.n)
        else:
            words = np.zeros(1, dtype="<f4")
        layout.add(f"STREAM_{name.upper()}", region, words,
                   "FP32 output poison" if name == output else
                   "unused sentinel" if name not in needed else "FP32 input")
        input_hashes[name] = hashlib.sha256(words.tobytes()).hexdigest()
    golden = np.fromiter((result_value(args.kind, i) for i in range(args.n)),
                         dtype="<f4", count=args.n)
    layout.add("STREAM_G", 1, golden, "complete upstream STREAM reference")
    layout.const("STREAM_OUT_ADDR", f"STREAM_{output.upper()}_ADDR")
    layout.write("data", "stream_data.h")
    receipt = {
        "operation": args.kind,
        "elements": args.n,
        "dtype": "fp32",
        "logical_payload_bytes": args.n * (8 if args.kind in ("copy", "scale") else 12),
        "reference_source": "kernels/stream/run.py",
        "reference_source_sha256": hashlib.sha256(
            (HERE.parents[1] / "stream/run.py").read_bytes()).hexdigest(),
        "initial_array_sha256": input_hashes,
        "golden_sha256": hashlib.sha256(golden.tobytes()).hexdigest(),
    }
    Path("data/stream.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
