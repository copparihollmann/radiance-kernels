#!/usr/bin/env python3
"""Generate a small independent FP32 causal depthwise-convolution fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib/golden"))
import golden as G

CHANNELS, KERNEL_WIDTH = 48, 4
TOKENS, PREFILL = 7, 5


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def reference(x, weight, bias, initial, reset_at_decode=False):
    state = initial.copy()
    y = np.empty_like(x)
    for first, last in ((0, PREFILL), (PREFILL, TOKENS)):
        if first and reset_at_decode:
            state.fill(0)
        for t in range(first, last):
            for channel in range(CHANNELS):
                state[channel, :-1] = state[channel, 1:]
                state[channel, -1] = x[t, channel]
                value = np.float32(bias[channel])
                for k in range(KERNEL_WIDTH):
                    value = np.float32(value + state[channel, k] * weight[channel, k])
                y[t, channel] = np.float32(value / (1.0 + np.exp(-value)))
    return y, state


def emit_fixture(out: Path, *, x, weight, bias, initial,
                 gold_y, gold_state, source: str, provenance: dict) -> dict:
    expected_shapes = ((x, (TOKENS, CHANNELS)),
                       (weight, (CHANNELS, KERNEL_WIDTH)),
                       (bias, (CHANNELS,)),
                       (initial, (CHANNELS, KERNEL_WIDTH)),
                       (gold_y, (TOKENS, CHANNELS)),
                       (gold_state, (CHANNELS, KERNEL_WIDTH)))
    if any(value.shape != shape or value.dtype != np.float32
           for value, shape in expected_shapes):
        raise ValueError("convolution fixture has a wrong shape or dtype")
    reset_y, _ = reference(x, weight, bias, initial, reset_at_decode=True)
    decode_delta = float(np.max(np.abs(reset_y[PREFILL:] - gold_y[PREFILL:])))
    if decode_delta < 1e-8:
        raise ValueError("prefill-to-decode convolution state control is too weak")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as stream:
        for name, value in (("CHANNELS", CHANNELS),
                            ("KERNEL_WIDTH", KERNEL_WIDTH),
                            ("TOKENS", TOKENS),
                            ("PREFILL_TOKENS", PREFILL)):
            stream.write(f"#define {name} {value}\n")
        for name, array in (("x_raw", x), ("weight_raw", weight),
                            ("bias_raw", bias), ("state_raw", initial),
                            ("gold_y_raw", gold_y),
                            ("gold_state_raw", gold_state)):
            G.emit_f32(stream, name, array.ravel(),
                       dims=f"[{array.size}]", trailing_comma=True)
    return {
        "schema": "radiance.mamba-depthwise-conv-fixture.v1",
        "source": source, "provenance": provenance,
        "channels": CHANNELS, "kernel_width": KERNEL_WIDTH,
        "tokens": TOKENS, "prefill_tokens": PREFILL,
        "input_sha256": {name: sha(array) for name, array in
                         (("x", x), ("weight", weight),
                          ("bias", bias), ("initial_state", initial))},
        "gold_y_sha256": sha(gold_y),
        "gold_final_state_sha256": sha(gold_state),
        "reset_at_decode_max_output_delta": decode_delta,
        "generated_header_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }


def generate(out: Path) -> dict:
    rng = np.random.default_rng(23)
    x = rng.normal(0, 0.12, (TOKENS, CHANNELS)).astype(np.float32)
    weight = rng.normal(0, 0.25, (CHANNELS, KERNEL_WIDTH)).astype(np.float32)
    bias = rng.normal(0, 0.03, CHANNELS).astype(np.float32)
    initial = rng.normal(0, 0.05, (CHANNELS, KERNEL_WIDTH)).astype(np.float32)
    gold_y, gold_state = reference(x, weight, bias, initial)
    return emit_fixture(out, x=x, weight=weight, bias=bias, initial=initial,
                        gold_y=gold_y, gold_state=gold_state,
                        source="numpy_synthetic", provenance={"seed": 23})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = generate(args.out)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
