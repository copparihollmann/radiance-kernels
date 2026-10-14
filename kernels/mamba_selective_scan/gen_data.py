#!/usr/bin/env python3
"""Generate a small independent FP32 scan fixture in the task temp folder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib/golden"))
import golden as G


HEADS, HEAD_DIM, GROUPS, STATE_SIZE = 4, 8, 2, 4
TOKENS, PREFILL = 7, 5


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def prepare_dt(raw_dt: np.ndarray, bias: np.ndarray,
               dt_min: float, dt_max: float) -> np.ndarray:
    z = np.asarray(raw_dt + bias, dtype=np.float32)
    return np.clip(np.logaddexp(np.float32(0), z), dt_min, dt_max).astype(np.float32)


def reference(x, b, c, dt, a, d, initial, prefill: int,
              reset_at_decode: bool = False):
    state = initial.copy()
    y = np.empty_like(x)
    for first, last in ((0, prefill), (prefill, TOKENS)):
        if first and reset_at_decode:
            state.fill(0)
        for t in range(first, last):
            for head in range(HEADS):
                group = head // (HEADS // GROUPS)
                decay = np.float32(np.exp(np.float32(dt[t, head] * a[head])))
                for dim in range(HEAD_DIM):
                    channel = head * HEAD_DIM + dim
                    value = x[t, channel]
                    total = np.float32(d[head] * value)
                    for n in range(STATE_SIZE):
                        state[channel, n] = np.float32(
                            state[channel, n] * decay
                            + np.float32(dt[t, head] * b[t, group, n]) * value)
                        total = np.float32(total + state[channel, n] * c[t, group, n])
                    y[t, channel] = total
    return y, state


def emit_fixture(out: Path, *, x, b, c, raw_dt, dt_bias, dt_min, dt_max,
                 a, d, initial,
                 gold_y, gold_state, source: str, provenance: dict) -> dict:
    dt = prepare_dt(raw_dt, dt_bias, dt_min, dt_max)
    expected_shapes = ((x, (TOKENS, HEADS * HEAD_DIM)),
                       (b, (TOKENS, GROUPS, STATE_SIZE)),
                       (c, (TOKENS, GROUPS, STATE_SIZE)),
                       (raw_dt, (TOKENS, HEADS)),
                       (dt_bias, (HEADS,)), (dt, (TOKENS, HEADS)),
                       (a, (HEADS,)),
                       (d, (HEADS,)),
                       (initial, (HEADS * HEAD_DIM, STATE_SIZE)),
                       (gold_y, (TOKENS, HEADS * HEAD_DIM)),
                       (gold_state, (HEADS * HEAD_DIM, STATE_SIZE)))
    if any(value.shape != shape or value.dtype != np.float32
           for value, shape in expected_shapes):
        raise ValueError("scan fixture has a wrong shape or dtype")
    reset_y, _ = reference(x, b, c, dt, a, d, initial, PREFILL,
                           reset_at_decode=True)
    decode_delta = float(np.max(np.abs(reset_y[PREFILL:] - gold_y[PREFILL:])))
    if decode_delta < 1e-8:
        raise ValueError("prefill-to-decode state control is too weak")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as stream:
        stream.write(f"#define DT_MIN {dt_min:.9e}f\n")
        stream.write(f"#define DT_MAX {dt_max:.9e}f\n")
        for name, value in (("HEADS", HEADS), ("HEAD_DIM", HEAD_DIM),
                            ("GROUPS", GROUPS), ("STATE_SIZE", STATE_SIZE),
                            ("TOKENS", TOKENS), ("PREFILL_TOKENS", PREFILL)):
            stream.write(f"#define {name} {value}\n")
        for name, array in (("x_raw", x), ("b_raw", b), ("c_raw", c),
                            ("dt_raw", raw_dt), ("dt_bias_raw", dt_bias),
                            ("a_raw", a), ("d_raw", d),
                            ("state_raw", initial), ("gold_y_raw", gold_y),
                            ("gold_state_raw", gold_state)):
            G.emit_f32(stream, name, array.ravel(),
                       dims=f"[{array.size}]", trailing_comma=True)
    return {
        "schema": "radiance.mamba-selective-scan-fixture.v1",
        "source": source, "provenance": provenance,
        "heads": HEADS, "head_dim": HEAD_DIM,
        "groups": GROUPS, "state_size": STATE_SIZE,
        "tokens": TOKENS, "prefill_tokens": PREFILL,
        "dt_min": dt_min, "dt_max": dt_max,
        "input_sha256": {name: sha(array) for name, array in
                         (("x", x), ("b", b), ("c", c),
                          ("raw_dt", raw_dt), ("dt_bias", dt_bias),
                          ("prepared_dt", dt),
                          ("a", a), ("d", d), ("initial_state", initial))},
        "gold_y_sha256": sha(gold_y),
        "gold_final_state_sha256": sha(gold_state),
        "reset_at_decode_max_output_delta": decode_delta,
        "generated_header_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }


def generate(out: Path) -> dict:
    rng = np.random.default_rng(19)
    x = rng.normal(0, 0.35, (TOKENS, HEADS * HEAD_DIM)).astype(np.float32)
    b = rng.normal(0, 0.2, (TOKENS, GROUPS, STATE_SIZE)).astype(np.float32)
    c = rng.normal(0, 0.2, (TOKENS, GROUPS, STATE_SIZE)).astype(np.float32)
    desired_dt = rng.uniform(0.005, 0.12, (TOKENS, HEADS)).astype(np.float32)
    dt_bias = rng.normal(0, 0.3, HEADS).astype(np.float32)
    raw_dt = (np.log(np.expm1(desired_dt)) - dt_bias).astype(np.float32)
    dt_min, dt_max = 0.0, float(np.finfo(np.float32).max)
    dt = prepare_dt(raw_dt, dt_bias, dt_min, dt_max)
    a = -rng.uniform(0.3, 1.6, HEADS).astype(np.float32)
    d = rng.uniform(0.5, 1.2, HEADS).astype(np.float32)
    initial = rng.normal(0, 0.03, (HEADS * HEAD_DIM, STATE_SIZE)).astype(np.float32)
    gold_y, gold_state = reference(x, b, c, dt, a, d, initial, PREFILL)
    return emit_fixture(out, x=x, b=b, c=c, raw_dt=raw_dt,
                        dt_bias=dt_bias, dt_min=dt_min, dt_max=dt_max,
                        a=a, d=d,
                        initial=initial, gold_y=gold_y, gold_state=gold_state,
                        source="numpy_synthetic", provenance={"seed": 19})


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
