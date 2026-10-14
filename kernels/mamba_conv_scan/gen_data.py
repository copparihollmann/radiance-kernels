#!/usr/bin/env python3
"""Join independently checked upstream conv and scan fixtures for one ELF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib/golden"))
import golden as G


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_array(header: str, name: str, size: int) -> np.ndarray:
    match = re.search(rf"__global float {name}\[{size}\] = \{{(.*?)\}};",
                      header, re.S)
    if match is None:
        raise ValueError(f"missing {name}[{size}] in source fixture")
    values = [float(token.rstrip("f")) for token in
              re.findall(r"[-+]?\d+\.\d+e[-+]?\d+f", match.group(1))]
    if len(values) != size:
        raise ValueError(f"wrong length for {name}: {len(values)}")
    return np.asarray(values, dtype="<f4")


def generate(out: Path, conv_dir: Path, scan_dir: Path) -> dict:
    conv = json.loads((conv_dir / "fixture.json").read_text())
    scan = json.loads((scan_dir / "fixture.json").read_text())
    conv_result = json.loads((conv_dir / "result.json").read_text())
    scan_result = json.loads((scan_dir / "result.json").read_text())
    cp, sp = conv["provenance"], scan["provenance"]
    for field in ("modeling_source_sha256", "upstream_input_sha256", "seed"):
        if cp[field] != sp[field]:
            raise ValueError(f"upstream conv and scan differ in {field}")
    if cp["upstream_mixer_output_sha256"] != sp["upstream_output_sha256"]:
        raise ValueError("upstream full-mixer outputs differ")
    if (conv["source"] != "transformers_reduced"
            or scan["source"] != "transformers_reduced_device_conv"
            or conv_result["status"] != "passed"
            or scan_result["status"] != "passed"
            or conv["channels"] != 48 or conv["kernel_width"] != 4
            or scan["heads"] != 4 or scan["head_dim"] != 8
            or scan["groups"] != 2 or scan["state_size"] != 4
            or conv["tokens"] != scan["tokens"] or conv["prefill_tokens"] != scan["prefill_tokens"]):
        raise ValueError("source fixtures are incompatible or unverified")
    if (digest((conv_dir / "data").read_bytes()) != conv["generated_header_sha256"]
            or digest((scan_dir / "data").read_bytes()) != scan["generated_header_sha256"]
            or digest((conv_dir / "output.bin").read_bytes()) != conv_result["output_sha256"]
            or sp["conv_handoff"]["device_output_sha256"] != conv_result["output_sha256"]):
        raise ValueError("source fixture hash mismatch")
    ch = (conv_dir / "data").read_text()
    sh = (scan_dir / "data").read_text()
    c = lambda name, size: load_array(ch, name, size)
    s = lambda name, size: load_array(sh, name, size)
    arrays = {
        "conv_x_raw": c("x_raw", 7 * 48),
        "conv_weight_raw": c("weight_raw", 48 * 4),
        "conv_bias_raw": c("bias_raw", 48),
        "conv_state_raw": c("state_raw", 48 * 4),
        "gold_conv_y_raw": c("gold_y_raw", 7 * 48),
        "gold_conv_state_raw": c("gold_state_raw", 48 * 4),
        "dt_raw": s("dt_raw", 7 * 4),
        "dt_bias_raw": s("dt_bias_raw", 4),
        "a_raw": s("a_raw", 4),
        "d_raw": s("d_raw", 4),
        "scan_state_raw": s("state_raw", 32 * 4),
        "gold_scan_y_raw": s("gold_y_raw", 7 * 32),
        "gold_scan_state_raw": s("gold_state_raw", 32 * 4),
    }
    input_fields = (("conv_x_raw", "x"), ("conv_weight_raw", "weight"),
                    ("conv_bias_raw", "bias"), ("conv_state_raw", "initial_state"))
    for source_name, key in input_fields:
        if digest(arrays[source_name].tobytes()) != conv["input_sha256"][key]:
            raise ValueError(f"convolution {key} hash mismatch")
    for name, field in (("gold_conv_y_raw", "gold_y_sha256"),
                        ("gold_conv_state_raw", "gold_final_state_sha256")):
        if digest(arrays[name].tobytes()) != conv[field]:
            raise ValueError(f"convolution {name} hash mismatch")
    for name, field in (("gold_scan_y_raw", "gold_y_sha256"),
                        ("gold_scan_state_raw", "gold_final_state_sha256")):
        if digest(arrays[name].tobytes()) != scan[field]:
            raise ValueError(f"scan {name} hash mismatch")
    actual_conv = np.frombuffer((conv_dir / "output.bin").read_bytes(),
                                dtype="<f4").reshape(7, 48)
    if not np.array_equal(actual_conv[:, :32],
                          s("x_raw", 7 * 32).reshape(7, 32)):
        raise ValueError("scan X does not match the checked device conv output")
    for name, start in (("b_raw", 32), ("c_raw", 40)):
        if not np.array_equal(actual_conv[:, start:start + 8],
                              s(name, 7 * 8).reshape(7, 8)):
            raise ValueError(f"scan {name} does not match device conv output")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as stream:
        stream.write(f"#define DT_MIN {scan['dt_min']:.9e}f\n")
        stream.write(f"#define DT_MAX {scan['dt_max']:.9e}f\n")
        for name, value in (("HEADS", 4), ("HEAD_DIM", 8), ("GROUPS", 2),
                            ("STATE_SIZE", 4), ("CHANNELS", 48),
                            ("KERNEL_WIDTH", 4), ("TOKENS", 7),
                            ("PREFILL_TOKENS", 5)):
            stream.write(f"#define {name} {value}\n")
        for name, array in arrays.items():
            G.emit_f32(stream, name, array, dims=f"[{array.size}]",
                       trailing_comma=True)
    return {
        "schema": "radiance.mamba-conv-scan-fixture.v1",
        "source": "transformers_reduced",
        "scope": "random-weight reduced mixer; conv and scan only",
        "upstream_input_sha256": cp["upstream_input_sha256"],
        "upstream_modeling_source_sha256": cp["modeling_source_sha256"],
        "source_conv_fixture_sha256": digest((conv_dir / "fixture.json").read_bytes()),
        "source_scan_fixture_sha256": digest((scan_dir / "fixture.json").read_bytes()),
        "source_device_conv_output_sha256": conv_result["output_sha256"],
        "array_sha256": {name: digest(a.tobytes()) for name, a in arrays.items()},
        "generated_header_sha256": digest(out.read_bytes()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conv-dir", type=Path, required=True)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    report = generate(args.out, args.conv_dir, args.scan_dir)
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
