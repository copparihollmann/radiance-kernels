#!/usr/bin/env python3
"""Join the checked conv/scan fixture with the remaining upstream mixer data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import sys

import numpy as np
import torch
import transformers
from transformers.models.nemotron_h import modeling_nemotron_h
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHMamba2Mixer, NemotronHHybridDynamicCache,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mamba_conv_scan"))
from gen_data import generate as generate_base, load_array, digest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib/golden"))
import golden as G


def generate(out: Path, conv_dir: Path, scan_dir: Path) -> dict:
    base = generate_base(out, conv_dir, scan_dir)
    original = out.read_text()
    torch.set_num_threads(1)
    config = NemotronHConfig(
        layers_block_type=["mamba"], hidden_size=32, intermediate_size=64,
        vocab_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, mamba_num_heads=4, mamba_head_dim=8,
        mamba_n_groups=2, ssm_state_size=4, mamba_d_conv=4,
        mamba_chunk_size=8, use_mamba_kernels=False,
    )
    torch.manual_seed(17)
    mixer = NemotronHMamba2Mixer(config, layer_idx=0).eval()
    values = torch.randn(1, 7, 32) * 0.1
    with torch.no_grad():
        cache = NemotronHHybridDynamicCache(
            config, batch_size=1, dtype=torch.float32, device="cpu")
        outputs = [mixer(values[:, :5], cache_params=cache)]
        cache.has_previous_state = True
        outputs.extend(mixer(values[:, t:t + 1], cache_params=cache)
                       for t in range(5, 7))
        projected = mixer.in_proj(values)[0]
    as_np = lambda t: np.ascontiguousarray(t.detach().numpy(), dtype=np.float32)
    projected_np = as_np(projected)
    conv_expected = load_array(original, "conv_x_raw", 336).reshape(7, 48)
    dt_expected = load_array(original, "dt_raw", 28).reshape(7, 4)
    if (not np.array_equal(projected_np[:, 32:80], conv_expected)
            or not np.array_equal(projected_np[:, 80:84], dt_expected)):
        raise ValueError("input projection does not match checked conv/scan fixtures")
    full_y = as_np(torch.cat(outputs, dim=1)[0])
    if digest(full_y.tobytes()) != base_source_full_hash(conv_dir, scan_dir):
        raise ValueError("upstream mixer output changed from source fixture")
    arrays = {
        "values_raw": as_np(values[0]).ravel(),
        "in_weight_raw": as_np(mixer.in_proj.weight).ravel(),
        "norm_weight_raw": as_np(mixer.norm.weight).ravel(),
        "out_weight_raw": as_np(mixer.out_proj.weight).ravel(),
        "gold_gate_raw": projected_np[:, :32].ravel(),
        "gold_conv_x_raw": conv_expected.ravel(),
        "gold_dt_raw": dt_expected.ravel(),
        "gold_full_y_raw": full_y.ravel(),
    }
    if mixer.in_proj.bias is not None or mixer.out_proj.bias is not None:
        raise ValueError("reduced mixer expects bias-free linear layers")
    if mixer.norm.group_size != 16:
        raise ValueError("reduced mixer expects two normalization groups")
    source = original
    for name, size in (("conv_x_raw", 336), ("dt_raw", 28)):
        source, count = re.subn(
            rf"__global float {name}\[{size}\] = \{{.*?\}};",
            f"__global float {name}[{size}] = {{0}};", source,
            flags=re.S, count=1)
        if count != 1:
            raise ValueError(f"cannot replace preloaded {name}")
    with out.open("w") as stream:
        stream.write(source)
        stream.write("#define HIDDEN_SIZE 32\n#define SCAN_CHANNELS 32\n")
        stream.write("#define PROJ_SIZE 84\n#define NORM_GROUP_SIZE 16\n")
        stream.write(f"#define NORM_EPS {mixer.norm.variance_epsilon:.9e}f\n")
        for name, array in arrays.items():
            G.emit_f32(stream, name, array, dims=f"[{array.size}]",
                       trailing_comma=True)
    return {
        "schema": "radiance.mamba-mixer-fixture.v1",
        "source": "transformers_reduced",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "modeling_source_sha256": digest(Path(modeling_nemotron_h.__file__).read_bytes()),
        "base_fixture": base,
        "array_sha256": {name: digest(array.tobytes()) for name, array in arrays.items()},
        "generated_header_sha256": digest(out.read_bytes()),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def base_source_full_hash(conv_dir: Path, scan_dir: Path) -> str:
    conv = json.loads((conv_dir / "fixture.json").read_text())
    scan = json.loads((scan_dir / "fixture.json").read_text())
    a = conv["provenance"]["upstream_mixer_output_sha256"]
    b = scan["provenance"]["upstream_output_sha256"]
    if a != b:
        raise ValueError("source full mixer hashes differ")
    return a


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--conv-dir", type=Path, required=True)
    parser.add_argument("--scan-dir", type=Path, required=True)
    args = parser.parse_args()
    report = generate(args.out, args.conv_dir, args.scan_dir)
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
