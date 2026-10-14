#!/usr/bin/env python3
"""Compare cached and full CPU Mamba scans around a chunk boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource


def digest(tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def run_case(chunk_size: int, prefill: int, total: int) -> dict:
    import torch
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        NemotronHMamba2Mixer, NemotronHHybridDynamicCache,
    )

    config = NemotronHConfig(
        layers_block_type=["mamba"], hidden_size=32, intermediate_size=64,
        vocab_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, mamba_num_heads=4, mamba_head_dim=8,
        mamba_n_groups=2, ssm_state_size=4, mamba_d_conv=3,
        mamba_chunk_size=chunk_size, use_mamba_kernels=False,
    )
    torch.manual_seed(17)
    mixer = NemotronHMamba2Mixer(config, layer_idx=0).eval()
    values = torch.randn(1, total, 32) * 0.1
    with torch.no_grad():
        full = mixer(values, cache_params=None)
        cache = NemotronHHybridDynamicCache(
            config, batch_size=1, dtype=torch.float32, device="cpu")
        pieces = [mixer(values[:, :prefill], cache_params=cache)]
        cache.has_previous_state = True
        pieces.extend(mixer(values[:, i:i + 1], cache_params=cache)
                      for i in range(prefill, total))
        cached = torch.cat(pieces, dim=1)
    error = (full - cached).abs().amax(dim=(0, 2))
    above = torch.nonzero(error > 1e-5).flatten().tolist()
    return {
        "chunk_size": chunk_size, "prefill_tokens": prefill,
        "total_tokens": total, "input_sha256": digest(values),
        "full_output_sha256": digest(full),
        "cached_output_sha256": digest(cached),
        "max_abs_error": float(error.max()),
        "prefill_max_abs_error": float(error[:prefill].max()),
        "first_token_above_1e_5": above[0] if above else None,
        "tail_token_errors": [float(x) for x in error[-3:]],
        "conv_state_shape": list(cache.conv_states[0].shape),
        "ssm_state_shape": list(cache.ssm_states[0].shape),
    }


def verify() -> dict:
    import torch
    import transformers

    torch.set_num_threads(1)
    inside = run_case(chunk_size=8, prefill=3, total=5)
    crossing = run_case(chunk_size=256, prefill=255, total=257)
    if inside["max_abs_error"] > 1e-5:
        raise ValueError("cached/full Mamba control differs inside one chunk")
    return {
        "schema": "radiance.nemotron-mamba-cache-control.v1",
        "scope": "one reduced random-weight Mamba2 mixer, CPU torch fallback; no pinned checkpoint or Radiance execution",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "random_seed": 17,
        "inside_one_chunk": inside,
        "crossing_pinned_chunk_size": crossing,
        "crossing_status": (
            "cached_full_difference_above_1e-5" if crossing["max_abs_error"] > 1e-5
            else "within_1e-5"),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = json.dumps(verify(), indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
