#!/usr/bin/env python3
"""Extract a reduced Mamba2 convolution fixture from Transformers' CPU fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource

import numpy as np
import torch
import transformers
from transformers.models.nemotron_h import modeling_nemotron_h
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHMamba2Mixer, NemotronHHybridDynamicCache,
)

from gen_data import (CHANNELS, KERNEL_WIDTH, PREFILL, TOKENS,
                      emit_fixture, reference, sha)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(out: Path) -> dict:
    torch.set_num_threads(1)
    config = NemotronHConfig(
        layers_block_type=["mamba"], hidden_size=32, intermediate_size=64,
        vocab_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, mamba_num_heads=4, mamba_head_dim=8,
        mamba_n_groups=2, ssm_state_size=4, mamba_d_conv=KERNEL_WIDTH,
        mamba_chunk_size=8, use_mamba_kernels=False,
    )
    torch.manual_seed(17)
    mixer = NemotronHMamba2Mixer(config, layer_idx=0).eval()
    if mixer.conv_dim != CHANNELS or mixer.conv1d.bias is None:
        raise ValueError("reduced mixer does not match convolution fixture")
    values = torch.randn(1, TOKENS, 32) * 0.1
    captured = []

    def capture_conv(_module, _inputs, output):
        captured.append(output.detach().clone())

    hook = mixer.act.register_forward_hook(capture_conv)
    with torch.no_grad():
        cache = NemotronHHybridDynamicCache(
            config, batch_size=1, dtype=torch.float32, device="cpu")
        outputs = [mixer(values[:, :PREFILL], cache_params=cache)]
        cache.has_previous_state = True
        outputs.extend(mixer(values[:, t:t + 1], cache_params=cache)
                       for t in range(PREFILL, TOKENS))
        hook.remove()
        if (len(captured) != 1 + TOKENS - PREFILL
                or tuple(captured[0].shape) != (1, PREFILL + KERNEL_WIDTH - 1, CHANNELS)
                or any(tuple(item.shape) != (1, CHANNELS) for item in captured[1:])):
            raise ValueError("upstream convolution activation capture changed")

        projected = mixer.in_proj(values)
        d_mlp = (projected.shape[-1] - 2 * mixer.intermediate_size
                 - 2 * mixer.n_groups * mixer.ssm_state_size
                 - mixer.num_heads) // 2
        if d_mlp != 0:
            raise ValueError("reduced fixture expects no extra MLP slice")
        _, _, _, conv_input, _ = projected.split(
            [d_mlp, d_mlp, mixer.intermediate_size, CHANNELS,
             mixer.num_heads], dim=-1)

    as_np = lambda tensor: np.ascontiguousarray(tensor.detach().numpy(),
                                                 dtype=np.float32)
    x = as_np(conv_input[0])
    weight = as_np(mixer.conv1d.weight[:, 0, :])
    bias = as_np(mixer.conv1d.bias)
    initial = np.zeros((CHANNELS, KERNEL_WIDTH), dtype=np.float32)
    gold_y, gold_state = reference(x, weight, bias, initial)
    upstream_y = as_np(torch.cat(
        [captured[0][:, :PREFILL]]
        + [item[:, None, :] for item in captured[1:]], dim=1)[0])
    upstream_state = as_np(cache.conv_states[0][0])
    output_error = float(np.max(np.abs(gold_y - upstream_y)))
    state_error = float(np.max(np.abs(gold_state - upstream_state)))
    if (not np.allclose(gold_y, upstream_y, rtol=3e-5, atol=3e-6)
            or not np.allclose(gold_state, upstream_state,
                               rtol=3e-5, atol=3e-6)):
        raise AssertionError("independent convolution differs from upstream mixer")
    provenance = {
        "scope": "reduced random-weight Mamba2 mixer, no pinned checkpoint",
        "reference_backend": "transformers_cpu_fallback",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "modeling_source_sha256": file_sha(Path(modeling_nemotron_h.__file__)),
        "seed": 17, "chunk_size": 8,
        "upstream_input_sha256": sha(as_np(values)),
        "upstream_mixer_output_sha256": sha(as_np(torch.cat(outputs, dim=1))),
        "upstream_conv_sha256": sha(upstream_y),
        "upstream_conv_state_sha256": sha(upstream_state),
        "numpy_vs_upstream_conv_max_abs": output_error,
        "numpy_vs_upstream_state_max_abs": state_error,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    return emit_fixture(out, x=x, weight=weight, bias=bias,
                        initial=initial, gold_y=gold_y, gold_state=gold_state,
                        source="transformers_reduced", provenance=provenance)


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
