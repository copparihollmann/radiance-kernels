#!/usr/bin/env python3
"""Extract a reduced Mamba2 scan fixture from Transformers' CPU fallback."""

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

from gen_data import (GROUPS, HEADS, HEAD_DIM, PREFILL, STATE_SIZE, TOKENS,
                      emit_fixture, reference, sha)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(out: Path) -> dict:
    torch.set_num_threads(1)
    config = NemotronHConfig(
        layers_block_type=["mamba"], hidden_size=32, intermediate_size=64,
        vocab_size=64, num_attention_heads=HEADS, num_key_value_heads=2,
        head_dim=HEAD_DIM, mamba_num_heads=HEADS, mamba_head_dim=HEAD_DIM,
        mamba_n_groups=GROUPS, ssm_state_size=STATE_SIZE, mamba_d_conv=3,
        mamba_chunk_size=8, use_mamba_kernels=False,
    )
    torch.manual_seed(17)
    mixer = NemotronHMamba2Mixer(config, layer_idx=0).eval()
    values = torch.randn(1, TOKENS, 32) * 0.1
    captured = []

    def capture_scan(_module, inputs):
        captured.append(inputs[0].detach().clone())

    hook = mixer.norm.register_forward_pre_hook(capture_scan)
    with torch.no_grad():
        cache = NemotronHHybridDynamicCache(
            config, batch_size=1, dtype=torch.float32, device="cpu")
        phase_outputs = [mixer(values[:, :PREFILL], cache_params=cache)]
        cache.has_previous_state = True
        phase_outputs.extend(
            mixer(values[:, t:t + 1], cache_params=cache)
            for t in range(PREFILL, TOKENS))
        hook.remove()

        projected = mixer.in_proj(values)
        conv_width = mixer.intermediate_size + 2 * GROUPS * STATE_SIZE
        d_mlp = (projected.shape[-1] - 2 * mixer.intermediate_size
                 - 2 * GROUPS * STATE_SIZE - HEADS) // 2
        if d_mlp != 0:
            raise ValueError("reduced fixture expects no extra MLP slice")
        _, _, gate, conv_input, raw_dt = projected.split(
            [d_mlp, d_mlp, mixer.intermediate_size, conv_width, HEADS],
            dim=-1)
        conv = mixer.act(mixer.conv1d(conv_input.transpose(1, 2))
                         [..., :TOKENS].transpose(1, 2))
        x, b, c = conv.split(
            [mixer.intermediate_size, GROUPS * STATE_SIZE,
             GROUPS * STATE_SIZE], dim=-1)
        dt = torch.nn.functional.softplus(raw_dt + mixer.dt_bias)
        dt = torch.clamp(dt, mixer.time_step_min)
        a = -torch.exp(mixer.A_log.float())
        d = mixer.D

        as_np = lambda tensor: np.ascontiguousarray(tensor.detach().numpy(),
                                                     dtype=np.float32)
        x_np = as_np(x[0])
        b_np = as_np(b[0].reshape(TOKENS, GROUPS, STATE_SIZE))
        c_np = as_np(c[0].reshape(TOKENS, GROUPS, STATE_SIZE))
        dt_np, a_np, d_np = as_np(dt[0]), as_np(a), as_np(d)
        initial = np.zeros((HEADS * HEAD_DIM, STATE_SIZE), dtype=np.float32)
        gold_y, gold_state = reference(
            x_np, b_np, c_np, dt_np, a_np, d_np, initial, PREFILL)
        upstream_scan = as_np(torch.cat(captured, dim=1)[0])
        upstream_state = as_np(cache.ssm_states[0][0].reshape(
            HEADS * HEAD_DIM, STATE_SIZE))
        output_from_reference = mixer.out_proj(
            mixer.norm(torch.from_numpy(gold_y).reshape(1, TOKENS, -1), gate))
        upstream_output = as_np(torch.cat(phase_outputs, dim=1))
        mapped_output = as_np(output_from_reference)

    scan_error = float(np.max(np.abs(gold_y - upstream_scan)))
    state_error = float(np.max(np.abs(gold_state - upstream_state)))
    output_error = float(np.max(np.abs(mapped_output - upstream_output)))
    if (not np.allclose(gold_y, upstream_scan, rtol=3e-5, atol=3e-6)
            or not np.allclose(gold_state, upstream_state,
                               rtol=3e-5, atol=3e-6)
            or not np.allclose(mapped_output, upstream_output,
                               rtol=3e-5, atol=3e-6)):
        raise AssertionError("independent recurrence differs from upstream mixer")
    provenance = {
        "scope": "reduced random-weight Mamba2 mixer, no pinned checkpoint",
        "reference_backend": "transformers_cpu_fallback",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "modeling_source_sha256": file_sha(Path(modeling_nemotron_h.__file__)),
        "seed": 17, "chunk_size": 8, "time_step_min": mixer.time_step_min,
        "upstream_input_sha256": sha(as_np(values)),
        "upstream_scan_sha256": sha(upstream_scan),
        "upstream_final_state_sha256": sha(upstream_state),
        "upstream_output_sha256": sha(upstream_output),
        "numpy_vs_upstream_scan_max_abs": scan_error,
        "numpy_vs_upstream_state_max_abs": state_error,
        "mapped_vs_upstream_output_max_abs": output_error,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    return emit_fixture(out, x=x_np, b=b_np, c=c_np, dt=dt_np,
                        a=a_np, d=d_np, initial=initial,
                        gold_y=gold_y, gold_state=gold_state,
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
