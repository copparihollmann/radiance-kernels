#!/usr/bin/env python3
"""Compare one stitched Nemotron Mamba layer with Transformers CPU fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource

import numpy as np

import nemotron_graph_reference
import reference
import stitch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare() -> dict:
    import torch
    import transformers
    from transformers.models.nemotron_h import modeling_nemotron_h
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        NemotronHMamba2Mixer, NemotronHHybridDynamicCache)

    torch.set_num_threads(1)
    config = NemotronHConfig(
        layers_block_type=["mamba"], hidden_size=32, intermediate_size=64,
        vocab_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, mamba_num_heads=4, mamba_head_dim=8,
        mamba_n_groups=2, ssm_state_size=4, mamba_d_conv=4,
        mamba_chunk_size=8, use_mamba_kernels=False)
    if config.time_step_limit != (0.0, float("inf")):
        raise ValueError("unexpected Transformers time-step limit")
    torch.manual_seed(17)
    mixer = NemotronHMamba2Mixer(config, layer_idx=0).eval()
    embedding = np.random.default_rng(19).standard_normal((64, 32)).astype(np.float32) * 0.1

    class Weights:
        def embedding_row(self, name: str, row: int, width: int) -> np.ndarray:
            if name != "backbone.embeddings.weight" or width != 32:
                raise ValueError(f"unexpected embedding {name}")
            return embedding[row]

        def __call__(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
            if name in ("backbone.layers.0.norm.weight", "backbone.norm_f.weight"):
                value = np.ones(shape, dtype=np.float32)
            elif name == "lm_head.weight":
                value = reference.parameter(name, shape)
            else:
                prefix = "backbone.layers.0.mixer."
                if not name.startswith(prefix):
                    raise ValueError(f"unexpected parameter {name}")
                component = name[len(prefix):]
                target = mixer
                for part in component.split("."):
                    target = getattr(target, part)
                value = target.detach().numpy().astype(np.float32)
                if component in ("in_proj.weight", "out_proj.weight"):
                    value = value.T
            if value.shape != shape:
                raise ValueError(f"{name}: weight shape {value.shape} != {shape}")
            return value

    pinned = stitch.model_specs()["nemotron3_nano_4b"]
    spec = nemotron_graph_reference.control_spec(pinned, 1)
    graph = stitch.build("nemotron3_nano_4b", prefill=3, decode_steps=2,
                         generation="teacher_forced",
                         specs={"nemotron3_nano_4b": spec})
    ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    values = nemotron_graph_reference.execute(graph, {
        "prefill.token_ids": ids[:, :3],
        "decode0.token_ids": ids[:, 3:4],
        "decode1.token_ids": ids[:, 4:5],
    }, weights=Weights())
    differences: dict[str, float] = {}
    with torch.no_grad():
        cache = NemotronHHybridDynamicCache(
            config, batch_size=1, dtype=torch.float32, device="cpu")
        for phase in ("prefill", "decode0", "decode1"):
            input_tensor = values[f"{phase}.layer00.pre_norm"]
            projected = mixer.in_proj(torch.from_numpy(input_tensor)).numpy()
            differences[f"{phase}.projection"] = float(np.max(np.abs(
                projected - values[f"{phase}.layer00.in_proj"])))
            output = mixer(torch.from_numpy(input_tensor), cache_params=cache).numpy()
            differences[f"{phase}.mixer_output"] = float(np.max(np.abs(
                output - values[f"{phase}.layer00.out_proj"])))
            cache.has_previous_state = True
        differences["final_conv_state"] = float(np.max(np.abs(
            cache.conv_states[0].numpy() - values["decode1.layer00.conv_state"])))
        differences["final_ssm_state"] = float(np.max(np.abs(
            cache.ssm_states[0].numpy() - values["decode1.layer00.ssm_state"])))
    for name, error in differences.items():
        if error > (2e-5 if name.endswith("mixer_output") else 1e-5):
            raise AssertionError(f"{name}: upstream difference {error}")
    return {
        "schema": "radiance.nemotron-graph-mixer-reference.v1",
        "status": "passed", "scope": "one reduced random-weight Mamba layer, three-token prefill and two cached decode calls",
        "pinned_model": pinned["name"],
        "pinned_config_sha256": pinned["source_sha256"],
        "graph_stage_count": len(graph.stages),
        "input_token_ids": ids.ravel().tolist(),
        "max_abs_error_by_check": differences,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "transformers_modeling_source_sha256": sha256(Path(modeling_nemotron_h.__file__)),
        "graph_reference_source_sha256": sha256(Path(nemotron_graph_reference.__file__)),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "device_execution": False,
        "checkpoint_weights": False,
        "full_network_equivalence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("matched one stitched Mamba layer to Transformers CPU fallback")


if __name__ == "__main__":
    main()
