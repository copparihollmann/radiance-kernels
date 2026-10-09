#!/usr/bin/env python3
"""Run SmolVLA's action/time embedding with its pinned checkpoint weights.

This covers `embed_suffix` up to the expert input. It is a real-weight
numerical control for one operator chain, not a full policy inference run.
"""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

from stitch import model_specs
from verify_smolvla_checkpoint import safetensors_header, sha256


class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as stream:
            self.header_size = struct.unpack("<Q", stream.read(8))[0]
        self.header = safetensors_header(path)

    def tensor(self, name: str) -> np.ndarray:
        item = self.header[name]
        start, end = item["data_offsets"]
        count = math.prod(item["shape"])
        offset = 8 + self.header_size + start
        if item["dtype"] == "F32":
            if end - start != 4 * count:
                raise ValueError(f"{name}: invalid F32 size")
            return np.memmap(self.path, dtype="<f4", mode="r", offset=offset,
                             shape=tuple(item["shape"]))
        if item["dtype"] == "BF16":
            if end - start != 2 * count:
                raise ValueError(f"{name}: invalid BF16 size")
            bits = np.memmap(self.path, dtype="<u2", mode="r", offset=offset,
                             shape=(count,)).astype(np.uint32) << 16
            return bits.view("<f4").reshape(item["shape"])
        raise ValueError(f"{name}: unsupported dtype {item['dtype']}")


def action_embedding(actions: np.ndarray, timestep: np.ndarray,
                     checkpoint: Checkpoint, spec: dict) -> dict[str, np.ndarray]:
    """Follow LeRobot v0.5.1 `embed_suffix` before the action-token masks."""
    if actions.ndim != 3 or actions.shape[-1] != spec["action_dim"]:
        raise ValueError("actions must have shape [batch, chunk, padded action dim]")
    if timestep.shape != (actions.shape[0],):
        raise ValueError("timestep must have shape [batch]")

    def project(name: str, value: np.ndarray) -> np.ndarray:
        weight = checkpoint.tensor(f"model.{name}.weight")
        bias = checkpoint.tensor(f"model.{name}.bias")
        return (value @ weight.T + bias).astype(np.float32)

    projected_actions = project("action_in_proj", actions)
    width = spec["expert_hidden_size"]
    fraction = np.linspace(0.0, 1.0, width // 2, dtype=np.float64)
    period = spec["min_period"] * (spec["max_period"] / spec["min_period"]) ** fraction
    phase = (2.0 * np.pi / period)[None, :] * timestep[:, None].astype(np.float64)
    time_embedding = np.concatenate((np.sin(phase), np.cos(phase)), axis=-1)
    time_embedding = time_embedding.astype(projected_actions.dtype)
    broadcast_time = np.broadcast_to(time_embedding[:, None, :], projected_actions.shape)
    joined = np.concatenate((projected_actions, broadcast_time), axis=-1)
    mlp_in = project("action_time_mlp_in", joined)
    activated = mlp_in / (1.0 + np.exp(-mlp_in))
    embedded = project("action_time_mlp_out", activated)
    return {"action_projection": projected_actions, "time_embedding": time_embedding,
            "action_time_concat": joined, "mlp_in": mlp_in,
            "activation": activated.astype(np.float32), "output": embedded}


def run(checkpoint_dir: Path) -> dict:
    spec = model_specs()["smolvla_base"]
    if sha256(checkpoint_dir / "config.json") != spec["source_sha256"]:
        raise ValueError("checkpoint config differs from pinned SmolVLA config")
    weight_path = checkpoint_dir / "model.safetensors"
    if sha256(weight_path) != spec["checkpoint_weight_sha256"]:
        raise ValueError("checkpoint weights differ from pinned SmolVLA checkpoint")
    checkpoint = Checkpoint(weight_path)
    shape = (1, spec["chunk_size"], spec["action_dim"])
    actions = np.linspace(-0.25, 0.25, math.prod(shape), dtype=np.float32).reshape(shape)
    values = action_embedding(actions, np.array([1.0], dtype=np.float32), checkpoint, spec)
    altered_time = action_embedding(actions, np.array([0.9], dtype=np.float32),
                                    checkpoint, spec)
    altered_actions = actions.copy()
    altered_actions[0, 0, 0] += 0.125
    altered_input = action_embedding(altered_actions,
                                     np.array([1.0], dtype=np.float32), checkpoint, spec)
    output = values["output"]
    if not np.isfinite(output).all():
        raise ValueError("action embedding produced nonfinite values")
    time_delta = float(np.max(np.abs(output - altered_time["output"])))
    action_delta = float(np.max(np.abs(output - altered_input["output"])))
    if time_delta <= 0 or action_delta <= 0:
        raise ValueError("action embedding ignored a required input")
    return {
        "model": "smolvla_base", "scope": "checkpoint_action_time_embedding_only",
        "checkpoint_revision": spec["checkpoint_revision"],
        "checkpoint_weight_sha256": spec["checkpoint_weight_sha256"],
        "input_shape": list(shape), "output_shape": list(output.shape),
        "timestep": 1.0,
        "stage_output_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest()
                                for name, value in values.items()},
        "time_change_max_abs_delta": time_delta,
        "action_change_max_abs_delta": action_delta,
        "all_finite": True,
        "upstream_pytorch_comparison": False,
        "full_model_execution": False, "device_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(args.checkpoint_dir)
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
