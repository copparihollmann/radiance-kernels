#!/usr/bin/env python3
"""Compare the checkpoint-backed NumPy action embedding with PyTorch.

The PyTorch path follows pinned LeRobot v0.5.1 `embed_suffix` using the six
real action embedding tensors. It does not instantiate the full policy.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from smolvla_action_reference import Checkpoint, action_embedding
from stitch import model_specs
from verify_smolvla_checkpoint import sha256


def pytorch_action_embedding(actions: torch.Tensor, timestep: torch.Tensor,
                             weights: dict[str, torch.Tensor], spec: dict) -> dict:
    def project(name: str, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, weights[f"model.{name}.weight"],
                        weights[f"model.{name}.bias"])

    action_projection = project("action_in_proj", actions)
    width = spec["expert_hidden_size"]
    fraction = torch.linspace(0.0, 1.0, width // 2, dtype=torch.float64)
    period = spec["min_period"] * (spec["max_period"] / spec["min_period"]) ** fraction
    phase = (2 * math.pi / period)[None, :] * timestep[:, None].to(torch.float64)
    time_embedding = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
    time_embedding = time_embedding.to(action_projection.dtype)
    joined = torch.cat((action_projection,
                        time_embedding[:, None, :].expand_as(action_projection)), dim=2)
    mlp_in = project("action_time_mlp_in", joined)
    activated = F.silu(mlp_in)
    output = project("action_time_mlp_out", activated)
    return {"action_projection": action_projection, "time_embedding": time_embedding,
            "action_time_concat": joined, "mlp_in": mlp_in,
            "activation": activated, "output": output}


def run(checkpoint_dir: Path) -> dict:
    spec = model_specs()["smolvla_base"]
    weight_path = checkpoint_dir / "model.safetensors"
    if sha256(checkpoint_dir / "config.json") != spec["source_sha256"]:
        raise ValueError("checkpoint config differs from pinned SmolVLA config")
    if sha256(weight_path) != spec["checkpoint_weight_sha256"]:
        raise ValueError("checkpoint weights differ from pinned SmolVLA checkpoint")
    checkpoint = Checkpoint(weight_path)
    names = [f"model.{module}.{kind}"
             for module in ("action_in_proj", "action_time_mlp_in",
                            "action_time_mlp_out")
             for kind in ("weight", "bias")]
    with safe_open(weight_path, framework="pt", device="cpu") as stream:
        weights = {name: stream.get_tensor(name) for name in names}
    shape = (1, spec["chunk_size"], spec["action_dim"])
    actions = np.linspace(-0.25, 0.25, math.prod(shape), dtype=np.float32).reshape(shape)
    timestep = np.array([1.0], dtype=np.float32)
    numpy_values = action_embedding(actions, timestep, checkpoint, spec)
    with torch.no_grad():
        torch_values = pytorch_action_embedding(torch.from_numpy(actions),
                                                torch.from_numpy(timestep), weights, spec)
    maximum_errors = {}
    for name, expected in torch_values.items():
        actual = numpy_values[name]
        reference = expected.numpy()
        if actual.shape != reference.shape:
            raise ValueError(f"{name}: shape mismatch")
        error = float(np.max(np.abs(actual - reference)))
        maximum_errors[name] = error
        if not np.allclose(actual, reference, rtol=1e-4, atol=2e-5):
            raise ValueError(f"{name}: NumPy differs from PyTorch, max error {error}")
    return {
        "model": "smolvla_base",
        "scope": "checkpoint_action_time_embedding_only",
        "check": "numpy_vs_pytorch_pinned_source_formula",
        "passed": True, "checkpoint_revision": spec["checkpoint_revision"],
        "checkpoint_weight_sha256": spec["checkpoint_weight_sha256"],
        "torch_version": torch.__version__,
        "tolerance": {"rtol": 1e-4, "atol": 2e-5},
        "max_abs_error_by_stage": maximum_errors,
        "pytorch_output_sha256": hashlib.sha256(torch_values["output"].numpy().tobytes()).hexdigest(),
        "full_policy_executed": False, "device_execution": False,
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
