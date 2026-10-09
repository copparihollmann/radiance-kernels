#!/usr/bin/env python3
"""Run one complete, deterministic SmolVLA action chunk in pinned LeRobot.

Requires LeRobot 0.5.1, Transformers 5.3.0, PyTorch, and the pinned checkpoint.
This CPU run is a golden software execution, not a Radiance measurement.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from stitch import build, model_specs
from smolvla_action_reference import Checkpoint, action_embedding
from verify_smolvla_checkpoint import sha256


def run(checkpoint_dir: Path) -> dict:
    spec = model_specs()["smolvla_base"]
    if sha256(checkpoint_dir / "config.json") != spec["source_sha256"]:
        raise ValueError("SmolVLA config hash differs from pinned source")
    if sha256(checkpoint_dir / "model.safetensors") != spec["checkpoint_weight_sha256"]:
        raise ValueError("SmolVLA checkpoint hash differs from pinned source")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import lerobot
    import torch
    import transformers
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if lerobot.__version__ != "0.5.1" or transformers.__version__ != "5.3.0":
        raise ValueError("golden run requires pinned LeRobot and Transformers versions")
    torch.set_num_threads(8)
    config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    config.device = "cpu"
    # The policy checkpoint contains the complete VLM. Avoid loading the
    # separate backbone first; strict=True still requires every policy weight.
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(checkpoint_dir, config=config, strict=True)
    graph = build("smolvla_base")
    counts = Counter()
    handles = []
    events = []

    def count(name: str):
        def hook(_module, _inputs, _output):
            counts[name] += 1
            if name == "action_out_proj":
                print(f"denoise step {counts[name]}/{spec['num_denoise_steps']}", flush=True)
        return hook

    model = policy.model
    for name, module in (("vision_model", model.vlm_with_expert.get_vlm_model().vision_model),
                         ("connector", model.vlm_with_expert.get_vlm_model().connector),
                         ("state_proj", model.state_proj),
                         ("action_time_mlp_out", model.action_time_mlp_out),
                         ("action_out_proj", model.action_out_proj)):
        handles.append(module.register_forward_hook(count(name)))

    def trace(kind: str, layer: int, expected_shape: list[int]):
        def hook(_module, inputs, _output):
            actual_shape = tuple(inputs[0].shape)
            if actual_shape != tuple(expected_shape):
                raise ValueError(f"{kind}:{layer} shape {actual_shape} != {expected_shape}")
            events.append(f"{kind}:{layer}")
        return hook

    vlm = model.vlm_with_expert.get_vlm_model()
    expert = model.vlm_with_expert.lm_expert
    for layer, module in enumerate(vlm.vision_model.encoder.layers):
        expected_shape = graph.tensors[f"camera0.vision.layer{layer:02d}.norm1"]["shape"]
        handles.append(module.register_forward_hook(trace("vision", layer, expected_shape)))
    for layer, module in enumerate(vlm.text_model.layers):
        expected_shape = graph.tensors[f"vlm.layer{layer:02d}.input_norm"]["shape"]
        handles.append(module.input_layernorm.register_forward_hook(
            trace("vlm", layer, expected_shape)))
    for layer, module in enumerate(expert.layers):
        expected_shape = graph.tensors[f"denoise0.expert{layer:02d}.input_norm"]["shape"]
        handles.append(module.input_layernorm.register_forward_hook(
            trace("expert", layer, expected_shape)))

    image = torch.linspace(-1.0, 1.0, 3 * spec["image_size"] ** 2,
                           dtype=torch.float32).reshape(1, 3, spec["image_size"],
                                                        spec["image_size"])
    images = [torch.roll(image, shifts=camera * 17, dims=-1)
              for camera in range(spec["image_cameras"])]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in images]
    lang_tokens = torch.arange(1, spec["language_tokens"] + 1, dtype=torch.int64)[None]
    lang_masks = torch.ones_like(lang_tokens, dtype=torch.bool)
    state = torch.linspace(-0.5, 0.5, spec["action_dim"], dtype=torch.float32)[None]
    noise = torch.linspace(-0.25, 0.25,
                           spec["chunk_size"] * spec["action_dim"],
                           dtype=torch.float32).reshape(1, spec["chunk_size"],
                                                        spec["action_dim"])
    with torch.no_grad():
        # Compare the independent NumPy implementation with the exact policy
        # method before running the full denoising loop.
        upstream_suffix = model.embed_suffix(noise, torch.ones(1))[0]
        numpy_suffix = action_embedding(
            noise.numpy(), np.ones(1, dtype=np.float32),
            Checkpoint(checkpoint_dir / "model.safetensors"), spec)["output"]
        suffix_max_error = float(np.max(np.abs(upstream_suffix.numpy() - numpy_suffix)))
        if suffix_max_error > 2e-5:
            raise ValueError(f"action embedding differs from upstream: {suffix_max_error}")
        counts.clear()
        events.clear()
        output = model.sample_actions(images, img_masks, lang_tokens,
                                      lang_masks, state, noise=noise)
    for handle in handles:
        handle.remove()
    expected_counts = {"vision_model": spec["image_cameras"],
                       "connector": spec["image_cameras"], "state_proj": 1,
                       "action_time_mlp_out": spec["num_denoise_steps"],
                       "action_out_proj": spec["num_denoise_steps"]}
    if dict(counts) != expected_counts:
        raise ValueError(f"policy loop counts differ: {dict(counts)}")
    schedule = graph.execution_schedule
    expected_events = [f"vision:{layer}"
                       for branch in schedule["prefix_once_per_refill"]["camera_branches"]
                       for layer, _ in enumerate(branch["vision_layers"])]
    expected_events += [f"vlm:{layer}" for layer, _ in enumerate(
        schedule["prefix_once_per_refill"]["vlm_layers"])]
    expected_events += [f"expert:{layer}"
                        for iteration in schedule["denoise_loop"]["iterations"]
                        for layer, _ in enumerate(iteration["expert_layers"])]
    if events != expected_events:
        first = next((index for index, pair in enumerate(zip(events, expected_events))
                      if pair[0] != pair[1]), min(len(events), len(expected_events)))
        raise ValueError(f"policy execution order differs from graph at event {first}")
    if tuple(output.shape) != (1, spec["chunk_size"], spec["action_dim"]):
        raise ValueError(f"policy output shape differs: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise ValueError("policy output is nonfinite")
    values = output.to(torch.float32).cpu().numpy()
    inputs = {f"camera{index}": image.numpy() for index, image in enumerate(images)}
    inputs.update({"language": lang_tokens.numpy(), "state": state.numpy(),
                   "noise": noise.numpy()})
    return {
        "model": "smolvla_base", "scope": "one_full_checkpoint_action_chunk",
        "passed": True, "checkpoint_revision": spec["checkpoint_revision"],
        "checkpoint_weight_sha256": spec["checkpoint_weight_sha256"],
        "lerobot_version": lerobot.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": "cpu", "strict_checkpoint_load": True,
        "load_vlm_weights_before_checkpoint_load": False,
        "policy_method": "model.sample_actions",
        "preprocessed_inputs": True,
        "input_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest()
                         for name, value in inputs.items()},
        "observed_module_calls": dict(counts),
        "observed_layer_calls": {
            "vision": sum(event.startswith("vision:") for event in events),
            "vlm": sum(event.startswith("vlm:") for event in events),
            "expert": sum(event.startswith("expert:") for event in events),
        },
        "graph_order_and_shapes_match_upstream": True,
        "layer_event_sequence_sha256": hashlib.sha256(
            "\n".join(events).encode()).hexdigest(),
        "action_embedding_upstream_max_abs_error": suffix_max_error,
        "output_shape": list(values.shape),
        "output_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "output_min": float(np.min(values)), "output_max": float(np.max(values)),
        "radiance_execution": False, "radiance_comparison": False,
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
