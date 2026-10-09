#!/usr/bin/env python3
"""Check the accessible pinned model configs behind pr1-models.json."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from stitch import model_specs


def fetch_bytes(url: str, expected_hash: str) -> bytes:
    with urlopen(url, timeout=30) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"source SHA-256 mismatch: {url}")
    return raw


def fetch(url: str, expected_hash: str) -> dict:
    return json.loads(fetch_bytes(url, expected_hash))


def check_fields(spec: dict, config: dict, fields: tuple[str, ...]) -> None:
    for field in fields:
        if spec[field] != config[field]:
            raise ValueError(f"{spec['name']}: {field} differs from pinned config")


def main() -> None:
    specs = model_specs()
    decoder_fields = ("hidden_size", "intermediate_size", "num_hidden_layers",
                      "num_attention_heads", "num_key_value_heads", "vocab_size",
                      "rms_norm_eps")
    for name in ("tinyllama", "deepseek_r1_distill_qwen_1_5b"):
        spec = specs[name]
        config = fetch(spec["source"], spec["source_sha256"])
        check_fields(spec, config, decoder_fields)
        if spec["head_dim"] != config["hidden_size"] // config["num_attention_heads"]:
            raise ValueError(f"{name}: head dimension differs")
        if name == "deepseek_r1_distill_qwen_1_5b":
            check_fields(spec, config, ("use_sliding_window",))
        print(f"{name}: pinned config and dimensions match")

    policy = specs["smolvla_base"]
    config = fetch(policy["source"], policy["source_sha256"])
    check_fields(policy, config, ("num_vlm_layers", "chunk_size", "n_action_steps"))
    if policy["output_action_dim"] != config["output_features"]["action"]["shape"][0]:
        raise ValueError("SmolVLA returned action dimension differs")
    if config.get("rtc_config") is not None:
        raise ValueError("SmolVLA graph does not model real-time chunking")
    if policy["num_denoise_steps"] != config["num_steps"]:
        raise ValueError("SmolVLA denoising step count differs")
    if policy["image_cameras"] != sum(
            feature["type"] == "VISUAL" for feature in config["input_features"].values()):
        raise ValueError("SmolVLA camera count differs")
    if policy["action_dim"] != config["max_action_dim"]:
        raise ValueError("SmolVLA action dimension differs")
    if policy["action_dim"] != config["max_state_dim"]:
        raise ValueError("SmolVLA padded state dimension differs")
    if (policy["language_tokens"] != config["tokenizer_max_length"]
            or [policy["image_size"]] * 2 != config["resize_imgs_with_padding"]
            or config["add_image_special_tokens"] or config["prefix_length"]):
        raise ValueError("SmolVLA prefix assumptions differ")
    if policy["attention_mode"] != config["attention_mode"] or policy[
            "self_attn_every_n_layers"] != config["self_attn_every_n_layers"]:
        raise ValueError("SmolVLA expert attention mode differs")
    expert_layers = config["num_expert_layers"] or config["num_vlm_layers"]
    if policy["num_expert_layers"] != expert_layers:
        raise ValueError("SmolVLA expert depth differs")
    backbone = fetch(policy["backbone_source"], policy["backbone_source_sha256"])
    if (policy["vlm_hidden_size"] != backbone["text_config"]["hidden_size"] or
            policy["vision_hidden_size"] != backbone["vision_config"]["hidden_size"] or
            policy["patch_size"] != backbone["vision_config"]["patch_size"]):
        raise ValueError("SmolVLA backbone dimensions differ")
    if policy["expert_hidden_size"] != int(backbone["text_config"]["hidden_size"] *
                                         config["expert_width_multiplier"]):
        raise ValueError("SmolVLA expert width differs")
    if (policy["vlm_num_key_value_heads"] != backbone["text_config"]["num_key_value_heads"]
            or policy["vlm_head_dim"] != backbone["text_config"]["head_dim"]):
        raise ValueError("SmolVLA VLM KV dimensions differ")
    if backbone["scale_factor"] != 4:
        raise ValueError("SmolVLA image connector scale differs")
    implementation = json.loads((Path(__file__).resolve().parent /
                                 "smolvla-implementation.json").read_text())
    for source, digest in (("modeling_source", "modeling_source_sha256"),
                           ("expert_source", "expert_source_sha256")):
        fetch_bytes(implementation[source], implementation[digest])
    print("smolvla_base: pinned policy, backbone, and LeRobot sources match")
    print("gemma_2_2b_it: official config access gated; fields were not network-verified")


if __name__ == "__main__":
    main()
