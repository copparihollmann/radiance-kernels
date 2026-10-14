#!/usr/bin/env python3
"""Compare the existing reduced OLMo 3 device fixture with upstream OLMo 3.

Uses the generated FP32 parameter declarations of the saved Cyclotron build.
The three-token execution does not cross the pinned 4,096-token local window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys

import numpy as np
import torch
import transformers

import olmo_graph_reference
from reference import reduced_spec
from stitch import build, model_specs
from verify_olmo3_reduced import parameter_key

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "kernels/model_chain"))
from compile_decoder import array_decl, param_data  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def compare(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    difference = np.abs(actual - expected)
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(difference.max()),
        "out_of_tolerance": int(np.count_nonzero(
            difference > 1e-4 + 1e-3 * np.abs(expected))),
        "mapped_sha256": digest(actual),
        "upstream_sha256": digest(expected),
    }


def check(build_dir: Path) -> dict:
    if transformers.__version__ != "5.3.0":
        raise ValueError("this check requires Transformers 5.3.0")
    torch.set_num_threads(1)
    build_receipt_path = ROOT / "kernels/model_chain/evaluation/olmo3-full-depth-reduced-build.json"
    device_receipt_path = ROOT / "kernels/model_chain/evaluation/olmo3-full-depth-reduced-functional.json"
    build_receipt = json.loads(build_receipt_path.read_text())
    device_receipt = json.loads(device_receipt_path.read_text())
    if (build_receipt["model"] != "olmo3_7b"
            or build_receipt["layers"] != 32
            or build_receipt["stages"] != 2123
            or build_receipt["generation"] != "greedy"
            or build_receipt["input_token_ids"] != {"prefill.token_ids": [1, 2, 3]}
            or device_receipt["status"] != "passed"
            or device_receipt["tohost"] != 0):
        raise ValueError("unexpected compiled OLMo 3 fixture")
    elf_path = build_dir / "kernel.radiance.elf"
    source_path = build_dir / "model_data.cpp"
    elf_hash = sha(elf_path)
    source_hash = sha(source_path)
    if (elf_hash != build_receipt["radiance_elf_sha256"]
            or elf_hash != device_receipt["device_elf_sha256"]
            or source_hash != device_receipt["device_source_files_sha256"]["model_data.cpp"]):
        raise ValueError("generated source or device ELF differs from passing run")

    pinned = model_specs()["olmo3_7b"]
    spec = reduced_spec(pinned)
    spec["num_hidden_layers"] = 32
    spec["layer_types"] = spec["layer_types"][:32]
    graph = build("olmo3_7b", prefill=3, decode_steps=2,
                  generation="greedy", specs={"olmo3_7b": spec})
    if len(graph.stages) != build_receipt["stages"]:
        raise ValueError("graph stage count differs from compiled program")
    if spec["layer_types"] != (["sliding_attention"] * 3
                                + ["full_attention"]) * 8:
        raise ValueError("local/full layer order differs from pinned config")
    source_config_path = HERE / "inputs/olmo3-7b.json"
    if sha(source_config_path) != build_receipt["source_config_sha256"]:
        raise ValueError("pinned source config differs from compiled program")
    source_config = json.loads(source_config_path.read_text())
    if source_config["use_cache"] is not False:
        raise ValueError("expected the pinned checkpoint's disabled cache default")
    config = transformers.Olmo3Config(
        vocab_size=spec["vocab_size"], hidden_size=spec["hidden_size"],
        intermediate_size=spec["intermediate_size"],
        num_hidden_layers=spec["num_hidden_layers"],
        num_attention_heads=spec["num_attention_heads"],
        num_key_value_heads=spec["num_key_value_heads"],
        head_dim=spec["head_dim"],
        max_position_embeddings=source_config["max_position_embeddings"],
        rope_parameters={**spec["rope_scaling"], "rope_theta": spec["rope_theta"]},
        rms_norm_eps=spec["rms_norm_eps"],
        sliding_window=spec["sliding_window"],
        layer_types=spec["layer_types"], pad_token_id=0,
        tie_word_embeddings=False, attention_bias=False, use_cache=True)
    config._attn_implementation = "eager"
    model = transformers.Olmo3ForCausalLM(config).eval()
    upstream_inverse = model.model.rotary_emb.inv_freq.detach().numpy()
    mapped_inverse = olmo_graph_reference.yarn_frequencies(
        spec["rope_scaling"], spec["rope_theta"], spec["head_dim"])
    inverse_error = float(np.max(np.abs(upstream_inverse - mapped_inverse)))
    if inverse_error > 1e-6:
        raise ValueError("YaRN inverse frequencies differ from upstream")

    state = model.state_dict()
    bound: dict[str, np.ndarray] = {}
    declarations: dict[str, str] = {}
    for stage in graph.stages:
        parameter = param_data(stage)
        if parameter is None:
            continue
        symbol, data = parameter
        source_name = parameter_key(stage["attrs"]["parameter"])
        if source_name in bound:
            if not np.array_equal(bound[source_name], data):
                raise ValueError(f"parameter changed across phases: {source_name}")
            continue
        bound[source_name] = data
        declarations[symbol] = array_decl(symbol, data, data.size).rstrip("\n")
        upstream_data = (data.T if stage["op"] == "linear"
                         and not stage["attrs"].get("weight_transposed") else data)
        if source_name not in state or state[source_name].shape != upstream_data.shape:
            raise ValueError(f"upstream parameter shape differs: {source_name}")
        state[source_name] = torch.from_numpy(np.ascontiguousarray(upstream_data))
    if set(bound) != set(state):
        raise ValueError(f"unbound upstream parameters: {sorted(set(state) - set(bound))}")
    model.load_state_dict(state, strict=True)

    seen = set()
    with source_path.open() as stream:
        for line in stream:
            if not line.startswith("alignas(64) __global float v_parameter_"):
                continue
            symbol = line.split("[", 1)[0].split()[-1]
            if symbol not in declarations or line.rstrip("\n") != declarations[symbol]:
                raise ValueError(f"generated parameter differs: {symbol}")
            if symbol in seen:
                raise ValueError(f"duplicate generated parameter: {symbol}")
            seen.add(symbol)
    if seen != set(declarations):
        raise ValueError(f"missing generated parameters: {sorted(set(declarations) - seen)}")

    mapped = olmo_graph_reference.execute(
        graph, {"prefill.token_ids": np.array([[1, 2, 3]], dtype=np.int32)})
    current_phase = [""]
    observed: dict[tuple[str, int], np.ndarray] = {}

    def capture(layer_index: int):
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            observed[(current_phase[0], layer_index)] = tensor.detach().numpy().copy()
        return hook

    handles = [layer.register_forward_hook(capture(i))
               for i, layer in enumerate(model.model.layers)]
    logits = []
    layers = []
    selected = []
    cache = None
    token_ids = np.array([[1, 2, 3]], dtype=np.int64)
    try:
        with torch.no_grad():
            for phase, output_name in zip(("prefill", "decode0", "decode1"),
                                          graph.outputs):
                current_phase[0] = phase
                result = model(input_ids=torch.from_numpy(token_ids),
                               past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                logits.append({"phase": phase, **compare(
                    mapped[output_name], result.logits.detach().numpy())})
                for layer_index in range(32):
                    layer_name = f"{phase}.layer{layer_index:02d}.ffn_residual"
                    layers.append({"phase": phase, "layer": layer_index,
                                   **compare(mapped[layer_name],
                                             observed[(phase, layer_index)])})
                if phase != "decode1":
                    next_phase = "decode0" if phase == "prefill" else "decode1"
                    token = torch.argmax(result.logits[:, -1, :], dim=-1)[:, None]
                    upstream_id = int(token.item())
                    mapped_id = int(mapped[f"{next_phase}.token_ids"][0, 0])
                    selected.append({"phase": next_phase,
                                     "upstream": upstream_id, "mapped": mapped_id})
                    token_ids = token.numpy()
    finally:
        for handle in handles:
            handle.remove()
    failures = sum(row["out_of_tolerance"] for row in logits + layers)
    if failures or any(row["upstream"] != row["mapped"] for row in selected):
        raise ValueError(f"upstream comparison failed: {failures} tensor values")
    if [row["mapped"] for row in selected] != build_receipt["reference_generated_token_ids"]:
        raise ValueError("generated tokens differ from device build")
    return {
        "schema": "radiance.olmo3-device-fixture-upstream.v1",
        "status": "passed",
        "scope": "32 original-order local/full layers at reduced FP32 widths, same compiled synthetic weights",
        "comparison": "upstream Transformers CPU vs stitched NumPy; the separate Cyclotron receipt checks device stages against the same NumPy reference",
        "build_receipt_sha256": sha(build_receipt_path),
        "device_receipt_sha256": sha(device_receipt_path),
        "source_config_sha256": sha(source_config_path),
        "device_elf_sha256": elf_hash, "model_data_sha256": source_hash,
        "parameters_checked": len(declarations),
        "layer_outputs_checked": len(layers),
        "layer_outputs": layers, "logits": logits,
        "max_layer_output_abs_error": max(row["max_abs_error"] for row in layers),
        "yarn_inverse_frequency_max_abs_error": inverse_error,
        "selected_tokens": selected,
        "source_sha256": sha(Path(__file__)),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "checkpoint_weights": False, "device_rerun": False,
        "upstream_cache_override": True,
        "window_boundary_exercised": False,
        "rtl_execution": False, "performance_measurement": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = check(args.build_dir)
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
