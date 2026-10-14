#!/usr/bin/env python3
"""Compare one reduced Mamba prefill block with the revision-pinned CPU source.

The remote files are supplied explicitly and checked by SHA. The in-process
Mamba normalization shim replaces an unavailable Triton dependency on CPU;
its formula is checked against the graph at the normalization boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import inspect
import json
import linecache
from pathlib import Path
import resource
import sys
import types

import numpy as np
import torch
import transformers

import audit_nemotron_source
import nemotron_graph_reference as graph_reference
import reference
import stitch


HERE = Path(__file__).resolve().parent
CONFIG_SHA256 = "07fa66e5b3da7e6a71c1a263e3dd68da11c8afa9178b47c49510ba628746fcff"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_pinned(modeling_raw: bytes, config_raw: bytes):
    if digest(modeling_raw) != audit_nemotron_source.SOURCE_SHA256:
        raise ValueError("pinned modeling file changed")
    if digest(config_raw) != CONFIG_SHA256:
        raise ValueError("pinned configuration file changed")

    # This package only satisfies the pinned source's CPU import. The fast
    # Triton/CUDA kernels remain unavailable and are not represented here.
    for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton",
                 "mamba_ssm.ops.triton.layernorm_gated"):
        module = types.ModuleType(name)
        module.__path__ = []
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = module

    def rmsnorm_fn(*, x, weight, bias, z, eps, group_size, norm_before_gate):
        if bias is not None or norm_before_gate or z is None:
            raise ValueError("unsupported pinned normalization mode")
        gated = x * torch.nn.functional.silu(z)
        shape = gated.shape
        groups = gated.reshape(*shape[:-1], -1, group_size)
        normalized = groups * torch.rsqrt(
            groups.float().square().mean(-1, keepdim=True) + eps)
        return normalized.reshape(shape) * weight

    sys.modules["mamba_ssm.ops.triton.layernorm_gated"].rmsnorm_fn = rmsnorm_fn
    package = types.ModuleType("pinned_nemotron_probe")
    package.__path__ = []
    sys.modules[package.__name__] = package
    modules = []
    for suffix, raw in (("configuration_nemotron_h", config_raw),
                        ("modeling_nemotron_h", modeling_raw)):
        name = f"{package.__name__}.{suffix}"
        module = types.ModuleType(name)
        module.__package__ = package.__name__
        sys.modules[name] = module
        linecache.cache[name] = (len(raw), None, raw.decode().splitlines(True), name)
        exec(compile(raw, name, "exec"), module.__dict__)
        modules.append(module)
    return modules


def run(modeling_raw: bytes, config_raw: bytes, prefill: int) -> dict:
    if prefill < 2:
        raise ValueError("prefill must be at least two tokens")
    if transformers.__version__ != "5.3.0":
        raise ValueError("this comparison requires Transformers 5.3.0")
    torch.set_num_threads(1)
    torch.manual_seed(17)
    config_module, modeling = load_pinned(modeling_raw, config_raw)
    source_text = modeling_raw.decode()
    if ("B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)" not in source_text
            or "C = C.repeat(1, 1, self.num_heads // self.n_groups, 1)" not in source_text
            or "B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1])" not in source_text):
        raise ValueError("pinned prefill/decode group layouts changed")
    if modeling.is_fast_path_available:
        raise ValueError("comparison requires the pinned CPU fallback")
    pinned = stitch.model_specs()["nemotron3_nano_4b"]
    spec = graph_reference.control_spec(pinned, 1)
    config = config_module.NemotronHConfig(
        hybrid_override_pattern="M", num_hidden_layers=1,
        hidden_size=spec["hidden_size"], intermediate_size=spec["intermediate_size"],
        vocab_size=spec["vocab_size"], num_attention_heads=spec["num_attention_heads"],
        attention_head_dim=spec["head_dim"],
        num_key_value_heads=spec["num_key_value_heads"],
        mamba_num_heads=spec["mamba_num_heads"],
        mamba_head_dim=spec["mamba_head_dim"],
        mamba_n_groups=spec["n_groups"], ssm_state_size=spec["ssm_state_size"],
        mamba_d_conv=spec["conv_kernel"], mamba_chunk_size=spec["chunk_size"],
        mlp_hidden_act="relu2", use_mamba_kernels=False)
    mixer = modeling.NemotronHMamba2Mixer(config, layer_idx=0).eval()
    embedding = (np.random.default_rng(19).standard_normal(
        (spec["vocab_size"], spec["hidden_size"])).astype(np.float32) * 0.1)

    class Weights:
        def embedding_row(self, name, row, width):
            if name != "backbone.embeddings.weight" or width != spec["hidden_size"]:
                raise ValueError("unexpected embedding parameter")
            return embedding[row]

        def __call__(self, name, shape):
            if name.endswith(".norm.weight") or name == "backbone.norm_f.weight":
                value = np.ones(shape, dtype=np.float32)
            elif name == "lm_head.weight":
                value = reference.parameter(name, shape)
            else:
                component = name.split(".mixer.", 1)[1]
                target = mixer
                for part in component.split("."):
                    target = getattr(target, part)
                value = target.detach().numpy().astype(np.float32)
                if component.endswith(".weight") and value.ndim == 2:
                    value = value.T
            if value.shape != shape:
                raise ValueError(f"{name}: {value.shape} != {shape}")
            return value

    graph = stitch.build("nemotron3_nano_4b", prefill=prefill, decode_steps=0,
                         specs={"nemotron3_nano_4b": spec})
    ids = np.array([[1 + index % 63 for index in range(prefill)]],
                   dtype=np.int32)
    inputs = {"prefill.token_ids": ids}
    values = graph_reference.execute(graph, inputs, weights=Weights())
    observed = {}

    def output_hook(name):
        def hook(module, args, output):
            observed[name] = output.detach().numpy().copy()
        return hook

    for name in ("in_proj", "conv1d", "norm", "out_proj"):
        getattr(mixer, name).register_forward_hook(output_hook(name))

    def scan_hook(module, args):
        observed["scan"] = args[0].detach().numpy().copy()
        observed["gate"] = args[1].detach().numpy().copy()

    mixer.norm.register_forward_pre_hook(scan_hook)
    with torch.no_grad():
        mixer(torch.from_numpy(values["prefill.layer00.pre_norm"]))

    def difference(name, expected):
        actual = observed[name]
        if name == "conv1d":
            actual = torch.nn.functional.silu(
                torch.from_numpy(actual[..., :prefill]).transpose(1, 2)).numpy()
        return float(np.max(np.abs(actual - expected)))

    stages = {"in_proj": "in_proj", "conv1d": "depthwise_conv",
              "gate": "gate", "scan": "selective_scan",
              "norm": "gated_norm", "out_proj": "out_proj"}
    standard = {name: difference(name, values[f"prefill.layer00.{stage}"])
                for name, stage in stages.items()}

    # The pinned prefill fallback uses B.repeat(..., heads/groups, ...), which
    # maps group indices 0,1,0,1. The graph and pinned decode branch use
    # contiguous indices 0,0,1,1. Change only this mapping in the oracle.
    source = inspect.getsource(graph_reference._scan)
    original = "group = head // (heads // groups)"
    if source.count(original) != 1:
        raise ValueError("graph scan group mapping changed")
    try:
        exec(source.replace(original, "group = head % groups"),
             graph_reference.__dict__)
        alternating = graph_reference.execute(graph, inputs, weights=Weights())
    finally:
        exec(source, graph_reference.__dict__)
    matched = {name: difference(name, alternating[f"prefill.layer00.{stage}"])
               for name, stage in stages.items()}
    if max(matched.values()) > 2e-5 or standard["scan"] < 1e-3:
        raise ValueError("pinned prefill group-layout control failed")
    return {
        "schema": "radiance.nemotron-pinned-prefill.v1", "status": "passed",
        "scope": "one reduced Mamba2 prefill block, same synthetic FP32 weights",
        "prefill_tokens": prefill,
        "modeling_sha256": digest(modeling_raw),
        "configuration_sha256": digest(config_raw),
        "standard_graph_max_abs_by_stage": standard,
        "alternating_group_max_abs_by_stage": matched,
        "pinned_cpu_prefill_group_order": "alternating: 0,1,0,1",
        "graph_and_pinned_decode_group_order": "contiguous: 0,0,1,1",
        "adapter": "in-memory CPU rmsnorm_fn shim; no pinned source file edited",
        "source_sha256": digest(Path(__file__).read_bytes()),
        "graph_reference_sha256": digest(Path(graph_reference.__file__).read_bytes()),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "checkpoint_weights": False, "full_network_equivalence": False,
        "device_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--prefill", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(args.modeling.read_bytes(), args.configuration.read_bytes(),
                 args.prefill)
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
