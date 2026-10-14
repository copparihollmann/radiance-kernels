#!/usr/bin/env python3
"""Check cache calls in the exact pinned Nemotron modeling source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from profile_nemotron_state import SOURCE_CODE, pinned_config


SOURCE_URL = SOURCE_CODE.replace("/blob/", "/resolve/")
SOURCE_SHA256 = "ea982af0b805f181573f919ecb001d5bbc0153459923cf4b2f1ccae194e415a4"


def method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name)


def audit(raw: bytes) -> dict:
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"pinned modeling source changed: {digest}")
    config, source = pinned_config()
    program = raw.decode("utf-8")
    tree = ast.parse(program)
    forward = method(tree, "NemotronHBlock", "forward")
    branch = next(node for node in ast.walk(forward)
                  if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "self.block_type == 'attention'")
    calls = [node for statement in branch.body for node in ast.walk(statement)
             if isinstance(node, ast.Call)
             and ast.unparse(node.func) == "self.mixer"]
    if len(calls) != 1:
        raise ValueError("expected one attention mixer call")
    call = calls[0]
    keywords = [argument.arg for argument in call.keywords]
    attention_forward = method(tree, "NemotronHAttention", "forward")
    attention_args = [argument.arg for argument in attention_forward.args.args]
    cache_init = method(tree, "HybridMambaAttentionDynamicCache", "__init__")
    cache_seq = method(tree, "HybridMambaAttentionDynamicCache", "get_seq_length")
    init_text = ast.get_source_segment(program, cache_init)
    seq_text = ast.get_source_segment(program, cache_seq)
    if ("self.transformer_layers.append(i)" not in init_text
            or "self.transformer_layers[0]" not in seq_text
            or "self.key_cache[layer_idx].shape[-2]" not in seq_text):
        raise ValueError("expected pinned hybrid-cache sequence logic")
    pattern = config["hybrid_override_pattern"]
    first_non_mamba = next(i for i, symbol in enumerate(pattern) if symbol != "M")
    first_attention = next(i for i, symbol in enumerate(pattern) if symbol == "*")
    return {
        "schema": "radiance.nemotron-source-audit.v1",
        "status": "static_source_contract_only",
        "model": source["model"],
        "revision": source["revision"],
        "modeling_url": SOURCE_URL,
        "modeling_sha256": digest,
        "config_sha256": source["config_sha256"],
        "attention_block_mixer_call_line": call.lineno,
        "attention_block_mixer_call_keywords": keywords,
        "attention_mixer_accepts_past_key_value": "past_key_value" in attention_args,
        "attention_block_passes_cache": "past_key_value" in keywords,
        "attention_block_passes_mask": "attention_mask" in keywords,
        "first_non_mamba_layer": first_non_mamba,
        "first_attention_layer": first_attention,
        "sequence_length_cache_selector_includes_mlp_layers": first_non_mamba < first_attention,
        "conclusion": (
            "The pinned attention block call omits past_key_value and attention_mask. "
            "The cache's default sequence selector can choose an MLP-only layer "
            "before the first attention layer. This identifies source-level "
            "handoff gaps; it does not execute the checkpoint."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        help="optional local copy of the pinned modeling file")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.source:
        raw = args.source.read_bytes()
    else:
        with urlopen(SOURCE_URL, timeout=30) as response:
            raw = response.read()
    output = json.dumps(audit(raw), indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
