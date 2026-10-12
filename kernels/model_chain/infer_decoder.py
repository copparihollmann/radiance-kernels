#!/usr/bin/env python3
"""Compile and run a pinned decoder checkpoint with greedy token handoff.

The prompt is fixed when the Radiance ELF is built. The native run executes the
same generated stages on the CPU and reports their selected token IDs. Optional
Cyclotron execution validates those stages on the functional ISA simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compile_decoder import build_device, generate
from run_checkpoint_native import run as run_native
from stitch import model_specs


MODELS = ("tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma_2_2b_it")


def tokenizer_at(path: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("text prompts require transformers; use --token-ids") from error
    if not (path / "tokenizer.json").is_file() and not (path / "tokenizer.model").is_file():
        raise FileNotFoundError(
            f"no tokenizer in {path}; supply --tokenizer-dir or --token-ids")
    return AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--weight-image", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--token-ids", type=int, nargs="+")
    parser.add_argument("--tokenizer-dir", type=Path,
                        help="defaults to the checkpoint directory")
    parser.add_argument("--new-tokens", type=int, default=2)
    parser.add_argument("--stages-per-object", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--reuse", action="store_true",
                        help="run an existing ELF only if its model and prompt match")
    args = parser.parse_args()
    if args.new_tokens <= 0 or args.stages_per_object <= 0 or args.jobs <= 0 or not 1 <= args.threads <= 32:
        parser.error("new tokens, stages per object, and jobs must be positive; threads must be 1..32")
    tokenizer_path = args.tokenizer_dir or args.checkpoint_dir
    tokenizer = tokenizer_at(tokenizer_path) if args.prompt is not None else None
    token_ids = (tokenizer.encode(args.prompt, add_special_tokens=True)
                 if tokenizer is not None else args.token_ids)
    if not token_ids:
        parser.error("the prompt must contain at least one token")
    layers = model_specs()[args.model]["num_hidden_layers"]
    target = args.out_root / args.model
    if args.reuse:
        manifest = json.loads((target / "manifest.json").read_text())
        if (manifest["model"] != args.model or manifest["layers"] != layers or
                manifest["generation"] != "greedy" or
                manifest["prefill_tokens"] != len(token_ids) or
                manifest["decode_steps"] != args.new_tokens or
                manifest["input_token_ids"] != {"prefill.token_ids": token_ids} or
                Path(manifest["weight_image_manifest"]).resolve() != args.weight_image.resolve()):
            raise ValueError("existing ELF does not match the requested model, prompt, or image")
    else:
        target = generate(args.model, layers, len(token_ids), args.new_tokens,
                          "greedy", args.out_root,
                          device_check_all_stages=True,
                          stages_per_object=args.stages_per_object,
                          checkpoint_dir=args.checkpoint_dir,
                          weight_image=args.weight_image,
                          token_ids=token_ids)
        build_device(target)
    native = run_native(args.model, args.out_root, args.threads, args.jobs,
                        target / "native-check" / "native-result.json")
    generated = native["generated_token_ids"]
    if generated is None or len(generated) != args.new_tokens:
        raise RuntimeError("native run did not report every generated token")
    if tokenizer is None and ((tokenizer_path / "tokenizer.json").is_file() or
                              (tokenizer_path / "tokenizer.model").is_file()):
        tokenizer = tokenizer_at(tokenizer_path)
    result = {
        "model": args.model,
        "prompt": args.prompt,
        "input_token_ids": token_ids,
        "generated_token_ids": generated,
        "continuation_text": tokenizer.decode(generated, skip_special_tokens=False)
        if tokenizer is not None else None,
        "full_text": tokenizer.decode(token_ids + generated, skip_special_tokens=False)
        if tokenizer is not None else None,
        "checkpoint_weight_format": native["checkpoint_weight_format"],
        "native_status": native["status"],
        "native_result": str(target / "native-check" / "native-result.json"),
        "radiance_elf": str(target / "kernel.radiance.elf"),
        "device_execution": False,
        "performance_measurement": False,
    }
    destination = target / "inference-result.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
