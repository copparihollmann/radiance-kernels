#!/usr/bin/env python3
"""Run one input through a reusable SpectFormer Radiance ELF in Cyclotron."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from prepare import sha256
from reference import load_input, logits


def scan_cyclotron_log(path: Path, weight_bytes: int, weight_address: int,
                       input_bytes: int, input_address: int) -> dict:
    """Read completion evidence without retaining the simulator log in RAM."""
    weight_line = f"preloaded {weight_bytes} bytes of weights @0x{weight_address:x}"
    input_line = f"preloaded {input_bytes} bytes of weights @0x{input_address:x}"
    found = {"passed": 0, "weights": 0, "input": 0, "cycles": []}
    with path.open(errors="replace") as stream:
        for line in stream:
            found["passed"] += line.count("isa-test passed with tohost=0")
            found["weights"] += line.count(weight_line)
            found["input"] += line.count(input_line)
            found["cycles"].extend(int(value) for value in re.findall(
                r"simulation finished after (\d+) cycles", line))
    return found


def upstream_output(image: dict, pixels: np.ndarray, kind: str) -> np.ndarray:
    if kind == "logits":
        if image["format"] == "spectformer.hierarchical.fp32.v1":
            from hierarchical_prepare import reference_logits
            return reference_logits(image, pixels).reshape(-1)
        return logits(image, pixels).reshape(-1)

    import torch

    if kind == "vanilla_patch_embed":
        from reference import make_model
        model = make_model(image)
        module = model.patch_embed
    elif kind.startswith("vanilla_block_"):
        from reference import make_model
        model = make_model(image)
        index = int(kind.removeprefix("vanilla_block_"))
        if not 0 <= index < len(model.blocks):
            raise ValueError("block index is outside the upstream graph")
        module = model.blocks[index]
    elif kind == "hierarchical_stem":
        from hierarchical_prepare import make_model
        model = make_model(image)
        module = model.patch_embed1
    elif kind.startswith("hierarchical_block_"):
        from hierarchical_prepare import make_model
        model = make_model(image)
        parts = kind.removeprefix("hierarchical_block_").split("_")
        if len(parts) != 2 or not all(part.isdecimal() for part in parts):
            raise ValueError("invalid hierarchical block output kind")
        stage, index = map(int, parts)
        blocks = getattr(model, f"block{stage}", None)
        if blocks is None or not 0 <= index < len(blocks):
            raise ValueError("block index is outside the upstream graph")
        module = blocks[index]
    else:
        raise ValueError(f"unsupported output kind: {kind}")
    captured = []
    hook = module.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output))
    with torch.no_grad():
        model(torch.from_numpy(pixels.copy()))
    hook.remove()
    if len(captured) != 1:
        raise ValueError("upstream prefix did not execute exactly once")
    value = captured[0]
    if kind == "vanilla_patch_embed":
        value = value + model.pos_embed
    else:
        value = value[0]
    result = value.detach().cpu().numpy().astype("<f4").reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("upstream prefix produced nonfinite values")
    return result


def run(target: Path, input_path: Path, out: Path, simulator_root: Path,
        timeout: int, sim_cycles: int, check_reference: bool = True) -> dict:
    target = target.resolve()
    out.mkdir(parents=True, exist_ok=True)
    out = out.resolve()
    build = json.loads((target / "manifest.json").read_text())
    if not build.get("runtime_input") or not build.get("device_elf_built"):
        raise ValueError("build a reusable ELF with --runtime-input --build")
    elf = target / "kernel.radiance.elf"
    if sha256(elf) != build["radiance_elf_sha256"]:
        raise ValueError("ELF differs from build manifest")
    image_manifest = Path(build["source_image_manifest"])
    image = json.loads(image_manifest.read_text())
    weights = image_manifest.parent / "weights.bin"
    if (sha256(weights) != image["weights_sha256"] or
            sha256(weights) != build["weights_sha256"]):
        raise ValueError("weights differ from export")
    pixels = load_input(input_path, image["image_size"])
    seed_kind = build.get("seed_kind")
    if seed_kind is not None and not check_reference:
        raise ValueError("a block-seeded probe requires an upstream output check")
    output_kind = build.get("output_kind", "logits")
    output_count = build.get("output_count", image["num_classes"])
    output_offset = build.get("output_offset", build["logits_offset"])
    input_bytes = out / "input.bin"
    seed = upstream_output(image, pixels, seed_kind) if seed_kind else pixels
    if seed_kind and seed.size != (image["image_size"] // 16) ** 2 * image["embed_dim"]:
        raise ValueError("upstream block seed has the wrong tensor size")
    input_bytes.write_bytes(seed.astype("<f4", copy=False).tobytes())
    expected = None
    if check_reference:
        expected = upstream_output(image, pixels, output_kind)
        if expected.size != output_count:
            raise ValueError("upstream output size differs from compiled output")
        reference_name = ("reference-logits.bin" if output_kind == "logits"
                          else "reference-output.bin")
        (out / reference_name).write_bytes(expected.tobytes())

    simulator_root = simulator_root.resolve()
    simulator = simulator_root / "target/release/cyclotron"
    source = (simulator_root / "config.toml").read_text()
    config_text, count = re.subn(r"(?m)^timeout\s*=\s*\d+",
                                 f"timeout = {sim_cycles}", source, count=1)
    if count != 1:
        raise ValueError("Cyclotron config has no unique timeout")
    config_text = config_text.replace('"config/timing/',
                                      f'"{simulator_root}/config/timing/')
    config = out / "cyclotron-config.toml"
    config.write_text(config_text)
    toolchain = Path(os.environ.get(
        "RISCV_TOOLCHAIN_PATH", "/scratch/agustin/projects/chipyard/.conda-env/riscv-tools"))
    symbols = subprocess.check_output(
        [str(toolchain / "bin/riscv64-unknown-elf-nm"), "-n", str(elf)], text=True)
    arenas = re.findall(r"(?m)^([0-9a-fA-F]+) [A-Za-z] v_arena$", symbols)
    if len(arenas) != 1:
        raise ValueError("ELF must contain one v_arena")
    input_address = (int(arenas[0], 16) + build["seed_offset"]
                     if seed_kind else build["input_address"])
    dump = out / ("logits.bin" if output_kind == "logits" else "output.bin")
    dump.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
    environment["CYCLOTRON_WEIGHTS"] = ",".join((
        f"0x{image['base_address']:x}:{weights.resolve()}",
        f"0x{input_address:x}:{input_bytes.resolve()}",
    ))
    environment["CYCLOTRON_DUMP_GMEM"] = (
        f"0x{int(arenas[0], 16) + output_offset:x}:"
        f"{output_count * 4}:{dump}")
    log = out / "cyclotron.log"
    with log.open("wb") as log_stream:
        try:
            result = subprocess.run([str(simulator), str(config), "--binary-path",
                                     str(elf), "--gen-trace", "false"],
                                    cwd=simulator_root, env=environment,
                                    stdout=log_stream, stderr=subprocess.STDOUT,
                                    timeout=timeout)
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
    events = scan_cyclotron_log(log, weights.stat().st_size,
                                image["base_address"], input_bytes.stat().st_size,
                                input_address)
    complete = (exit_code == 0 and events["passed"] == 1 and
                events["weights"] == 1 and events["input"] == 1 and
                len(events["cycles"]) == 1 and dump.is_file() and
                dump.stat().st_size == output_count * 4)
    comparison = None
    if expected is not None and dump.is_file() and dump.stat().st_size == expected.nbytes:
        actual = np.fromfile(dump, dtype="<f4")
        finite = bool(np.isfinite(actual).all())
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        close = finite and bool(np.allclose(actual, expected,
                                           rtol=build["logit_rtol"],
                                           atol=build["logit_atol"]))
        comparison = {
            "all_finite": finite, "all_close": close,
            "max_abs_error": float(np.max(np.abs(delta))) if finite else None,
            "rms_error": float(np.sqrt(np.mean(delta * delta))) if finite else None,
            "top1_agrees": (bool(np.argmax(actual) == np.argmax(expected))
                            if finite and output_kind == "logits" else None),
            "logits_sha256": sha256(dump),
        }
    passed = complete and check_reference and comparison is not None and comparison["all_close"]
    status = ("passed" if passed else "completed_unverified" if complete and
              not check_reference else "failed")
    record = {
        "model": build["model"], "status": status,
        "output_kind": output_kind, "output_count": output_count,
        "failure_reason": None if status != "failed" else ("wall_clock_timeout" if exit_code == 124 else
            "logit_mismatch" if complete else "simulator_or_program_failure"),
        "variant": image.get("variant", "vanilla"),
        "variant_depth": image.get("depth"), "image_size": image["image_size"],
        "weight_provenance": image["weight_provenance"],
        "input_path": str(input_path.resolve()), "input_sha256": sha256(input_bytes),
        "source_input_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "seed_kind": seed_kind, "loaded_input_address": input_address,
        "checkpoint_sha256": image["checkpoint_sha256"],
        "weights_sha256": sha256(weights), "elf_sha256": sha256(elf),
        "simulator_sha256": sha256(simulator),
        "config_sha256": sha256(config), "log_sha256": sha256(log),
        "output_sha256": (sha256(dump) if dump.is_file() and
                          dump.stat().st_size == output_count * 4 else None),
        "output_comparison": comparison,
        "logit_comparison": comparison if output_kind == "logits" else None,
        "reference_checked": check_reference,
        "cycles_functional": (events["cycles"][0]
                              if len(events["cycles"]) == 1 else None),
        "performance_measurement": False, "rtl_execution": False,
        "exit_code": exit_code,
    }
    (out / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cyclotron-root", type=Path,
                        default=Path("/scratch/agustin/projects/chipyard/generators/radiance/cyclotron"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--sim-cycles", type=int, default=100_000_000)
    parser.add_argument("--no-reference", action="store_true",
                        help="run inference and dump logits without importing upstream PyTorch")
    args = parser.parse_args()
    result = run(args.target, args.input, args.out, args.cyclotron_root,
                 args.timeout, args.sim_cycles, not args.no_reference)
    print(json.dumps(result, indent=2))
    if result["status"] not in ("passed", "completed_unverified"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
