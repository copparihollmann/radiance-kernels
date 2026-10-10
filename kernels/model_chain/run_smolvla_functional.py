#!/usr/bin/env python3
"""Run a compiled SmolVLA ELF in Cyclotron with its pinned weight image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from split_decoder_weights import (image_segments, verify_image,
                                   verify_image_placement)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(generated_root: Path, simulator_root: Path, timeout: int,
        sim_cycles: int) -> dict:
    target = (generated_root / "smolvla_base").resolve()
    manifest = json.loads((target / "manifest.json").read_text())
    elf = target / "kernel.radiance.elf"
    if not manifest["device_elf_built"] or sha256(elf) != manifest[
            "radiance_elf_sha256"]:
        raise ValueError("SmolVLA ELF differs from its build manifest")
    image, image_paths = verify_image(Path(manifest["weight_image_manifest"]))
    if image["image_sha256"] != manifest["weight_image_sha256"]:
        raise ValueError("SmolVLA image differs from build manifest")
    input_image = None
    input_paths = []
    if manifest.get("input_image_manifest"):
        input_image, input_paths = verify_image(
            Path(manifest["input_image_manifest"]))
        if input_image["image_sha256"] != manifest["input_image_sha256"]:
            raise ValueError("SmolVLA input image differs from build manifest")
    verify_image_placement(elf, [image] + ([input_image] if input_image else []))
    simulator_root = simulator_root.resolve()
    simulator = simulator_root / "target/release/cyclotron"
    source = (simulator_root / "config.toml").read_text()
    modified, matches = re.subn(r"(?m)^timeout\s*=\s*\d+",
                                f"timeout = {sim_cycles}", source, count=1)
    if matches != 1:
        raise ValueError("Cyclotron config has no unique cycle limit")
    modified = modified.replace('"config/timing/',
                                f'"{simulator_root}/config/timing/')
    config = target / "cyclotron-config.toml"
    config.write_text(modified)
    env = os.environ.copy()
    env["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
    segments = image_segments(image)
    input_segments = image_segments(input_image) if input_image else []
    env["CYCLOTRON_WEIGHTS"] = ",".join(
        f"0x{item['gpu_base_address']:x}:{path.resolve()}"
        for item, path in zip(segments + input_segments,
                              image_paths + input_paths))
    try:
        result = subprocess.run([str(simulator), str(config), "--binary-path",
                                 str(elf), "--gen-trace", "false"],
                                cwd=simulator_root, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=timeout)
        output = result.stdout
        exit_code = result.returncode
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        exit_code = 124
    log = target / "functional.log"
    log.write_text(output)
    cycles = re.findall(r"simulation finished after (\d+) cycles", output)
    failed_tohost = re.search(r"isa-test failed with tohost=(\d+)", output)
    tohost = int(failed_tohost.group(1)) if failed_tohost else None
    passed = (exit_code == 0 and "isa-test passed with tohost=0" in output
              and len(cycles) == 1 and all(
                  f"preloaded {item['image_size_bytes']} bytes of weights "
                  f"@0x{item['gpu_base_address']:x}" in output
                  for item in segments + input_segments))
    record = {
        "model": "smolvla_base", "scope": manifest["scope"],
        "stage_count": manifest["stages"], "full_graph_stages": manifest["full_graph_stages"],
        "status": "passed" if passed else "failed",
        "failure_reason": (None if passed else "wall_clock_timeout" if exit_code == 124
                           else "device_output_mismatch_or_nonfinite"
                           if tohost is not None and tohost & 1
                           else "device_or_simulator_failure"),
        "cycles_functional": int(cycles[0]) if cycles else None,
        "tohost": 0 if passed else tohost,
        "failed_output_element_index": tohost >> 1 if tohost and tohost & 1 else None,
        "output_validation": manifest["output_validation"],
        "golden_output_comparison": (
            manifest["output_validation"] == "all_elements_vs_upstream_policy"),
        "golden_output_sha256": manifest.get("golden_output_sha256"),
        "input_image_sha256": input_image["image_sha256"] if input_image else None,
        "upstream_execution_equivalent": False,
        "rtl_execution": False, "performance_measurement": False,
        "process_exit_code": exit_code,
        "device_elf_sha256": sha256(elf),
        "weight_image_sha256": image["image_sha256"],
        "simulator_sha256": sha256(simulator),
        "config_sha256": sha256(config), "log_sha256": sha256(log),
    }
    (target / "functional-result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--cyclotron-root", type=Path,
                        default=ROOT.parent / "generators/radiance/cyclotron")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--sim-cycles", type=int, default=5_000_000_000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(args.generated_root, args.cyclotron_root, args.timeout,
                 args.sim_cycles)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"smolvla_base: {result['status']} ({result['cycles_functional']} functional cycles)")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
