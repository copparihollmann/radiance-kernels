#!/usr/bin/env python3
"""Count SmolVLA scalar work and run a bounded Cyclotron diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time

from run_smolvla_functional import sha256
from split_decoder_weights import image_segments, verify_image, verify_image_placement


STAGE = re.compile(r"^void stage_v_(.*?)\(")
LINEAR = re.compile(r",\s*(\d+),\s*(\d+),\s*(\d+),\s*tid,\s*tpb\);")
LAUNCH = re.compile(r"mu_schedule\(stage_v_[^,]+,\s*nullptr,\s*(\d+)\);")


def scalar_work(target: Path, expected_stages: int) -> dict:
    kernel = (target / "kernel.cpp").read_text()
    warp_match = re.search(r"__mu_num_warps\s*=\s*(\d+)", kernel)
    if warp_match is None:
        raise ValueError("Missing warp count in generated kernel")
    launches = [int(value) for value in LAUNCH.findall(kernel)]
    if len(launches) != expected_stages:
        raise ValueError("Generated launch count differs from graph stage count")

    sources = sorted(target.glob("stage_chunk_*.cpp"))
    if not sources:
        raise ValueError("No generated stage sources found")
    source_digest = hashlib.sha256()
    classes = {name: {"calls": 0, "multiply_accumulates": 0}
               for name in ("vision", "expert", "other")}
    for source in sources:
        source_digest.update(source.name.encode() + b"\0" + source.read_bytes())
        stage = None
        for line in source.read_text().splitlines():
            match = STAGE.match(line)
            if match:
                stage = match.group(1)
            if "model_chain::linear(" not in line:
                continue
            dimensions = LINEAR.search(line)
            if stage is None or dimensions is None:
                raise ValueError(f"Cannot identify linear stage in {source.name}")
            m, k, n = map(int, dimensions.groups())
            family = ("vision" if "vision_layer" in stage else
                      "expert" if "expert" in stage else "other")
            classes[family]["calls"] += 1
            classes[family]["multiply_accumulates"] += m * k * n
    return {
        "scheduled_stages": len(launches),
        "scheduled_warps_per_stage": sorted(set(launches)),
        "declared_warps": int(warp_match.group(1)),
        "scalar_linear_calls": sum(item["calls"] for item in classes.values()),
        "scalar_linear_multiply_accumulates": sum(
            item["multiply_accumulates"] for item in classes.values()),
        "by_family": classes,
        "stage_sources_sha256": source_digest.hexdigest(),
    }


def run_probe(target: Path, cyclotron_root: Path, work_dir: Path,
              cycle_cap: int, wall_limit: int) -> dict:
    manifest = json.loads((target / "manifest.json").read_text())
    elf = target / "kernel.radiance.elf"
    if sha256(elf) != manifest["radiance_elf_sha256"]:
        raise ValueError("ELF differs from its build manifest")
    weights, weight_paths = verify_image(Path(manifest["weight_image_manifest"]))
    inputs, input_paths = verify_image(Path(manifest["input_image_manifest"]))
    if (weights["image_sha256"] != manifest["weight_image_sha256"] or
            inputs["image_sha256"] != manifest["input_image_sha256"]):
        raise ValueError("Image differs from its build manifest")
    verify_image_placement(elf, [weights, inputs])

    work_dir.mkdir(parents=True, exist_ok=True)
    simulator = cyclotron_root / "target/release/cyclotron"
    source = (cyclotron_root / "config.toml").read_text()
    source, matches = re.subn(r"(?m)^timeout\s*=\s*\d+",
                              f"timeout = {cycle_cap}", source, count=1)
    if matches != 1:
        raise ValueError("Cyclotron config has no unique cycle limit")
    source = source.replace('"config/timing/',
                            f'"{cyclotron_root}/config/timing/')
    config = work_dir / f"smolvla-{cycle_cap}-cycles.toml"
    config.write_text(source)
    segments = image_segments(weights) + image_segments(inputs)
    paths = weight_paths + input_paths
    env = os.environ.copy()
    env["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
    env["CYCLOTRON_WEIGHTS"] = ",".join(
        f"0x{segment['gpu_base_address']:x}:{path.resolve()}"
        for segment, path in zip(segments, paths))

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    try:
        process = subprocess.run(
            [str(simulator), str(config), "--binary-path", str(elf),
             "--gen-trace", "false"], cwd=cyclotron_root, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=wall_limit)
        output, exit_code = process.stdout, process.returncode
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        exit_code = 124
    wall_seconds = time.monotonic() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    log = work_dir / f"smolvla-{cycle_cap}-cycles.log"
    log.write_text(output)
    loaded = all(
        f"preloaded {segment['image_size_bytes']} bytes of weights "
        f"@0x{segment['gpu_base_address']:x}" in output
        for segment in segments)
    cap_reached = (exit_code == 1 and output.rstrip().endswith("Error: 0")
                   and loaded and "simulation finished after" not in output)
    return {
        "scope": "bounded_cyclotron_diagnostic_not_model_completion",
        "cycle_cap": cycle_cap,
        "status": "cycle_cap_reached" if cap_reached else "incomplete_or_unexpected",
        "exit_code": exit_code,
        "all_images_preloaded": loaded,
        "wall_seconds_including_preload": wall_seconds,
        "user_cpu_seconds": after.ru_utime - before.ru_utime,
        "system_cpu_seconds": after.ru_stime - before.ru_stime,
        "elf_sha256": sha256(elf),
        "weight_image_sha256": weights["image_sha256"],
        "input_image_sha256": inputs["image_sha256"],
        "simulator_sha256": sha256(simulator),
        "config_sha256": sha256(config),
        "log_sha256": sha256(log),
        "full_model_output_checked": False,
        "performance_measurement": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--cyclotron-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cycle-cap", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cycle_cap <= 0 or args.timeout <= 0:
        parser.error("cycle cap and timeout must be positive")
    target = (args.generated_root / "smolvla_base").resolve()
    manifest = json.loads((target / "manifest.json").read_text())
    work = scalar_work(target, manifest["stages"])
    probe = run_probe(target, args.cyclotron_root.resolve(),
                      args.work_dir.resolve(), args.cycle_cap, args.timeout)
    record = {"model": "smolvla_base", "mapping": work, "probe": probe}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{work['scalar_linear_multiply_accumulates']} scalar linear MACs; "
          f"{probe['status']} after {probe['wall_seconds_including_preload']:.2f}s")
    if probe["status"] != "cycle_cap_reached":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
