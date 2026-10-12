#!/usr/bin/env python3
"""Run fused Spatter ELFs on Cyclotron's timing model with GPU readback checks."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

KERNEL = Path(__file__).resolve().parents[1]
SOURCE = KERNEL / "runs"
TOOLS = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def guard_address(elf: Path, readelf: Path) -> int:
    listing = subprocess.check_output([str(readelf), "-SW", str(elf)], text=True)
    match = re.search(r"\.rv32\.seg5\s+PROGBITS\s+([0-9a-fA-F]+)", listing)
    if not match:
        raise ValueError(f"no fused RV32 BSS segment in {elf}")
    addr = int(match.group(1), 16) - 0x100000000
    if not 0 <= addr < 0x100000000:
        raise ValueError(f"invalid GPU guard address {addr:#x}")
    return addr


def save(path: Path, data: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def run(name: str, timeout: int, functional: bool, source_root: Path,
        output_root: Path, cyclotron: Path, checker: Path, config: Path,
        readelf: Path) -> int:
    source = source_root / name
    out = output_root / name
    out.mkdir(parents=True, exist_ok=True)
    elf = out / "kernel.soc.elf"
    if not elf.exists():
        shutil.copy2(source / "kernel.soc.elf", elf)
    elf_hash = digest(source / "kernel.soc.elf")
    if digest(elf) != elf_hash:
        raise ValueError(f"ELF copy changed: {name}")
    result = out / "result.json"
    if result.exists():
        data = json.loads(result.read_text())
        if data.get("status") in ("passed", "exploratory"):
            print(f"{name}: already {data['status']}", flush=True)
            return 0
    else:
        data = json.loads((source / "result.json").read_text())
    data.update({
        "simulator": "Cyclotron functional model" if functional else "Cyclotron timing model",
        "simulator_path": str(checker),
        "checker_source_sha256": digest(cyclotron / "src/bin/spatter_check.rs"),
        "timing_config_sha256": digest(config),
        "elf_sha256": elf_hash,
        "gpu_guard_before_addr": f"0x{guard_address(elf, readelf):08x}",
        "correctness": ("guards-and-nonzero-exploratory"
                        if data["destination_overlap"] and
                        data.get("collision_policy", "parallel") == "parallel"
                        else "digest-checked"),
        "status": "running",
    })
    data.pop("gpu_cycles", None)
    data.pop("error", None)
    save(result, data)
    env = os.environ.copy()
    env["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
    env["CYCLOTRON_PERF_LOG_DIR"] = str(out / "performance_logs")
    argv = [str(checker), str(config), str(elf), data["gpu_guard_before_addr"],
            str(data["output_elements"]), data["expected_digest"] or "none"]
    if functional:
        argv.append("--functional")
    started = time.monotonic()
    try:
        with (out / "cyclotron.log").open("w") as log:
            proc = subprocess.run(argv, cwd=cyclotron, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout)
        log_text = (out / "cyclotron.log").read_text(errors="replace")
        cycles = re.search(r"simulation finished after (\d+) cycles", log_text)
        check = re.search(r"SPATTER_CHECK digest=([0-9a-f]{16}) expected=(\w+) guards_intact=(\w+) nonzero_words=(\d+)", log_text)
        if functional:
            data["model_steps"] = int(cycles.group(1)) if cycles else None
        else:
            data["gpu_cycles"] = int(cycles.group(1)) if cycles else None
        data["output_digest"] = check.group(1) if check else None
        data["simulation_wall_seconds"] = round(time.monotonic() - started, 3)
        if proc.returncode or not cycles or not check or check.group(3) != "true" or int(check.group(4)) == 0:
            data["status"] = "failed"
            data["error"] = f"model exit {proc.returncode}; cycles={bool(cycles)}; readback={bool(check)}"
        else:
            data["status"] = ("exploratory" if data["destination_overlap"] and
                              data.get("collision_policy", "parallel") == "parallel"
                              else "passed")
    except subprocess.TimeoutExpired:
        data["status"] = "timeout"
        data["error"] = f"model exceeded {timeout} wall seconds"
    save(result, data)
    metric = f"model_steps={data.get('model_steps')}" if functional else f"gpu_cycles={data.get('gpu_cycles')}"
    print(f"{name}: {data['status']}, {metric}", flush=True)
    return 0 if data["status"] in ("passed", "exploratory") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--functional", action="store_true",
                        help="check outputs quickly without timing simulation")
    parser.add_argument("--cyclotron", required=True, type=Path,
                        help="Radiance Cyclotron checkout with spatter_check built")
    parser.add_argument("--config", type=Path,
                        help="Cyclotron config in its checkout (defaults to spatter-config.toml)")
    parser.add_argument("--readelf", type=Path, default=Path("readelf"))
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--output-root", type=Path,
                        help="defaults to separate timing and functional result roots")
    args = parser.parse_args()
    if args.jobs <= 0 or args.timeout <= 0:
        parser.error("--jobs and --timeout must be positive")
    source_root = args.source_root.resolve()
    output_root = (args.output_root.resolve() if args.output_root else
                   KERNEL / "runs" / ("cyclotron-functional" if args.functional else
                                        "cyclotron"))
    cyclotron = args.cyclotron.resolve()
    checker = cyclotron / "target/release/spatter_check"
    config = args.config.resolve() if args.config else cyclotron / "spatter-config.toml"
    if not checker.is_file() or not config.is_file():
        parser.error("--cyclotron needs target/release/spatter_check and --config must exist")
    if digest(cyclotron / "src/bin/spatter_check.rs") != digest(TOOLS / "spatter_check.rs"):
        parser.error("Cyclotron spatter_check.rs differs from this repo; copy tools/spatter_check.rs and rebuild")
    if digest(config) != digest(TOOLS / "spatter-config.toml"):
        parser.error("Cyclotron config differs from this repo; copy tools/spatter-config.toml to its checkout")
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run, name, args.timeout, args.functional,
                                   source_root, output_root, cyclotron, checker,
                                   config, args.readelf) for name in args.names]
        raise SystemExit(max(future.result() for future in as_completed(futures)))
