#!/usr/bin/env python3
"""Resume the built Spatter ELFs on the local latest-Radiance Verilator model."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time


KERNEL = Path(__file__).resolve().parents[1]
SOURCE = KERNEL / "runs"
OUTPUT = KERNEL / "runs" / "verilator"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path: Path, data: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def run(name: str, max_cycles: int, sim: Path, output_root: Path, source_root: Path) -> int:
    source = source_root / name
    out = output_root / name
    out.mkdir(parents=True, exist_ok=True)
    source_elf = source / "kernel.soc.elf"
    elf = out / "kernel.soc.elf"
    if not elf.exists():
        shutil.copy2(source_elf, elf)
    source_hash = digest(source_elf)
    if digest(elf) != source_hash:
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
        "simulator": (
            "Verilator 5.022 (16 threads)" if "simulator-mt16" in sim.name else
            "Verilator 5.022 (8 threads)" if "simulator-mt" in sim.name else
            "Verilator 5.022"
        ),
        "simulator_path": str(sim),
        "elf_sha256": source_hash,
        "status": "running",
        "cyclotron_trace_requested": False,
    })
    data.pop("gpu_cycles", None)
    data.pop("error", None)
    save(result, data)

    argv = [str(sim.resolve()), "+permissive", "+gpu_finish_keeps_sim=1",
            f"+max-cycles={max_cycles}", f"+trace-db={out / 'trace.sqlite'}",
            f"+loadmem={elf}", "+permissive-off", str(elf)]
    env = os.environ.copy()
    env["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
    started = time.monotonic()
    try:
        with (out / "verilator.log").open("w") as log:
            proc = subprocess.run(argv, cwd=out, stdout=log, stderr=subprocess.STDOUT, env=env)
        log_text = (out / "verilator.log").read_text(errors="replace")
        cycles = [int(v) for v in re.findall(r"\bCycles:\s*(\d+)", log_text)]
        data["gpu_cycles"] = max(cycles) if cycles else None
        data["simulation_wall_seconds"] = round(time.monotonic() - started, 3)
        if (proc.returncode or not cycles or "Verilog $finish" not in log_text
                or "*** FAILED ***" in log_text or "%Error" in log_text):
            data["status"] = "failed"
            data["error"] = (f"simulator exit {proc.returncode}; GPU reports {len(cycles)}; "
                             f"finish={('Verilog $finish' in log_text)}")
        else:
            data["status"] = "exploratory" if data["destination_overlap"] else "passed"
    except KeyboardInterrupt:
        data["status"] = "interrupted"
        data["error"] = "interrupted by signal"
    save(result, data)
    print(f"{name}: {data['status']}, gpu_cycles={data.get('gpu_cycles')}", flush=True)
    return 0 if data["status"] in ("passed", "exploratory") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="case directories under spatter-vcs-runs")
    parser.add_argument("--max-cycles", type=int, default=2_000_000_000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--sim", required=True, type=Path,
                        help="built RadianceSingleClusterConfig Verilator simulator")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    args = parser.parse_args()
    if not args.sim.is_file():
        parser.error(f"simulator has not been built: {args.sim}")
    if args.max_cycles <= 0:
        parser.error("--max-cycles must be positive")
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    sim = args.sim.resolve()
    output_root = args.output_root.resolve()
    source_root = args.source_root.resolve()
    # This large Verilated model constructs more than 8 MiB of stack state.
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run, name, args.max_cycles, sim, output_root, source_root)
                   for name in args.names]
        raise SystemExit(max(future.result() for future in as_completed(futures)))
