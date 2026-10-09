#!/usr/bin/env python3
"""Run an already-built model-chain SoC ELF on the local VCS Radiance RTL."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from record_vcs import DEFAULT_SIM, HERE


MODELS = ("handoff", "tinyllama", "deepseek_r1_distill_qwen_1_5b",
          "gemma_2_2b_it")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--generated-root", type=Path, default=HERE / "generated")
    parser.add_argument("--out", type=Path, help="result JSON path")
    parser.add_argument("--sim", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-cycles", type=int, default=6_000_000)
    args = parser.parse_args()
    target = HERE if args.model == "handoff" else args.generated_root.resolve() / args.model
    elf = target / "kernel.soc.elf"
    manifest = target / "manifest.json"
    if not elf.is_file() or (args.model != "handoff" and not manifest.is_file()):
        parser.error(f"missing built ELF or manifest in {target}")
    if not args.sim.is_file():
        parser.error(f"missing VCS simulator: {args.sim}")
    if args.timeout <= 0 or args.max_cycles <= 0:
        parser.error("timeout and max cycles must be positive")
    log = (HERE / "generated/rtl/handoff-vcs.log" if args.model == "handoff"
           else target / "vcs.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(args.sim), "+permissive", "+gpu_finish_keeps_sim=1",
               f"+max-cycles={args.max_cycles}", f"+loadmem={elf}",
               "+permissive-off", str(elf)]
    env = os.environ.copy()
    cyclotron_lib = HERE.parents[2] / "generators/radiance/cyclotron/target/release"
    env["LD_LIBRARY_PATH"] = str(cyclotron_lib) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
    with log.open("w") as stream:
        try:
            process = subprocess.run(command, cwd=log.parent, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=args.timeout, check=False)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
    output = args.out or (HERE / "evaluation" /
                          ("rtl-handoff-result.json" if args.model == "handoff"
                           else f"rtl-{args.model}-result.json"))
    record = [sys.executable, str(HERE / "record_vcs.py"), "--log", str(log),
              "--elf", str(elf), "--sim", str(args.sim),
              "--out", str(output), "--run-exit-code", str(exit_code),
              "--timeout-seconds", str(args.timeout)]
    if args.model != "handoff":
        record.extend(["--manifest", str(manifest)])
    status = subprocess.run(record, check=False).returncode
    print(f"VCS exit={exit_code}; log={log}; result={output}")
    if exit_code or status:
        raise SystemExit(exit_code or status)


if __name__ == "__main__":
    main()
