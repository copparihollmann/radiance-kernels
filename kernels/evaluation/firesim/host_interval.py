#!/usr/bin/env python3
"""Archive a passing U250 host release-to-completion timing experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil

from capture import HWDB_SHA256, sha256, snapshot
from splice_device import device_image


PASS = re.compile(r"\*\*\* PASSED \*\*\* after (\d+) cycles")
FAIL = re.compile(r"\*\*\* FAILED \*\*\*")
EXIT = re.compile(r'COMMAND_EXIT_CODE="(\d+)"')
INTERVAL = re.compile(r"(?m)^HOST_RELEASE_TO_DONE_CYCLES=(\d+)\s*$")


def parse_uart(uart: str) -> tuple[int, int]:
    """Return whole-program and RV64 release-to-completion cycles on a pass."""
    passes = PASS.findall(uart)
    intervals = INTERVAL.findall(uart)
    exits = EXIT.findall(uart)
    if (len(passes) != 1 or len(intervals) != 1 or exits != ["0"] or
            FAIL.search(uart) or "Simulation complete." not in uart):
        raise ValueError("UART lacks one passing guest, one timing interval, or clean driver exit")
    interval = int(intervals[0])
    if interval <= 0:
        raise ValueError("release-to-completion interval must be positive")
    return int(passes[0]), interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--untimed-elf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs-root", type=Path, default=Path("/scratch/firesim_queue/jobs"))
    parser.add_argument("--plan", type=Path,
                        default=Path(__file__).resolve().parent / "host-timing/plan.csv")
    args = parser.parse_args()

    source = args.elf.resolve()
    control = args.untimed_elf.resolve()
    with args.plan.open(newline="") as stream:
        planned = {int(row["job_id"]): row for row in csv.DictReader(stream)}
    row = planned[args.job]
    if (Path(row["timed_elf"]).resolve() != source or
            Path(row["untimed_elf"]).resolve() != control or
            row["timed_elf_sha256"] != sha256(source) or
            row["untimed_elf_sha256"] != sha256(control) or
            row["host_check"] != "full-digest-and-guards" or
            row["expected_uart"] != "pass"):
        raise ValueError("ELFs or expected outcome differ from pinned plan")
    build_path = source.parent / "result.json"
    build = json.loads(build_path.read_text())
    if build.get("host_timing_requested") is not True or build.get("host_full_readback") is not True:
        raise ValueError("ELF build did not request host timing and complete readback")
    if build.get("expected_digest") != row["expected_digest"]:
        raise ValueError("ELF expected digest differs from pinned plan")
    if device_image(source) != device_image(control):
        raise ValueError("timed and untimed RV32 device images differ")

    job_dir = args.jobs_root / str(args.job)
    request_path = job_dir / "runworkload-full.json"
    request = json.loads(request_path.read_text())
    if (request.get("job_id") != args.job or
            Path(request.get("stage_from", "")).resolve() != source or
            request.get("hwdb_config_artifact_sha256") != HWDB_SHA256):
        raise ValueError("queue request, source ELF, or pinned HWDB differs")
    # The deploy/workloads copy is shared by sequential jobs with one workload
    # name. Verify this job's own retained copy, not the mutable deploy file.
    job_elf = (job_dir / "simulation/sim_slot_0/rsyncdir" /
               f"{request['workload']}0-{request['bootbinary']}")
    if not job_elf.is_file() or sha256(job_elf) != sha256(source):
        raise ValueError("this job's FireSim ELF is missing or differs from build")
    uart_path = job_dir / "simulation/sim_slot_0/uartlog"
    whole_program, interval = parse_uart(uart_path.read_text(errors="replace"))
    stdout = (job_dir / "stdout.log").read_text(errors="replace")
    if "Flashing FPGA Slot: 0" not in stdout:
        raise ValueError("queue stdout does not show U250 flash")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    snapshot(job_dir, out / "raw")
    shutil.copy2(source, out / "kernel.soc.elf")
    shutil.copy2(control, out / "untimed-kernel.soc.elf")
    shutil.copy2(build_path, out / "build-result.json")
    shutil.copy2(source.parent / "build.log", out / "build.log")
    result = {
        "job_id": args.job,
        "guest_check": "pass-full-output-digest-and-guards",
        "host_release_to_done_cycles": interval,
        "firesim_target_cycles_whole_program": whole_program,
        "cycle_domain": "RV64 host rdcycle; GPU clock relation uncalibrated",
        "interval_includes": "reset-release MMIO, Muon work, completion polling",
        "interval_excludes": "later output readback and digest, earlier boot and initialization",
        "timed_elf_sha256": sha256(source),
        "job_elf_sha256": sha256(job_elf),
        "untimed_elf_sha256": sha256(control),
        "rv32_device_image_equal": True,
        "build_manifest_sha256": sha256(build_path),
        "queue_request_sha256": sha256(request_path),
        "uart_sha256": sha256(uart_path),
        "hwdb_sha256": HWDB_SHA256,
        "raw_artifacts": {p.relative_to(out).as_posix(): sha256(p)
                          for p in sorted((out / "raw").rglob("*")) if p.is_file()},
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
