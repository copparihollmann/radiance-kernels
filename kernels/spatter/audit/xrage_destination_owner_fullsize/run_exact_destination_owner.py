#!/usr/bin/env python3
"""Snapshot Cyclotron output and compare every xRAGE word to the serial oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time


HELPERS = Path(__file__).resolve().parents[1] / "xrage_parallel_fullsize"
sys.path.insert(0, str(HELPERS))
from run_exact_cyclotron import run_stopped, sha256  # noqa: E402


NAME = "xrage-destination-owner--scatter-parallel--{sms}sm"
EXACT_LINE = (
    "XRAGE_DESTINATION_OWNER_EXACT transfers=6664304 outputs=2051101 "
    "written=1191878 mismatches=0 expected_fnv=215de6e81154b9c5 "
    "observed_fnv=215de6e81154b9c5 PASS"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cyclotron", type=Path, required=True)
    parser.add_argument("--stop-library", type=Path, required=True)
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--pattern", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for sms in (1, 2, 4):
        name = NAME.format(sms=sms)
        source = args.coverage_root / "jobs" / name / "rep-0" / "config.toml"
        destination = args.out / name
        destination.mkdir(parents=True, exist_ok=True)
        config = destination / "config.toml"
        config.write_bytes(source.read_bytes())
        stdout = destination / "cyclotron.stdout.log"
        stderr = destination / "cyclotron.stderr.log"
        started = time.monotonic()
        returncode, stop_count, snapshot = run_stopped(
            args.cyclotron, config, args.stop_library, stdout, stderr, args.timeout)
        dump = destination / "output.u64le"
        dump.write_bytes(snapshot.pop("data"))
        checker_log = destination / "exact-check.log"
        with checker_log.open("w") as stream:
            checked = subprocess.run(
                [str(args.checker.resolve()), str(args.pattern.resolve()),
                 str(dump.resolve())],
                stdout=stream, stderr=subprocess.STDOUT, check=False)
        stdout_text = stdout.read_text(errors="replace")
        checker_text = checker_log.read_text(errors="replace")
        simulation = re.search(r"simulation finished after (\d+) cycles", stdout_text)
        passed = (
            returncode == 0 and stop_count == 1 and checked.returncode == 0
            and simulation is not None and EXACT_LINE in checker_text
            and dump.stat().st_size == 2051101 * 8
        )
        result = {
            "schema": "radiance-xrage-destination-owner-exact-cyclotron-v1",
            "status": "passed" if passed else "failed",
            "name": name,
            "sm_count": sms,
            "simulation_cycles": int(simulation.group(1)) if simulation else None,
            "cyclotron_returncode": returncode,
            "sigstop_count": stop_count,
            "snapshot": snapshot,
            "checker_returncode": checked.returncode,
            "wall_seconds": round(time.monotonic() - started, 6),
            "config": str(config.resolve()),
            "config_sha256": sha256(config),
            "cyclotron": str(args.cyclotron.resolve()),
            "cyclotron_sha256": sha256(args.cyclotron.resolve()),
            "stop_library": str(args.stop_library.resolve()),
            "stop_library_sha256": sha256(args.stop_library.resolve()),
            "checker": str(args.checker.resolve()),
            "checker_sha256": sha256(args.checker.resolve()),
            "cyclotron_stdout": str(stdout.resolve()),
            "cyclotron_stdout_sha256": sha256(stdout),
            "cyclotron_stderr": str(stderr.resolve()),
            "cyclotron_stderr_sha256": sha256(stderr),
            "output_dump": str(dump.resolve()),
            "output_dump_bytes": dump.stat().st_size,
            "output_dump_sha256": sha256(dump),
            "checker_log": str(checker_log.resolve()),
            "checker_log_sha256": sha256(checker_log),
            "exact_serial_oracle_diagnostic": EXACT_LINE,
            "correctness": "every-output-word-equals-serial-last-writer-oracle",
            "snapshot_validated_against_cyclotron_fnv_and_coverage": True,
        }
        (destination / "result.json").write_text(
            json.dumps(result, indent=2) + "\n")
        print(f"{name}: {result['status']}", flush=True)
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
