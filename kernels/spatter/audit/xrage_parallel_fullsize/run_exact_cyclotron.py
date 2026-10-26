#!/usr/bin/env python3
"""Snapshot live Cyclotron GMEM and apply the shared exact winner classifier."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


NAME = "xrage-parallel-winner-post20k--scatter-parallel--{sms}sm"
EXACT_LINE = (
    "XRAGE_EXACT_WINNER_CHECK transfers=6664304 outputs=2051101 "
    "intended=1191878 admissible=1191878 missing=0 invalid_source=0 "
    "wrong_destination=0 unexpected_nonzero=0 illegal=0 PASS"
)
GMEM_LINE = re.compile(
    r"gmem check .*: PASS address=0x([0-9a-f]+) words=(\d+) "
    r"fnv64=([0-9a-f]{16}) nonzero_u64=(\d+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fnv1a_u32(data: bytes) -> int:
    words = array("I")
    words.frombytes(data)
    if sys.byteorder != "little":
        words.byteswap()
    digest = 0xCBF29CE484222325
    for word in words:
        digest = ((digest ^ word) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return digest


def nonzero_u64(data: bytes) -> int:
    words = array("Q")
    words.frombytes(data)
    if sys.byteorder != "little":
        words.byteswap()
    return sum(value != 0 for value in words)


def large_anonymous_mappings(pid: int) -> list[tuple[int, int, str]]:
    mappings = []
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        start_text, end_text = fields[0].split("-")
        start, end = int(start_text, 16), int(end_text, 16)
        perms = fields[1]
        pathname = fields[5] if len(fields) == 6 else ""
        if perms.startswith("rw-p") and not pathname and end - start >= (1 << 32):
            mappings.append((start, end, line))
    return mappings


def snapshot_gmem(pid: int, address: int, byte_count: int,
                  expected_fnv: int, expected_nonzero: int) -> tuple[bytes, dict]:
    mappings = large_anonymous_mappings(pid)
    if len(mappings) != 1:
        raise RuntimeError(f"expected one Cyclotron FlatMemory mapping, found {len(mappings)}")
    start, end, map_line = mappings[0]
    if end - start != (1 << 32) + 4096:
        raise RuntimeError(f"unexpected Cyclotron FlatMemory map size: {map_line}")
    # glibc's mmap-backed 4-GiB Vec starts after its 16-byte allocation header.
    # The mapping shape and offset are asserted, then both independent metrics
    # printed by Cyclotron are recomputed from the captured bytes.
    offset = 16
    begin = start + offset + address
    if begin + byte_count > end:
        raise RuntimeError("Cyclotron output lies outside FlatMemory")
    with Path(f"/proc/{pid}/mem").open("rb", buffering=0) as memory:
        memory.seek(begin)
        data = memory.read(byte_count)
    if len(data) != byte_count:
        raise RuntimeError("short read from Cyclotron FlatMemory")
    if fnv1a_u32(data) != expected_fnv:
        raise RuntimeError("live snapshot differs from Cyclotron full-output FNV")
    if nonzero_u64(data) != expected_nonzero:
        raise RuntimeError("live snapshot differs from Cyclotron coverage count")
    return data, {"map": map_line, "flat_memory_offset": offset,
                  "snapshot_address": begin}


def run_stopped(cyclotron: Path, config: Path, stop_library: Path,
                stdout: Path, stderr: Path, timeout: int) -> tuple[int, int, dict]:
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(stop_library.resolve())
    started = time.monotonic()
    stopped = 0
    snapshot: dict = {}
    with stdout.open("wb", buffering=0) as out_stream, stderr.open("wb", buffering=0) as err_stream:
        proc = subprocess.Popen([str(cyclotron.resolve()), str(config.resolve())],
                                stdout=out_stream, stderr=err_stream, env=env)
        while True:
            if time.monotonic() - started > timeout:
                proc.kill()
                os.waitpid(proc.pid, 0)
                raise TimeoutError(f"Cyclotron exceeded {timeout} seconds")
            waited, status = os.waitpid(proc.pid, os.WNOHANG | os.WUNTRACED | os.WCONTINUED)
            if waited == 0:
                time.sleep(0.02)
                continue
            if os.WIFSTOPPED(status):
                stopped += 1
                if stopped != 1 or os.WSTOPSIG(status) != signal.SIGSTOP:
                    os.kill(proc.pid, signal.SIGKILL)
                    os.waitpid(proc.pid, 0)
                    raise RuntimeError("unexpected Cyclotron stop")
                text = stdout.read_text(errors="replace")
                matches = GMEM_LINE.findall(text)
                if len(matches) != 1:
                    os.kill(proc.pid, signal.SIGKILL)
                    os.waitpid(proc.pid, 0)
                    raise RuntimeError(f"expected one complete GMEM PASS line, found {len(matches)}")
                address_text, words_text, fnv_text, nonzero_text = matches[0]
                address, words = int(address_text, 16), int(words_text)
                data, receipt = snapshot_gmem(proc.pid, address, words * 4,
                                              int(fnv_text, 16), int(nonzero_text))
                snapshot = {"data": data, "address": address, "words": words,
                            "fnv64": fnv_text, "nonzero_u64": int(nonzero_text), **receipt}
                os.kill(proc.pid, signal.SIGCONT)
                continue
            if os.WIFCONTINUED(status):
                continue
            if os.WIFEXITED(status) or os.WIFSIGNALED(status):
                return os.waitstatus_to_exitcode(status), stopped, snapshot


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
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        config = out / "config.toml"
        config.write_bytes(source.read_bytes())
        stdout = out / "cyclotron.stdout.log"
        stderr = out / "cyclotron.stderr.log"
        started = time.monotonic()
        returncode, stop_count, snapshot = run_stopped(
            args.cyclotron, config, args.stop_library, stdout, stderr, args.timeout)
        dump = out / "output.u64le"
        dump.write_bytes(snapshot.pop("data"))
        checker_log = out / "exact-check.log"
        with checker_log.open("w") as stream:
            check = subprocess.run([str(args.checker.resolve()),
                                    str(args.pattern.resolve()), str(dump.resolve())],
                                   stdout=stream, stderr=subprocess.STDOUT)
        stdout_text = stdout.read_text(errors="replace")
        check_text = checker_log.read_text(errors="replace")
        simulation = re.search(r"simulation finished after (\d+) cycles", stdout_text)
        passed = (returncode == 0 and stop_count == 1 and check.returncode == 0 and
                  simulation is not None and EXACT_LINE in check_text and
                  dump.stat().st_size == 2051101 * 8)
        result = {
            "schema": "radiance-xrage-exact-cyclotron-result-v1",
            "status": "passed" if passed else "failed",
            "name": name, "sm_count": sms,
            "simulation_cycles": int(simulation.group(1)) if simulation else None,
            "cyclotron_returncode": returncode, "sigstop_count": stop_count,
            "snapshot": snapshot,
            "checker_returncode": check.returncode,
            "wall_seconds": round(time.monotonic() - started, 6),
            "config": str(config.resolve()), "config_sha256": sha256(config),
            "cyclotron": str(args.cyclotron.resolve()),
            "cyclotron_sha256": sha256(args.cyclotron.resolve()),
            "stop_library": str(args.stop_library.resolve()),
            "stop_library_sha256": sha256(args.stop_library.resolve()),
            "checker": str(args.checker.resolve()),
            "checker_sha256": sha256(args.checker.resolve()),
            "cyclotron_stdout": str(stdout.resolve()), "cyclotron_stdout_sha256": sha256(stdout),
            "cyclotron_stderr": str(stderr.resolve()), "cyclotron_stderr_sha256": sha256(stderr),
            "output_dump": str(dump.resolve()), "output_dump_bytes": dump.stat().st_size,
            "output_dump_sha256": sha256(dump),
            "checker_log": str(checker_log.resolve()), "checker_log_sha256": sha256(checker_log),
            "exact_winner_diagnostic": EXACT_LINE,
            "correctness": "admissible-winner-set-checked",
            "snapshot_validated_against_cyclotron_fnv_and_coverage": True,
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"{name}: {result['status']}", flush=True)
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
