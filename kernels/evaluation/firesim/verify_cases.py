#!/usr/bin/env python3
"""Check an archived FireSim case series without contacting the queue."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

from capture import HWDB_SHA256

PASS = re.compile(r"\*\*\* PASSED \*\*\* after (\d+) cycles")
FAIL = re.compile(r"\*\*\* FAILED \*\*\* \(code = (\d+)\) after (\d+) cycles")
EXIT = re.compile(r'COMMAND_EXIT_CODE="(\d+)"')


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--golden-table", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    plan = rows(directory / "plan.csv")
    results = rows(directory / "results.csv")
    assert plan and len(plan) == len(results)
    assert len({r["job_id"] for r in plan}) == len(plan)
    indexed = {r["job_id"]: r for r in results}
    golden = {r["run"]: r for r in rows(args.golden_table)} if args.golden_table else None
    source_hashes = {r["source_hash"] for r in rows(
        Path(__file__).resolve().parent / "source-snapshot-manifest.csv")}
    verified = 0
    for expected in plan:
        job_id = expected["job_id"]
        row = indexed[job_id]
        assert all(row[key] == value for key, value in expected.items()), job_id
        if golden is not None:
            reference = golden[expected["case"]]
            assert reference["status"] == "passed" and reference["count"] == row["count"]
            assert reference["golden_digest"] == row["expected_digest"]
            assert reference["radiance_elf_sha256"] == row["elf_sha256"]
        if row["validation"] == "pending" and args.allow_pending:
            continue
        assert row["validation"] in ("passed", "expected-diagnostic-failure",
                                     "infrastructure-failed"), job_id
        raw = directory / "raw" / job_id
        request = json.loads((raw / "runworkload-full.json").read_text())
        assert request["job_id"] == int(job_id), job_id
        assert request["hwdb_config_artifact_sha256"] == HWDB_SHA256, job_id
        assert request["stage_from"].endswith(row["elf_path"]), job_id
        build = json.loads((raw / "build-result.json").read_text())
        assert build["kernel_source_sha256"] in source_hashes, job_id
        stdout = (raw / "stdout.log").read_text(errors="replace")
        if row["validation"] == "infrastructure-failed":
            assert row["queue_state"] == row["queue_phase"] == "FAILED", job_id
            assert row["observed_uart"] == "pending" and not (raw / "uartlog").exists()
            assert "infrasetup rc=1" in (raw / "stderr.log").read_text(errors="replace")
            continue
        assert row["queue_state"] == row["queue_phase"] == "DONE", job_id
        assert "Flashing FPGA Slot: 0" in stdout, job_id
        assert str(build["count"]) == row["count"], job_id
        assert str(build["original_count"]) == row["original_count"], job_id
        assert build["expected_digest"] == row["expected_digest"], job_id
        if row["host_check"] == "full-digest":
            assert build["host_full_readback"], job_id
        uart_file = raw / "uartlog"
        uart = uart_file.read_text(errors="replace")
        assert sha256(uart_file) == row["uart_sha256"], job_id
        assert "Simulation complete." in uart, job_id
        passing, failing, exit_match = PASS.search(uart), FAIL.search(uart), EXIT.search(uart)
        if row["expected_uart"] == "pass":
            assert passing and not failing and row["observed_uart"] == "pass", job_id
            assert row["target_cycles"] == passing.group(1), job_id
            assert row["driver_exit_code"] == exit_match.group(1) == "0", job_id
        else:
            assert failing and not passing and row["observed_uart"] == "fail", job_id
            assert row["target_cycles"] == failing.group(2), job_id
            assert row["guest_failure_code"] == failing.group(1), job_id
            assert row["driver_exit_code"] == exit_match.group(1) == "1", job_id
        comparison = raw / "sample-comparison.json"
        if comparison.is_file():
            diagnostic = json.loads(comparison.read_text())
            assert diagnostic["uart_sha256"] == row["uart_sha256"], job_id
            assert diagnostic["guard_status"] == "1", job_id
            assert diagnostic["matched_samples"] + len(diagnostic["mismatches"]) == 64
            assert all(item["expected"] == "0000000000000000"
                       for item in diagnostic["mismatches"]), job_id
        verified += 1
    artifacts = rows(directory / "artifacts.csv")
    actual_paths = {path.relative_to(directory).as_posix()
                    for path in (directory / "raw").rglob("*") if path.is_file()}
    assert {entry["path"] for entry in artifacts} == actual_paths
    for entry in artifacts:
        path = directory / entry["path"]
        assert path.stat().st_size == int(entry["bytes"])
        assert sha256(path) == entry["sha256"], path
    print(f"verified {verified}/{len(plan)} jobs and {len(artifacts)} raw artifacts")


if __name__ == "__main__":
    main()
