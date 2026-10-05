#!/usr/bin/env python3
"""Verify the committed FireSim log snapshot without queue access."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

from make_comparison import build_rows


HERE = Path(__file__).resolve().parent
HWDB_SHA256 = "d4015580a0f7d58c0cd2606832d9579981feef98f4c5c1a29d25f7ba4f3c6bc4"
PASS = re.compile(r"\*\*\* PASSED \*\*\* after (\d+) cycles")
FAIL = re.compile(r"\*\*\* FAILED \*\*\* \(code = (\d+)\) after (\d+) cycles")
DRIVER_EXIT = re.compile(r'COMMAND_EXIT_CODE="(\d+)"')


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    assert sha256(HERE / "hwdb-entry.yaml") == HWDB_SHA256
    plan = read_csv(HERE / "plan.csv")
    results = read_csv(HERE / "results.csv")
    assert len(plan) == len(results) == 23
    assert {p["job_id"] for p in plan} == {r["job_id"] for r in results}
    rows = {r["job_id"]: r for r in results}
    passed = 0
    for expected in plan:
        job_id = expected["job_id"]
        row = rows[job_id]
        assert all(row[k] == v for k, v in expected.items()), job_id
        if expected["group"] != "control" and expected["case"] != "xrage9-guard-diagnostic":
            build = json.loads((HERE / "raw/build-metadata" / f"{expected['case']}.json").read_text())
            samples = int(build["readback_samples"])
            host_check = (
                "guards-and-nonzero-exploratory" if build["correctness"] == "guards-and-nonzero-exploratory"
                else "sample-and-guards" if samples else "full-digest"
            )
            assert expected["host_check"] == host_check, job_id
            if build.get("elf_sha256"):
                assert build["elf_sha256"] == expected["elf_sha256"], job_id
        if row["validation"] == "pending" and args.allow_pending:
            continue
        assert row["validation"] in ("passed", "expected-control-failure", "workload-failed", "diagnostic-failed"), (job_id, row["validation"])
        assert row["queue_state"] == row["queue_phase"] == "DONE", job_id
        if row["validation"] in ("workload-failed", "diagnostic-failed"):
            assert expected["expected_uart"] == "pass" and row["observed_uart"] == "fail"
            assert (expected["group"] == "diagnostic") == (row["validation"] == "diagnostic-failed")
        raw = HERE / "raw" / job_id
        request = json.loads((raw / "runworkload-full.json").read_text())
        assert request["job_id"] == int(job_id)
        assert request["hwdb_config_artifact_sha256"] == HWDB_SHA256
        assert request["workload"] == "agustin-radiance-" + expected["case"].replace("smoke-gather", "spatter-smoke").replace("smoke-negative-control", "smoke-negative").replace("copy-negative-control", "copy-negative")
        stdout = (raw / "stdout.log").read_text(errors="replace")
        assert f"staged {request['stage_from']} -> " in stdout, job_id
        assert "Flashing FPGA Slot: 0" in stdout, job_id
        assert f"hwdb verify phase=before_runworkload sha256={HWDB_SHA256}" in stdout, job_id
        uart_path = raw / "uartlog"
        uart = uart_path.read_text(errors="replace")
        assert sha256(uart_path) == row["uart_sha256"], job_id
        assert "Simulation complete." in uart, job_id
        if row["observed_uart"] == "pass":
            match = PASS.search(uart)
            assert match and not FAIL.search(uart), job_id
            assert row["driver_exit_code"] == "0"
        else:
            match = FAIL.search(uart)
            assert match and not PASS.search(uart), job_id
            assert row["driver_exit_code"] == "1" and row["guest_failure_code"] == match.group(1)
        assert row["target_cycles"] == match.group(1 if row["observed_uart"] == "pass" else 2)
        assert DRIVER_EXIT.search(uart).group(1) == row["driver_exit_code"], job_id
        passed += 1
    comparison = read_csv(HERE / "comparison.csv")
    assert len(comparison) == 17
    assert comparison == build_rows(), "comparison.csv differs from the source results"
    assert {r["case"] for r in comparison} == {r["case"] for r in plan if r["group"] not in ("control", "diagnostic") and r["case"] != "smoke-gather"}
    for row in comparison:
        observed = rows[row["firesim_job_id"]]
        assert row["firesim_uart"] == observed["observed_uart"]
        assert row["firesim_target_cycles_whole_program"] == observed["target_cycles"]
        assert row["elf_sha256"] == observed["elf_sha256"]
    model_dir = HERE / "raw/model-diagnostic"
    diagnostic = json.loads((model_dir / "xrage9-rebuild-functional.json").read_text())
    rebuild = next(row for row in plan if row["case"] == "xrage9-ordered-rebuild")
    golden = next(row for row in read_csv(HERE.parent.parent / "spatter/evaluation/upstream-golden-results.csv")
                  if row["run"] == "xrage9-ordered")
    dependencies = read_csv(HERE.parent / "dependency-snapshot.csv")
    assert diagnostic["elf_sha256"] == rebuild["elf_sha256"]
    assert diagnostic["checker_sha256"] in {
        row["sha256"] for row in dependencies if row["kind"] == "simulator"
    }
    assert diagnostic["config_sha256"] == sha256(model_dir / "spatter-config.toml")
    for name, digest in diagnostic["timing_includes_sha256"].items():
        assert digest == sha256(model_dir / "config/timing" / name), name
    log = (model_dir / "xrage9-rebuild-functional.log").read_text()
    steps = re.search(r"simulation finished after (\d+) cycles", log)
    check = re.search(r"SPATTER_CHECK digest=([0-9a-f]{16}) expected=([0-9a-f]{16}) "
                      r"guards_intact=(\w+) nonzero_words=(\d+) output_elements=(\d+)", log)
    assert steps and check and diagnostic["status"] == "passed"
    assert int(steps.group(1)) == diagnostic["model_steps"]
    assert check.group(1) == check.group(2) == diagnostic["model_digest"] == golden["golden_digest"]
    assert check.group(3) == "true" and diagnostic["guards_intact"] is True
    assert int(check.group(4)) == diagnostic["nonzero_words"] > 0
    assert int(check.group(5)) == diagnostic["output_elements"] == 2051101
    artifacts = read_csv(HERE / "artifacts.csv")
    assert {r["path"] for r in artifacts} == {
        str(path.relative_to(HERE)) for path in (HERE / "raw").rglob("*") if path.is_file()
    }
    for artifact in artifacts:
        path = HERE / artifact["path"]
        assert path.stat().st_size == int(artifact["bytes"])
        assert sha256(path) == artifact["sha256"]
    failed = sum(r["validation"] == "workload-failed" for r in results)
    diagnostic_failed = sum(r["validation"] == "diagnostic-failed" for r in results)
    print(f"verified {passed} FireSim results ({failed} workload failures, "
          f"{diagnostic_failed} diagnostic failures), "
          f"{len(artifacts)} raw files, and {len(comparison)} comparison rows")


if __name__ == "__main__":
    main()
