#!/usr/bin/env python3
"""Fail-closed audit for full-size xRAGE parallel-Scatter artifacts and plan."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


BASE_COMMIT = "8bf256a5406e749a6342522e398b468d56c1b5db"
SUITE_SHA256 = "deeecaffe7392f8f0aa881887ab804690e3c93248493a3effbc0e592f496569f"
PATTERN_SHA256 = "283191aeb3d9750b4fb811851a7576c60110966adcbcda3e6fd917e5880545c3"
TRANSFER_COUNT = 6_664_304
WRITTEN_DESTINATIONS = 1_191_878
REPEATED_TRANSFERS = 5_472_426
OUTPUT_ELEMENTS = 2_051_101
ARTIFACT_NAME = "xrage-parallel-winner-post20k--scatter-parallel--{sms}sm"

TARGETS = {
    "firesim-1sm-dram": (1, "dram", "1SM_DRAM"),
    "firesim-2sm-dram": (2, "dram", "2SM_DRAM"),
    "firesim-2sm-dram-baseline": (2, "dram", "2SM_DRAM_baseline"),
    "firesim-2sm-hbm": (2, "hbm", "2SM_HBM"),
    "firesim-4sm-dram": (4, "dram", "4SM_DRAM"),
}


class AuditError(ValueError):
    """A required invariant was absent or contradicted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def attach_device(result_path: Path, device_path: Path) -> None:
    result = load_json(result_path)
    device = device_path.resolve()
    require(device.is_file(), f"missing device ELF: {device}")
    result["device_elf"] = str(device)
    result["device_elf_sha256"] = sha256(device)
    save_json(result_path, result)


def expected_run_argv() -> list[str]:
    return [
        "timeout", "--signal=TERM", "--kill-after=30", "1800", "bash",
        "{runner}", "-b", "{bitstream_config}", "--bdf", "{bdf}",
        "--fastloadmem", "{elf}",
    ]


def validate_target(target: dict[str, Any], workload: dict[str, Any]) -> str:
    target_id = target.get("id")
    require(target_id in TARGETS, f"unexpected target: {target_id}")
    sms, memory, config = TARGETS[target_id]
    require(target.get("sm_count") == sms, f"{target_id}: SM mismatch")
    require(target.get("memory") == memory, f"{target_id}: memory mismatch")
    require(target.get("loader") == "direct-hashed", f"{target_id}: wrong loader")
    require(target.get("bdf") == "1a:00.0", f"{target_id}: wrong FPGA BDF")
    require(target.get("bitstream_config") == config,
            f"{target_id}: wrong bitstream config")
    argv = target.get("run_argv")
    require(argv == expected_run_argv(), f"{target_id}: unsafe run argv")
    require("--no-program" not in argv, f"{target_id}: reuse is forbidden")
    require(workload.get("name") == ARTIFACT_NAME.format(sms=sms),
            f"{target_id}: wrong workload")
    require(workload.get("correctness") == "admissible-winner-set-checked",
            f"{target_id}: weak correctness contract")
    return target_id


def pattern_receipt(result: dict[str, Any]) -> dict[str, Any]:
    receipts = [
        item for item in result.get("pattern_inputs", [])
        if item.get("field") == "pattern"
    ]
    require(len(receipts) == 1, "expected one pattern receipt")
    return receipts[0]


def validate_result(result_path: Path, sms: int, loader: str) -> dict[str, Any]:
    result = load_json(result_path)
    expected = {
        "status": "built",
        "suite_sha256": SUITE_SHA256,
        "case": 1,
        "sm_count": sms,
        "loader": loader,
        "gpu_address_hash": loader == "direct-hashed",
        "kind": "scatter",
        "pattern_length": TRANSFER_COUNT,
        "count": 1,
        "wrap": 1,
        "source_elements": TRANSFER_COUNT,
        "output_elements": OUTPUT_ELEMENTS,
        "logical_payload_bytes": TRANSFER_COUNT * 16,
        "expected_nonzero_outputs": WRITTEN_DESTINATIONS,
        "admissible_winner_check": True,
        "parallel_scatter_output": "schedule-dependent-non-atomic",
        "destination_overlap": True,
        "collision_policy": "parallel",
        "correctness": "admissible-winner-set-checked",
        "host_full_readback": True,
        "muon_warps_per_core": 8,
        "robust_end_drain": True,
        "robust_spin_until_reset": loader == "direct-hashed",
        "robust_serial_fence": True,
        "robust_token_fence": False,
        "robust_drain_passes": 3,
        "robust_post_spin": 20_000,
        "diagnostic_readback": True,
        "count_overridden": False,
    }
    for key, value in expected.items():
        require(result.get(key) == value,
                f"{result_path}: {key}={result.get(key)!r}, expected {value!r}")
    require(isinstance(result.get("kernel_revision"), str),
            f"{result_path}: missing source revision")
    require(result.get("address_plan", {}).get("schedule") == "task",
            f"{result_path}: device task partition changed")
    receipt = pattern_receipt(result)
    require(receipt.get("sha256") == PATTERN_SHA256,
            f"{result_path}: pinned pattern hash changed")
    require(int(receipt.get("length", -1)) == TRANSFER_COUNT,
            f"{result_path}: pinned transfer count changed")
    elf = result_path.parent / "kernel.soc.elf"
    require(elf.is_file(), f"missing fused ELF: {elf}")
    device = Path(result.get("device_elf", ""))
    require(device.is_file(), f"missing recorded device ELF: {device}")
    require(sha256(device) == result.get("device_elf_sha256"),
            f"device ELF hash mismatch: {device}")
    return {
        "sms": sms,
        "loader": loader,
        "result": str(result_path.resolve()),
        "result_sha256": sha256(result_path),
        "elf": str(elf.resolve()),
        "elf_bytes": elf.stat().st_size,
        "elf_sha256": sha256(elf),
        "device_elf": str(device.resolve()),
        "device_elf_bytes": device.stat().st_size,
        "device_elf_sha256": sha256(device),
        "kernel_source_sha256": result.get("kernel_source_sha256"),
        "kernel_revision": result.get("kernel_revision"),
    }


def rv32_segments(elf: Path) -> dict[str, dict[str, Any]]:
    listing = subprocess.check_output(["readelf", "-SW", str(elf)], text=True)
    data = elf.read_bytes()
    expression = re.compile(
        r"\.rv32\.seg(\d+)\s+PROGBITS\s+[0-9a-fA-F]+\s+"
        r"([0-9a-fA-F]+)\s+([0-9a-fA-F]+)"
    )
    segments: dict[str, dict[str, Any]] = {}
    for match in expression.finditer(listing):
        index, offset, size = match.groups()
        start, length = int(offset, 16), int(size, 16)
        blob = data[start:start + length]
        require(len(blob) == length, f"truncated RV32 segment {index}: {elf}")
        segments[index] = {"bytes": length, "sha256": hashlib.sha256(blob).hexdigest()}
    require(bool(segments), f"no embedded RV32 segments: {elf}")
    return segments


def isa_receipt(device: Path, readelf: Path) -> dict[str, Any]:
    attributes = subprocess.check_output(
        [str(readelf), "-A", str(device)], text=True, stderr=subprocess.STDOUT,
    )
    match = re.search(r'Tag_RISCV_arch:\s+"([^"]+)"', attributes)
    require(match is not None, f"missing RISC-V ISA attributes: {device}")
    architecture = match.group(1)
    require(architecture.startswith("rv32i") and "_m" in architecture,
            f"device ISA is not scalar RV32I+M: {architecture}")
    tokens = architecture.split("_")
    require(not architecture.startswith("rv32v") and "v" not in tokens,
            f"vector ISA present: {architecture}")
    lower = attributes.lower()
    require("gemmini" not in lower and "mx" not in lower,
            f"Gemmini/MX attribute present: {device}")
    return {
        "path": str(device.resolve()),
        "sha256": sha256(device),
        "riscv_arch": architecture,
        "scalar_no_vpu": True,
        "gemmini": False,
        "mx": False,
    }


def validate_manifest(path: Path, loader: str) -> dict[str, Any]:
    manifest = load_json(path)
    require(manifest.get("no_vpu") is True, f"{path}: no_vpu not asserted")
    require(manifest.get("hardware_rebuilt") is False,
            f"{path}: hardware_rebuilt must be false")
    require(len(manifest.get("artifact_base_revisions", [])) == 1,
            f"{path}: artifacts do not share one source revision")
    rows = manifest.get("workloads", [])
    require(len(rows) == 3, f"{path}: expected exactly three artifacts")
    require({int(row["sm_count"]) for row in rows} == {1, 2, 4},
            f"{path}: SM matrix is not 1/2/4")
    require(all(row.get("loader") == loader for row in rows),
            f"{path}: loader mismatch")
    for row in rows:
        expected_name = ARTIFACT_NAME.format(sms=int(row["sm_count"]))
        require(row.get("name") == expected_name,
                f"{path}: unexpected artifact name {row.get('name')}")
        elf = Path(row["elf"])
        require(elf.is_file() and sha256(elf) == row.get("elf_sha256"),
                f"{path}: artifact hash mismatch: {elf}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "loader": loader,
        "rows": len(rows),
    }


def validate_model(model_root: Path) -> list[dict[str, Any]]:
    results = sorted(model_root.glob("jobs/*/rep-0/result.json"))
    require(len(results) == 3, f"expected three Cyclotron results, found {len(results)}")
    evidence = []
    for result_path in results:
        result = load_json(result_path)
        sms = int(result.get("sm_count", 0))
        require(sms in (1, 2, 4), f"invalid model SM count: {result_path}")
        require(result.get("status") == "passed", f"model did not pass: {result_path}")
        require(result.get("correctness_checks") == result.get("expected_correctness_checks") == 1,
                f"incomplete Cyclotron readback: {result_path}")
        log = result_path.with_name("stdout.log")
        text = log.read_text(errors="replace")
        require(f"nonzero_u64={WRITTEN_DESTINATIONS}" in text,
                f"exact coverage diagnostic absent: {log}")
        require("*** FAILED ***" not in text, f"failure marker in model log: {log}")
        evidence.append({
            "sms": sms,
            "result": str(result_path.resolve()),
            "result_sha256": sha256(result_path),
            "stdout": str(log.resolve()),
            "stdout_sha256": sha256(log),
            "simulation_cycles": result.get("simulation_cycles"),
            "kernel_cycles": result.get("kernel_cycles"),
            "coverage_destinations": WRITTEN_DESTINATIONS,
        })
    return sorted(evidence, key=lambda row: row["sms"])


def validate_exact_model(model_root: Path) -> list[dict[str, Any]]:
    results = sorted(model_root.glob("*/result.json"))
    require(len(results) == 3, f"expected three exact Cyclotron results, found {len(results)}")
    evidence = []
    expected_line = (
        "XRAGE_EXACT_WINNER_CHECK transfers=6664304 outputs=2051101 "
        "intended=1191878 admissible=1191878 missing=0 invalid_source=0 "
        "wrong_destination=0 unexpected_nonzero=0 illegal=0 PASS"
    )
    for result_path in results:
        result = load_json(result_path)
        sms = int(result.get("sm_count", 0))
        require(sms in (1, 2, 4), f"invalid exact-model SM count: {result_path}")
        require(result.get("status") == "passed", f"exact model did not pass: {result_path}")
        require(result.get("correctness") == "admissible-winner-set-checked",
                f"exact checker contract missing: {result_path}")
        require(result.get("exact_winner_diagnostic") == expected_line,
                f"exact checker diagnostic changed: {result_path}")
        dump = Path(result["output_dump"])
        log = Path(result["checker_log"])
        require(dump.is_file() and dump.stat().st_size == OUTPUT_ELEMENTS * 8,
                f"invalid complete output dump: {dump}")
        require(sha256(dump) == result.get("output_dump_sha256"),
                f"output dump hash mismatch: {dump}")
        require(expected_line in log.read_text(errors="replace"),
                f"exact diagnostic absent: {log}")
        evidence.append({
            "sms": sms,
            "result": str(result_path.resolve()),
            "result_sha256": sha256(result_path),
            "simulation_cycles": result.get("simulation_cycles"),
            "output_dump": str(dump.resolve()),
            "output_dump_bytes": dump.stat().st_size,
            "output_dump_sha256": sha256(dump),
            "checker_log": str(log.resolve()),
            "checker_log_sha256": sha256(log),
            "exact_winner_diagnostic": expected_line,
        })
    require({row["sms"] for row in evidence} == {1, 2, 4},
            "exact Cyclotron SM matrix is incomplete")
    return sorted(evidence, key=lambda row: row["sms"])


def validate_plan(plan_path: Path, asset_root: Path) -> dict[str, Any]:
    plan = load_json(plan_path)
    require(plan.get("schema") == "radiance-fpga-1a-campaign-plan-v1",
            "wrong FPGA plan schema")
    jobs = plan.get("jobs", [])
    require(len(jobs) == 5, f"expected exactly five FPGA jobs, found {len(jobs)}")
    seen: set[str] = set()
    for job in jobs:
        target = job.get("target", {})
        target_id = validate_target(target, job.get("workload", {}))
        require(target_id not in seen, f"duplicate target: {target_id}")
        seen.add(target_id)
    require(seen == set(TARGETS), f"missing FPGA targets: {set(TARGETS) - seen}")

    receipts = plan.get("asset_receipts", {})
    checked = []
    for label, receipt in sorted(receipts.items()):
        path = asset_root / receipt["path"]
        require(path.is_file(), f"missing copied Nicolas asset: {path}")
        require(path.stat().st_size == int(receipt["bytes"]),
                f"asset size mismatch: {path}")
        require(sha256(path) == receipt["sha256"], f"asset hash mismatch: {path}")
        checked.append({"label": label, "path": str(path.resolve()),
                        "bytes": path.stat().st_size, "sha256": receipt["sha256"]})
    require(len(checked) == 25, f"expected 25 five-image asset receipts, found {len(checked)}")
    return {
        "path": str(plan_path.resolve()),
        "sha256": sha256(plan_path),
        "jobs": len(jobs),
        "targets": sorted(seen),
        "bdf": "1a:00.0",
        "fresh_programming": True,
        "run_argv": expected_run_argv(),
        "fpga_commands_executed": 0,
        "asset_root": str(asset_root.resolve()),
        "asset_receipts": checked,
    }


def validate_source(repo: Path) -> dict[str, Any]:
    subprocess.run(["git", "-C", str(repo), "cat-file", "-e", BASE_COMMIT + "^{commit}"],
                   check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                   text=True).strip()
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                    BASE_COMMIT, head], check=True)
    protected = ["kernel.cpp", "spatter_ops.hpp", "plan.py", "host.cpp", "run.py",
                 "parallel_scatter_winner.hpp"]
    for name in protected:
        path = repo / "kernels/spatter" / name
        base_blob = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{BASE_COMMIT}:kernels/spatter/{name}"]
        )
        require(path.read_bytes() == base_blob, f"protected source differs from {BASE_COMMIT}: {name}")
    return {"base_commit": BASE_COMMIT,
            "audit_branch_descends_from_base": True,
            "protected_files_identical_to_base": protected}


def validate_artifact_revision(repo: Path, revision: str) -> None:
    subprocess.run(["git", "-C", str(repo), "cat-file", "-e", revision + "^{commit}"],
                   check=True)
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                    BASE_COMMIT, revision], check=True)
    for name in ("kernel.cpp", "spatter_ops.hpp", "plan.py", "host.cpp", "run.py",
                 "parallel_scatter_winner.hpp"):
        base_blob = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{BASE_COMMIT}:kernels/spatter/{name}"])
        artifact_blob = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{revision}:kernels/spatter/{name}"])
        require(artifact_blob == base_blob,
                f"artifact revision changes protected source: {revision} {name}")


def collect(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    root = args.root.resolve()
    source = validate_source(repo)
    suite = args.suite.resolve()
    require(sha256(suite) == SUITE_SHA256, "suite receipt changed")
    pattern = suite.parent / "xrage-scatter.u32le"
    require(sha256(pattern) == PATTERN_SHA256, "pinned xRAGE pattern changed")
    require(pattern.stat().st_size == TRANSFER_COUNT * 4, "pattern byte count changed")

    artifacts: list[dict[str, Any]] = []
    for loader in ("logical", "direct-hashed"):
        for sms in (1, 2, 4):
            result_path = root / "build" / loader / ARTIFACT_NAME.format(sms=sms) / "result.json"
            artifacts.append(validate_result(result_path, sms, loader))
    artifact_revisions = {row["kernel_revision"] for row in artifacts}
    require(len(artifact_revisions) == 1, "artifacts do not share one source revision")
    artifact_revision = artifact_revisions.pop()
    validate_artifact_revision(repo, artifact_revision)
    source["artifact_revision"] = artifact_revision
    source["artifact_revision_protected_files_identical_to_base"] = True

    shadow_pairs = []
    for sms in (1, 2, 4):
        logical = next(row for row in artifacts if row["sms"] == sms and row["loader"] == "logical")
        direct = next(row for row in artifacts if row["sms"] == sms and row["loader"] == "direct-hashed")
        require(logical["kernel_source_sha256"] == direct["kernel_source_sha256"],
                f"{sms}SM logical/direct source closure differs")
        # The FPGA-only spin-until-reset macro deliberately changes the device
        # tail, so byte-identical device ELFs are neither expected nor claimed.
        # Both sides were already checked above for the same transfer list,
        # task partition, occupancy, drains, fence, and post-spin interval.
        shadow_pairs.append({
            "sms": sms,
            "kernel_source_sha256": logical["kernel_source_sha256"],
            "intentional_differences": [
                "logical-vs-direct address encoding",
                "FPGA-only spin-until-reset disabled in Cyclotron shadow",
            ],
            "byte_identical_device_elf_claimed": False,
        })

    isa = []
    for row in artifacts:
        if row["loader"] == "direct-hashed":
            isa.append(isa_receipt(Path(row["device_elf"]), args.readelf.resolve()))

    manifests = [
        validate_manifest(root / "build/logical-manifest.json", "logical"),
        validate_manifest(root / "build/direct-hashed-manifest.json", "direct-hashed"),
    ]
    model = validate_model(root / "cyclotron-functional")
    exact_model = validate_exact_model(root / "cyclotron-exact")
    plan = validate_plan(root / "fpga-plan/plan.json", args.asset_root.resolve())

    native = load_json(root / "tests/native-test.json")
    python_tests = load_json(root / "tests/python-tests.json")
    require(native.get("status") == "passed" and native.get("transfers") == TRANSFER_COUNT,
            "native full-pattern test did not pass")
    require(python_tests.get("status") == "passed" and python_tests.get("tests") == 26,
            "Python focused tests did not pass")

    evidence = {
        "schema": "radiance-xrage-parallel-winner-fullsize-audit-v1",
        "status": "passed",
        "source": source,
        "fixture": {
            "suite": str(suite), "suite_sha256": SUITE_SHA256,
            "pattern": str(pattern), "pattern_sha256": PATTERN_SHA256,
            "transfers": TRANSFER_COUNT,
            "written_destinations": WRITTEN_DESTINATIONS,
            "repeated_transfers": REPEATED_TRANSFERS,
        },
        "contract": {
            "collision_policy": "parallel",
            "final_output": "schedule-dependent-non-atomic",
            "checker": "admissible-winner-set-checked",
            "device_task_partition": "task",
            "robust_completion": {
                "end_drain": True, "serial_fence": True, "drain_passes": 3,
                "post_spin_cycles": 20_000,
                "direct_hashed_spin_until_reset": True,
            },
            "cyclotron_shadow_limitation": (
                "Cyclotron has no direct-loader address-hash model; logical shadows disable "
                "only address pre-hashing and FPGA-only spin-until-reset"
            ),
        },
        "tests": {"python": python_tests, "native": native},
        "artifacts": artifacts,
        "cyclotron_shadow_pairs": shadow_pairs,
        "isa": isa,
        "manifests": manifests,
        "cyclotron_functional_coverage": model,
        "cyclotron_exact_winner": exact_model,
        "fpga_plan": plan,
        "toolchain": load_json(root / "toolchain.json"),
        "resource": load_json(root / "resource.json"),
    }
    save_json(args.out.resolve(), evidence)
    print(json.dumps({"status": "passed", "evidence": str(args.out.resolve()),
                      "evidence_sha256": sha256(args.out.resolve())}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    attach = subparsers.add_parser("attach-device")
    attach.add_argument("--result", type=Path, required=True)
    attach.add_argument("--device", type=Path, required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--repo", type=Path, required=True)
    collect_parser.add_argument("--root", type=Path, required=True)
    collect_parser.add_argument("--suite", type=Path, required=True)
    collect_parser.add_argument("--asset-root", type=Path, required=True)
    collect_parser.add_argument("--readelf", type=Path, required=True)
    collect_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "attach-device":
        attach_device(args.result, args.device)
    else:
        collect(args)


if __name__ == "__main__":
    main()
