#!/usr/bin/env python3
"""Record a fixed model FireSim run with its UART, ELF, and HWDB identity."""

import argparse
import hashlib
import json
from pathlib import Path
import re


PASS_STATUS = 0x600D0000
FIELDS = ("model_status", "gpu_start_cycle64", "gpu_end_cycle64",
          "kernel_cycles64")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_uart(log) -> dict:
    """Accept one complete host report while retaining only its field lines."""
    stream = log.splitlines() if isinstance(log, str) else log
    counts = {key: 0 for key in FIELDS}
    values = {}
    positions = {}
    markers = {"model_chain_hbm: launch cluster 0": 0, "PASS": 0,
               "FAIL": 0}
    for index, raw in enumerate(stream):
        line = raw.strip()
        if line in markers:
            markers[line] += 1
            positions.setdefault(line, index)
        for key in FIELDS:
            if line.startswith(key + "="):
                counts[key] += 1
                if counts[key] == 1:
                    values[key] = line[len(key) + 1:]
                    positions[key] = index
    problems = []
    if markers["model_chain_hbm: launch cluster 0"] != 1:
        problems.append("expected one model-chain launch marker")
    for key in FIELDS:
        if counts[key] != 1:
            problems.append(f"expected one {key} field")
    if markers["PASS"] != 1 or markers["FAIL"]:
        problems.append("expected one PASS and no FAIL")
    if not problems:
        sequence = [positions["model_chain_hbm: launch cluster 0"]]
        sequence.extend(positions[key] for key in FIELDS)
        sequence.append(positions["PASS"])
        if sequence != sorted(sequence):
            problems.append("host report fields are out of order")
    parsed = {}
    if not problems:
        try:
            status = values["model_status"]
            if not re.fullmatch(r"[0-9a-fA-F]{8}", status):
                raise ValueError("model_status must be eight hex digits")
            parsed["model_status"] = int(status, 16)
            for key in FIELDS[1:]:
                raw = values[key]
                if not re.fullmatch(r"[0-9]+", raw):
                    raise ValueError(f"{key} must be decimal")
                parsed[key] = int(raw)
                if parsed[key] >= 1 << 64:
                    raise ValueError(f"{key} exceeds 64 bits")
        except ValueError as error:
            problems.append(str(error))
    if not problems:
        if parsed["model_status"] != PASS_STATUS:
            problems.append("device model_status is not PASS")
        if parsed["gpu_end_cycle64"] <= parsed["gpu_start_cycle64"]:
            problems.append("device end stamp does not follow start")
        elif parsed["kernel_cycles64"] != (parsed["gpu_end_cycle64"] -
                                           parsed["gpu_start_cycle64"]):
            problems.append("kernel_cycles64 differs from end minus start")
    return {"status": "passed" if not problems else "failed",
            "problems": problems, "fields": parsed,
            "launch_markers": markers["model_chain_hbm: launch cluster 0"],
            "pass_markers": markers["PASS"],
            "fail_markers": markers["FAIL"]}


def collect(*, model: str, sm_count: int, job_id: str, bitstream_id: str,
            elf: Path, receipt: Path, source_manifest: Path, hwdb_config: Path,
            workload_config: Path, uart_log: Path,
            firesim_exit_code: int, counter_files: list[Path]) -> dict:
    receipt_data = json.loads(receipt.read_text())
    source_data = json.loads(source_manifest.read_text())
    elf_digest = sha256(elf)
    receipt_digest = receipt_data.get("firesim_elf_sha256")
    with uart_log.open(errors="replace") as stream:
        uart = parse_uart(stream)
    problems = list(uart["problems"])
    if not re.fullmatch(r"[0-9a-f]{64}", str(receipt_digest)):
        problems.append("ELF receipt has no SHA-256")
    elif elf_digest != receipt_digest:
        problems.append("staged ELF differs from its receipt")
    if (source_data.get("model") != model or
            source_data.get("stage_limit") is not None or
            source_data.get("scope") not in
            ("full_dimension_checkpoint_decoder", "full_checkpoint_action_chunk") or
            not source_data.get("device_elf_built") or
            not source_data.get("stages") or
            source_data.get("target_protocol") != "nightly_hbm_printbuf_cluster0"):
        problems.append("source manifest is not the requested full model HBM build")
    if source_data.get("soc_elf_sha256") != receipt_data.get("source_elf_sha256"):
        problems.append("packaged ELF receipt differs from the model source build")
    if firesim_exit_code != 0:
        problems.append(f"FireSim exited {firesim_exit_code}")
    hashes = [{"path": str(path.resolve()), "sha256": sha256(path)}
              for path in counter_files]
    return {
        "model": model, "sm_count": sm_count,
        "sm_count_source": "operator_declared",
        "job_id": job_id, "bitstream_id": bitstream_id,
        "bitstream_id_source": "operator_declared",
        "status": "passed" if not problems else "failed",
        "problems": problems, "firesim_exit_code": firesim_exit_code,
        "device_self_check_passed": uart["status"] == "passed",
        "full_model_fpga_execution": not problems,
        "performance_measurement": not problems,
        "utilization_measurement": False,
        "bandwidth_measurement": False,
        "area_power_measurement": False,
        "hardware_map_match_verified": False,
        "latency_unit": "GPU mcycle ticks; clock frequency not inferred",
        "kernel_cycles64": uart["fields"].get("kernel_cycles64"),
        "gpu_start_cycle64": uart["fields"].get("gpu_start_cycle64"),
        "gpu_end_cycle64": uart["fields"].get("gpu_end_cycle64"),
        "model_status": uart["fields"].get("model_status"),
        "upstream_execution_equivalent": False,
        "elf_sha256": elf_digest,
        "elf_receipt_sha256": sha256(receipt),
        "receipt_elf_sha256": receipt_digest,
        "source_manifest_sha256": sha256(source_manifest),
        "source_elf_sha256": source_data.get("soc_elf_sha256"),
        "source_stage_count": source_data.get("stages"),
        "source_scope": source_data.get("scope"),
        "checkpoint_sha256": source_data.get("checkpoint_sha256"),
        "hwdb_config_sha256": sha256(hwdb_config),
        "workload_config_sha256": sha256(workload_config),
        "uart_log_sha256": sha256(uart_log),
        "counter_files": hashes,
        "paths": {"elf": str(elf.resolve()),
                  "elf_receipt": str(receipt.resolve()),
                  "source_manifest": str(source_manifest.resolve()),
                  "hwdb_config": str(hwdb_config.resolve()),
                  "workload_config": str(workload_config.resolve()),
                  "uart_log": str(uart_log.resolve())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=("tinyllama", "deepseek_r1_distill_qwen_1_5b",
                                 "gemma_2_2b_it", "smolvla_base"))
    parser.add_argument("--sm-count", type=int, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--bitstream-id", required=True)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--hwdb-config", type=Path, required=True)
    parser.add_argument("--workload-config", type=Path, required=True)
    parser.add_argument("--uart-log", type=Path, required=True)
    parser.add_argument("--firesim-exit-code", type=int, required=True)
    parser.add_argument("--counter-file", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.sm_count not in (1, 2, 4):
        parser.error("SM count must be 1, 2, or 4")
    result = collect(model=args.model, sm_count=args.sm_count,
                     job_id=args.job_id, bitstream_id=args.bitstream_id,
                     elf=args.elf, receipt=args.receipt,
                     source_manifest=args.source_manifest,
                     hwdb_config=args.hwdb_config,
                     workload_config=args.workload_config,
                     uart_log=args.uart_log,
                     firesim_exit_code=args.firesim_exit_code,
                     counter_files=args.counter_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
