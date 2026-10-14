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
EXPECTED_MODELS = {
    "tinyllama": (1133, "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"),
    "deepseek_r1_distill_qwen_1_5b": (
        1691, "58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945"),
    "gemma_2_2b_it": (1496, "1189d2f21b21a2ffce4119ee251a1385902eae85f43647a4fe5bc6bf3f57fe6a"),
    "smolvla_base": (3673, "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"),
}
EXPECTED_BATCH2 = {
    "tinyllama": {
        "selected_tensors": 135,
        "reference_output_sha256": "657e42c9256223d9427ce85256e4ad0ade329292b486f28e347aa77ed1b5bfab",
    },
    "deepseek_r1_distill_qwen_1_5b": {
        "selected_tensors": 171,
        "reference_output_sha256": "b1550ba6488f4c898444db34e234afde01cb8e5c31a176bed8402b570d8c793c",
    },
}


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
            if not re.fullmatch(r"(?:0x)?[0-9a-fA-F]{8}", status):
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


def parse_uart_batch2(log) -> dict:
    """Accept exactly two ordered cluster reports from the HBM batch host."""
    lines = list(log.splitlines() if isinstance(log, str) else log)
    launch = [index for index, raw in enumerate(lines)
              if raw.strip() == "model_chain_hbm: launch clusters=2"]
    passes = [index for index, raw in enumerate(lines) if raw.strip() == "PASS"]
    fails = [index for index, raw in enumerate(lines) if raw.strip() == "FAIL"]
    problems = []
    if len(launch) != 1 or len(passes) != 1 or fails:
        problems.append("expected one two-cluster launch and PASS with no FAIL")
    if any(raw.strip().startswith("model_chain_hbm: launch") and
           raw.strip() != "model_chain_hbm: launch clusters=2"
           for raw in lines):
        problems.append("contradictory launch marker")
    fields = {}
    claimed = []
    cursor = launch[0] if launch else -1
    for cluster in (0, 1):
        expected = [f"cluster_status={cluster}:", f"cluster={cluster}",
                    "gpu_start_cycle64=", "gpu_end_cycle64=", "kernel_cycles64="]
        record = {}
        for position, prefix in enumerate(expected):
            matches = [(index, raw.strip()[len(prefix):])
                       for index, raw in enumerate(lines)
                       if raw.strip().startswith(prefix)]
            if position >= 2:
                matches = [(index, value) for index, value in matches
                           if index > cursor and
                           (cluster == 1 or not any(
                               lines[prior].strip() == "cluster=1"
                               for prior in range(cursor + 1, index)))]
            if len(matches) != 1 or matches[0][0] <= cursor:
                problems.append(f"cluster {cluster}: missing, repeated, or unordered {prefix}")
                break
            cursor, value = matches[0]
            claimed.append(cursor)
            if position == 0:
                if not re.fullmatch(r"(?:0x)?[0-9a-fA-F]{8}", value):
                    problems.append(f"cluster {cluster}: malformed status")
                else:
                    record["model_status"] = int(value, 16)
            elif position >= 2:
                key = prefix[:-1]
                if not re.fullmatch(r"[0-9]+", value) or int(value) >= 1 << 64:
                    problems.append(f"cluster {cluster}: malformed {key}")
                else:
                    record[key] = int(value)
        if len(record) == 4:
            if record["model_status"] != PASS_STATUS:
                problems.append(f"cluster {cluster}: device status is not PASS")
            if record["gpu_end_cycle64"] <= record["gpu_start_cycle64"]:
                problems.append(f"cluster {cluster}: end does not follow start")
            elif record["kernel_cycles64"] != (record["gpu_end_cycle64"] -
                                                record["gpu_start_cycle64"]):
                problems.append(f"cluster {cluster}: cycles differ from end minus start")
            fields[cluster] = record
    report_prefixes = ("cluster_status=", "cluster=", "model_status=",
                       "gpu_start_cycle64=", "gpu_end_cycle64=",
                       "kernel_cycles64=")
    reported = [index for index, raw in enumerate(lines)
                if raw.strip().startswith(report_prefixes)]
    if sorted(claimed) != reported:
        problems.append("extra or duplicate cluster report field")
    if passes and cursor >= passes[0]:
        problems.append("PASS precedes a cluster report")
    return {"status": "passed" if not problems else "failed",
            "problems": problems, "fields": fields}


def check_source(model: str, source: dict, sm_count: int = 1) -> list[str]:
    stages, checkpoint = EXPECTED_MODELS[model]
    problems = []
    if (source.get("model") != model or source.get("stages") != stages or
            source.get("checkpoint_sha256") != checkpoint or
            source.get("stage_limit") is not None or
            not source.get("device_elf_built") or
            source.get("target_protocol") != (
                "nightly_hbm_printbuf_clusters" if sm_count == 2 else
                "nightly_hbm_printbuf_cluster0") or
            source.get("hbm_launch_clusters") != sm_count):
        problems.append("source manifest is not the pinned full model HBM build")
    transform = source.get("hbm_transform") or {}
    if model == "smolvla_base":
        if sm_count != 1:
            problems.append("two-cluster SmolVLA build is not supported")
        if (source.get("scope") != "full_checkpoint_action_chunk" or
                source.get("output_validation") != "all_elements_vs_upstream_policy" or
                source.get("output_check_tolerance") !=
                {"rtol": 1e-3, "atol": 1e-3} or
                not re.fullmatch(r"[0-9a-f]{64}",
                                 str(source.get("golden_output_sha256"))) or
                transform.get("scheduled_calls") != stages + 1 or
                transform.get("status_return_sites", 0) < 2):
            problems.append("SmolVLA build lacks its final-action reference check")
    elif sm_count == 2:
        expected_batch = EXPECTED_BATCH2.get(model)
        if (expected_batch is None or
                source.get("scope") != "full_dimension_checkpoint_decoder" or
                source.get("batch_clusters") != 2 or
                source.get("device_check_all_stages") is not False or
                len(source.get("verified_tensors", [])) !=
                expected_batch["selected_tensors"] or
                source.get("reference_output_sha256") !=
                expected_batch["reference_output_sha256"] or
                source.get("hbm_verifier_files") != 0 or
                transform.get("scheduled_calls") != stages or
                transform.get("status_return_sites", 0) < 1):
            problems.append("two-cluster decoder build lacks its selected checks")
    else:
        if (source.get("scope") != "full_dimension_checkpoint_decoder" or
                source.get("device_check_all_stages") is not True or
                len(source.get("verified_tensors", [])) != stages or
                source.get("native_verified_float_stages", 0) +
                source.get("native_verified_integer_stages", 0) != stages or
                source.get("hbm_verifier_files", 0) <= 0 or
                transform.get("chunk_return_sites") !=
                source.get("hbm_verifier_files") or
                transform.get("scheduled_calls") != stages or
                transform.get("status_return_sites", 0) < 1):
            problems.append("decoder build does not check every scheduled stage")
    return problems


def collect(*, model: str, sm_count: int, job_id: str, bitstream_id: str,
            elf: Path, receipt: Path, source_manifest: Path, hwdb_config: Path,
            workload_config: Path, uart_log: Path,
            firesim_exit_code: int, counter_files: list[Path],
            native_result: Path | None = None) -> dict:
    receipt_data = json.loads(receipt.read_text())
    source_data = json.loads(source_manifest.read_text())
    elf_digest = sha256(elf)
    receipt_digest = receipt_data.get("firesim_elf_sha256")
    with uart_log.open(errors="replace") as stream:
        uart = parse_uart_batch2(stream) if sm_count == 2 else parse_uart(stream)
    problems = list(uart["problems"])
    if not re.fullmatch(r"[0-9a-f]{64}", str(receipt_digest)):
        problems.append("ELF receipt has no SHA-256")
    elif elf_digest != receipt_digest:
        problems.append("staged ELF differs from its receipt")
    problems.extend(check_source(model, source_data, sm_count))
    if sm_count == 2:
        if native_result is None:
            problems.append("two-cluster native check receipt is required")
        else:
            native = json.loads(native_result.read_text())
            if (native.get("status") != "passed" or native.get("replicas") != 2 or
                    native.get("stages_per_replica") != source_data.get("stages") or
                    native.get("device_elf_sha256") != source_data.get("radiance_elf_sha256") or
                    native.get("weight_image_sha256") != source_data.get("weight_image_sha256") or
                    native.get("checkpoint_sha256") != source_data.get("checkpoint_sha256")):
                problems.append("two-cluster native check differs from the source build")
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
        "cluster_timings": uart["fields"] if sm_count == 2 else None,
        "upstream_execution_equivalent": False,
        "elf_sha256": elf_digest,
        "elf_receipt_sha256": sha256(receipt),
        "receipt_elf_sha256": receipt_digest,
        "source_manifest_sha256": sha256(source_manifest),
        "source_elf_sha256": source_data.get("soc_elf_sha256"),
        "source_stage_count": source_data.get("stages"),
        "source_scope": source_data.get("scope"),
        "source_output_check": ("final_action_chunk" if model == "smolvla_base"
                                else "selected_outputs_each_replica" if sm_count == 2
                                else "every_decoder_stage"),
        "checkpoint_sha256": source_data.get("checkpoint_sha256"),
        "hwdb_config_sha256": sha256(hwdb_config),
        "workload_config_sha256": sha256(workload_config),
        "uart_log_sha256": sha256(uart_log),
        "counter_files": hashes,
        "native_result_sha256": sha256(native_result) if native_result else None,
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
                        choices=tuple(EXPECTED_MODELS))
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
    parser.add_argument("--native-result", type=Path,
                        help="required for a two-SM model batch")
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
                     counter_files=args.counter_file,
                     native_result=args.native_result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
