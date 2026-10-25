#!/usr/bin/env python3
"""Run the logical HPC matrix on Cyclotron's generic DRAM timing model.

This is intentionally separate from campaign.py: it only drives the public
Cyclotron model and never labels that model as HBM. FPGA and HBM runs remain
asset-driven so they use the copied bitstreams/configurations verbatim.
"""

from __future__ import annotations

import argparse
from array import array
import csv
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


FINISHED = re.compile(r"simulation finished after (\d+) cycles")
KERNEL_CYCLES = re.compile(r"nightly_kernel_cycles=(\d+)")
CLUSTER_CYCLES = re.compile(
    r"nightly_cluster_cycles cluster=(\d+) cycles=(\d+) start=(\d+) end=(\d+)"
)
GMEM_PASS = re.compile(r"gmem check .*: PASS")
SMEM_PASS = re.compile(r"smem check .*: PASS")
RV32_SECTION = re.compile(
    r"\.rv32\.seg\d+\s+\S+\s+([0-9a-fA-F]+)\s+\S+\s+([0-9a-fA-F]+)\b"
)
TIMING_FILES = (
    "gmem.toml", "smem.toml", "icache.toml", "lsu_scheduler.toml",
    "execute.toml", "writeback_operand_fetch.toml", "fence.toml",
    "barrier.toml", "dma_tensor.toml",
)
SM_NAME_TOKEN = re.compile(r"-{1,2}(?:1|2|4)sm(?=-|$)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def workload_group_name(name: str) -> str:
    """Remove an explicit SM-count token wherever a build name places it."""
    return SM_NAME_TOKEN.sub("", name)


def output_address(elf: Path) -> int:
    """Return the first output word, 64 bytes after the device BSS start."""
    proc = subprocess.run(
        ["readelf", "-SW", str(elf)], capture_output=True, text=True, check=True,
    )
    candidates = []
    for match in RV32_SECTION.finditer(proc.stdout):
        address, size = (int(value, 16) for value in match.groups())
        if size == 0 and 0x1_0000_0000 <= address < 0x2_0000_0000:
            candidates.append(address - 0x1_0000_0000)
    if len(candidates) != 1:
        raise ValueError(
            f"expected one zero-sized RV32 BSS marker in {elf}, found {candidates}"
        )
    return candidates[0] + 64


def toml_quote(value: object) -> str:
    return json.dumps(str(value))


@lru_cache(maxsize=None)
def unique_u32_values(path: Path) -> int:
    values = array("I")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    if sys.byteorder != "little":
        values.byteswap()
    return len(set(values))


def workload_checks(row: dict, result: dict, elf: Path) -> tuple[list[str], list[str]]:
    gmem: list[str] = []
    smem: list[str] = []
    family = row["family"]
    if family in ("stream", "stream_dot", "spatter", "xrage", "gemv_bf16",
                  "stencil3d", "nearn", "spmv_jds", "sfilter"):
        address = output_address(elf)
        if family == "spatter":
            address += int(result.get("output_pad_cache_lines", 0)) * 64
        if family in ("stream", "stream_dot"):
            words = int(result.get("output_element_count", result["stream_elements"]))
        elif family in ("stencil3d", "nearn", "spmv_jds", "sfilter"):
            words = int(result["output_element_count"])
        elif family == "gemv_bf16":
            words = int(result["output_words"])
        else:
            words = int(result["output_elements"]) * 2
        expected = result.get("expected_digest")
        if expected:
            gmem.append(
                "{ label = %s, address = 0x%08x, words = %d, expected_fnv64 = %s }"
                % (toml_quote(row["name"]), address, words, toml_quote(expected))
            )
        else:
            probe = result.get("exploratory_probe_index")
            if probe is None:
                raise ValueError(f"{row['name']}: no digest or nonzero probe")
            expected_nonzero = result.get("expected_nonzero_outputs")
            if expected_nonzero is None and result.get("count") == 1:
                pattern = next(
                    (item for item in result.get("pattern_inputs", [])
                     if item.get("field") == "pattern"), None,
                )
                if pattern:
                    expected_nonzero = unique_u32_values(Path(pattern["path"]))
            if expected_nonzero is None:
                raise ValueError(
                    f"{row['name']}: exploratory output has no exact coverage count"
                )
            gmem.append(
                "{ label = %s, address = 0x%08x, words = %d, expected_nonzero_u64 = %d }"
                % (toml_quote(row["name"] + "-coverage"), address, words,
                   int(expected_nonzero))
            )
    elif family == "fp8_gemm":
        # nightly/verify.h: done, bad, poison, and nonfinite slots.
        for cluster in range(int(row["sm_count"])):
            for label, slot, expected in (
                ("done", 16, 0x5EF1D0E5), ("bad", 20, 0),
                ("poison", 21, 0), ("nonfinite", 22, 0),
            ):
                smem.append(
                    "{ label = %s, cluster = %d, address = 0x%08x, expected = 0x%08x }"
                    % (toml_quote(f"gemm-{label}"), cluster, 0x80000 + slot * 8, expected)
                )
    else:
        raise ValueError(f"unsupported family: {family}")
    return gmem, smem


def make_config(row: dict, result: dict, elf: Path, timing_dir: Path,
                timeout: int, timing: bool) -> str:
    sms = int(row["sm_count"])
    gmem, smem = workload_checks(row, result, elf)
    tags = ",\n  ".join(
        f"[[0x80038, 0x{0xC1D00000 | cluster:08x}], "
        f"[0x8003c, 0x{0xC1D00000 | cluster:08x}]]"
        for cluster in range(sms)
    )
    lines = [
        "[muon]", "num_lanes = 16", "num_warps = 8", "num_cores = 2",
        "num_regs = 256", "start_pc = 0x10000000", "",
        "[mem]", "io_cout_addr = 0xFF080000", "io_cout_size = 64", "",
        "[sim]", f"elf = {toml_quote(elf)}", f"timeout = {timeout}",
        "log_level = 0", "trace = false", f"timing = {str(timing).lower()}",
        f"num_clusters = {sms}", "report_nightly_cycles = true",
        "smem_init = [", "  [0x80000, 0x6e1a0c11],",
        "  [0x80004, 0x6e1a0c11],", "  [0x800c0, 0x60c0ffee],",
        "  [0x800c4, 0x60c0ffee],", "]",
        "smem_init_per_cluster = [", f"  {tags}", "]",
        "gmem_checks = [",
        *(f"  {check}," for check in gmem),
        "]", "smem_checks = [",
        *(f"  {check}," for check in smem),
        "]",
    ]
    if timing:
        includes = ",\n  ".join(
            toml_quote((timing_dir / name).resolve()) for name in TIMING_FILES
        )
        lines.extend(["", "[timing]", "include = [", f"  {includes}", "]"])
    return "\n".join(lines) + "\n"


def select_rows(manifest: dict, sms: set[int], families: set[str]) -> list[dict]:
    rows = [
        row for row in manifest["workloads"]
        if row["loader"] == "logical" and int(row["sm_count"]) in sms
        and row["family"] in families
    ]
    return sorted(rows, key=lambda row: (row["name"], row["sm_count"]))


def run(args: argparse.Namespace) -> None:
    manifest_path = args.build_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("hardware_rebuilt") is not False or manifest.get("no_vpu") is not True:
        raise ValueError("matrix must assert hardware_rebuilt=false and no_vpu=true")
    rows = select_rows(manifest, set(args.sms), set(args.families))
    if args.match:
        matcher = re.compile(args.match)
        rows = [row for row in rows if matcher.search(row["name"])]
    if not rows:
        raise ValueError("selection produced no workloads")
    if not args.cyclotron.is_file():
        raise ValueError(f"missing executable: {args.cyclotron}")
    for name in TIMING_FILES:
        if args.timing and not (args.timing_dir / name).is_file():
            raise ValueError(f"missing timing component: {args.timing_dir / name}")

    campaign = {
        "schema": "radiance-cyclotron-dram-campaign-v1",
        "memory_model": "generic-dram", "is_hbm": False,
        "timing": args.timing, "build_manifest": str(manifest_path),
        "build_manifest_sha256": sha256(manifest_path),
        "cyclotron": str(args.cyclotron.resolve()),
        "cyclotron_sha256": sha256(args.cyclotron),
        "rows": len(rows), "repetitions": args.repetitions,
    }
    write_json(args.out / "campaign.json", campaign)
    for row in rows:
        elf = Path(row["elf"]).resolve()
        if sha256(elf) != row["elf_sha256"]:
            raise ValueError(f"ELF changed: {elf}")
        result = json.loads((elf.parent / "result.json").read_text())
        for repetition in range(args.repetitions):
            run_dir = args.out / "jobs" / row["name"] / f"rep-{repetition}"
            result_file = run_dir / "result.json"
            if args.resume and result_file.is_file():
                old = json.loads(result_file.read_text())
                # A reused output directory may be pointed at a rebuilt
                # manifest.  Never let --resume silently promote a passing
                # result from an older ELF.
                if (old.get("status") == "passed" and
                        old.get("elf_sha256") == row["elf_sha256"] and
                        old.get("timing") == args.timing and
                        int(old.get("sm_count", -1)) == int(row["sm_count"])):
                    continue
            run_dir.mkdir(parents=True, exist_ok=True)
            config = make_config(
                row, result, elf, args.timing_dir, args.timeout, args.timing,
            )
            (run_dir / "config.toml").write_text(config)
            started = time.time()
            proc = subprocess.run(
                [str(args.cyclotron.resolve()), str(run_dir / "config.toml")],
                cwd=run_dir, capture_output=True, text=True,
                env={**os.environ, "CYCLOTRON_PERF_LOG_DIR": str(run_dir / "perf")},
            )
            (run_dir / "stdout.log").write_text(proc.stdout)
            (run_dir / "stderr.log").write_text(proc.stderr)
            finished = FINISHED.search(proc.stdout)
            kernel = KERNEL_CYCLES.search(proc.stdout)
            cluster_cycles = [
                {
                    "cluster": int(cluster), "cycles": int(cycles),
                    "start": int(start), "end": int(end),
                }
                for cluster, cycles, start, end in CLUSTER_CYCLES.findall(proc.stdout)
            ]
            expected_checks = 1 if row["family"] != "fp8_gemm" else 4 * row["sm_count"]
            observed_checks = len(GMEM_PASS.findall(proc.stdout)) + len(SMEM_PASS.findall(proc.stdout))
            passed = (proc.returncode == 0 and finished is not None and kernel is not None
                      and observed_checks == expected_checks
                      and len(cluster_cycles) == int(row["sm_count"])
                      and {item["cluster"] for item in cluster_cycles}
                      == set(range(int(row["sm_count"]))))
            record = {
                "schema": "radiance-cyclotron-dram-result-v1",
                "status": "passed" if passed else "failed",
                "name": row["name"], "family": row["family"],
                "sm_count": row["sm_count"], "loader": row["loader"],
                "memory_model": "generic-dram", "is_hbm": False,
                "timing": args.timing, "repetition": repetition,
                "kernel_cycles": int(kernel.group(1)) if kernel else None,
                "cluster_kernel_cycles": cluster_cycles,
                "simulation_cycles": int(finished.group(1)) if finished else None,
                "correctness_checks": observed_checks,
                "expected_correctness_checks": expected_checks,
                "returncode": proc.returncode,
                "wall_seconds": time.time() - started,
                "elf": str(elf), "elf_sha256": row["elf_sha256"],
            }
            write_json(result_file, record)
            print(f"{record['status']:6} {row['name']:40} "
                  f"cycles={record['kernel_cycles']}", flush=True)
            if not passed and not args.keep_going:
                raise RuntimeError(f"model run failed: {row['name']}")
    summarize(args.out)


def summarize(root: Path) -> None:
    records = [
        json.loads(path.read_text()) for path in sorted((root / "jobs").rglob("result.json"))
    ]
    passed = [row for row in records if row.get("status") == "passed"]
    grouped: dict[tuple[str, int], list[int]] = {}
    for row in passed:
        grouped.setdefault((workload_group_name(row["name"]), row["sm_count"]), []).append(
            row["kernel_cycles"]
        )
    rows = []
    workloads = sorted({name for name, _ in grouped})
    for name in workloads:
        medians = {
            sm: statistics.median(grouped[(name, sm)])
            for sm in (1, 2, 4) if (name, sm) in grouped
        }
        rows.append({
            "workload": name,
            "cycles_1sm": medians.get(1), "cycles_2sm": medians.get(2),
            "cycles_4sm": medians.get(4),
            "speedup_1_to_2": (medians[1] / medians[2]) if 1 in medians and 2 in medians else None,
            "speedup_2_to_4": (medians[2] / medians[4]) if 2 in medians and 4 in medians else None,
        })
    summary = {
        "schema": "radiance-cyclotron-dram-summary-v1",
        "memory_model": "generic-dram", "is_hbm": False,
        "result_count": len(records), "passed": len(passed),
        "failed": len(records) - len(passed), "workloads": rows,
    }
    write_json(root / "summary.json", summary)
    with (root / "summary.csv").open("w", newline="") as stream:
        fields = list(rows[0]) if rows else ["workload"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--cyclotron", type=Path, required=True)
    parser.add_argument("--timing-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sms", type=int, nargs="+", choices=(1, 2, 4), default=(1, 2, 4))
    parser.add_argument("--families", nargs="+",
                        choices=("stream", "stream_dot", "spatter", "xrage",
                                 "fp8_gemm", "gemv_bf16", "stencil3d", "nearn",
                                 "spmv_jds", "sfilter"),
                        default=("stream", "stream_dot", "spatter", "xrage",
                                 "fp8_gemm", "gemv_bf16", "stencil3d", "nearn",
                                 "spmv_jds", "sfilter"))
    parser.add_argument("--match", help="regular expression matched against workload names")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=2_000_000_000)
    parser.add_argument("--functional", dest="timing", action="store_false",
                        help="run functional validation without DRAM timing")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    if args.repetitions <= 0 or args.timeout <= 0:
        parser.error("repetitions and timeout must be positive")
    args.out = args.out.resolve()
    run(args)


if __name__ == "__main__":
    main()
