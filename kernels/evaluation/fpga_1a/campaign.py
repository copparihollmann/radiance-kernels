#!/usr/bin/env python3
"""Inventory, plan, run, and summarize the FPGA-1A HPC campaign.

The adapter is deliberately data driven: Nicolas's copied scripts define the
actual board/model command in assets.json.  This code never builds hardware and
never reads from another user's directory.  It hashes every consumed artifact,
matches ELFs by physical SM count and loader semantics, and resumes completed
repetitions safely.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import time


PASS = re.compile(r"(?:^|\n)(?:PASS|\*\*\* PASSED \*\*\*)", re.MULTILINE)
FAIL = re.compile(r"(?:^|\n)(?:FAIL|\*\*\* FAILED \*\*\*)", re.MULTILINE)
CYCLES = re.compile(r"\bkernel_cycles\s*[=:]\s*(\d+)")
CORE_CYCLES = re.compile(
    r"cluster\s+(\d+)\s+core\s+(\d+):\s+start=(\d+)\s+end=(\d+)\s+cycles=(\d+)"
)
ROLES = {
    ".bit": "bitstream", ".xclbin": "bitstream", ".bin": "binary",
    ".elf": "elf", ".json": "configuration", ".yaml": "configuration",
    ".yml": "configuration", ".toml": "configuration", ".ini": "configuration",
    ".py": "script", ".sh": "script", ".pl": "script",
}
PATH_FIELDS = ("bitstream", "driver", "runtime_config", "hwdb_config",
               "model_config", "runner", "parser")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def role(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith((".tar.gz", ".tgz")) and any(
            word in lower for word in ("firesim", "bitstream", "radiance")):
        return "bitstream_archive"
    return ROLES.get(path.suffix.lower(), "other")


def inventory(root: Path) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"asset root is missing: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            target = path.resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"asset symlink escapes the handoff: {path}")
        if path.is_file():
            files.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role_guess": role(path),
            })
    if not files:
        raise ValueError(
            f"asset handoff is empty: {root}; copy Nicolas's bitstreams, configs, "
            "model executables, and scripts here first"
        )
    return {"schema": "radiance-fpga-1a-asset-inventory-v1",
            "root": str(root), "file_count": len(files), "files": files}


def inside(root: Path, raw: str, label: str) -> Path:
    path = (root / raw).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes the copied asset root: {raw}")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    return path


def load_assets(path: Path, asset_root: Path) -> tuple[dict, dict]:
    spec = json.loads(path.read_text())
    if spec.get("schema") != "radiance-fpga-1a-assets-v1":
        raise ValueError("assets.json has an unsupported schema")
    targets = spec.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("assets.json must contain at least one target")
    ids = set()
    receipts = {}
    for target in targets:
        target_id = target.get("id")
        if not isinstance(target_id, str) or not target_id or target_id in ids:
            raise ValueError(f"target id is empty or duplicated: {target_id!r}")
        ids.add(target_id)
        if target.get("kind") not in ("firesim", "model"):
            raise ValueError(f"{target_id}: kind must be firesim or model")
        if target.get("sm_count") not in (1, 2, 4):
            raise ValueError(f"{target_id}: sm_count must be 1, 2, or 4")
        if target.get("memory") not in ("dram", "hbm"):
            raise ValueError(f"{target_id}: memory must be dram or hbm")
        if target.get("loader") not in ("logical", "direct-hashed"):
            raise ValueError(f"{target_id}: loader must be logical or direct-hashed")
        argv = target.get("run_argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
            raise ValueError(f"{target_id}: run_argv must be a nonempty string array")
        joined_argv = "\0".join(argv)
        if "{elf}" not in joined_argv:
            raise ValueError(f"{target_id}: run_argv does not consume the ELF")
        registry_config = target.get("bitstream_config")
        registry_selected = "{bitstream_config}" in joined_argv
        if target["kind"] == "firesim" and not (
                "{bitstream}" in joined_argv or registry_selected):
            raise ValueError(
                f"{target_id}: FireSim run_argv must consume either the bitstream "
                "path or a registry bitstream_config"
            )
        if registry_selected:
            if not isinstance(registry_config, str) or not registry_config:
                raise ValueError(f"{target_id}: missing bitstream_config")
            expected_archive = f"firesim_radiance{registry_config}.tar.gz"
            expected_driver = (
                f"FireSim-corigine_mimicturbo_gt-radiance{registry_config}"
            )
            if Path(str(target.get("bitstream", ""))).name != expected_archive:
                raise ValueError(
                    f"{target_id}: -b {registry_config} does not match "
                    f"bitstream {target.get('bitstream')!r}"
                )
            if Path(str(target.get("driver", ""))).name != expected_driver:
                raise ValueError(
                    f"{target_id}: -b {registry_config} does not match "
                    f"driver {target.get('driver')!r}"
                )
        if target["kind"] == "model" and "{model_config}" not in joined_argv:
            raise ValueError(f"{target_id}: model run_argv does not consume model_config")
        if target.get("parser_argv") and not target.get("parser"):
            raise ValueError(f"{target_id}: parser_argv needs a copied parser")
        reuse_argv = target.get("reuse_argv")
        if reuse_argv is not None and (
                not isinstance(reuse_argv, list) or not reuse_argv or
                not all(isinstance(v, str) for v in reuse_argv)):
            raise ValueError(f"{target_id}: reuse_argv must be a nonempty string array")
        bdf = target.get("bdf")
        if bdf is not None and not re.fullmatch(
                r"[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", str(bdf)):
            raise ValueError(f"{target_id}: invalid FPGA BDF: {bdf!r}")
        for field in PATH_FIELDS:
            raw = target.get(field)
            if raw is not None:
                resolved = inside(asset_root, raw, f"{target_id}.{field}")
                receipts[f"{target_id}.{field}"] = {
                    "path": raw, "bytes": resolved.stat().st_size,
                    "sha256": sha256(resolved),
                }
        if target["kind"] == "firesim" and "bitstream" not in target:
            raise ValueError(f"{target_id}: FireSim target has no bitstream")
        if target["kind"] == "model" and "model_config" not in target:
            raise ValueError(f"{target_id}: model target has no model_config")
    single_bdf = spec.get("single_fpga_bdf")
    if single_bdf is not None:
        mismatched = [
            target["id"] for target in targets
            if target.get("kind") == "firesim" and target.get("bdf") != single_bdf
        ]
        if mismatched:
            raise ValueError(
                f"single_fpga_bdf {single_bdf} is not used by: " +
                ", ".join(mismatched)
            )
    referenced = [t["bitstream"] for t in targets if t.get("bitstream")]
    if len(referenced) != len(set(referenced)):
        raise ValueError("each bitstream must be represented by exactly one target")
    discovered = {
        row["path"] for row in inventory(asset_root)["files"]
        if row["role_guess"] in ("bitstream", "bitstream_archive")
    }
    unrepresented = discovered - set(referenced)
    if unrepresented:
        raise ValueError("bitstreams omitted from assets.json:\n  " +
                         "\n  ".join(sorted(unrepresented)))
    return spec, receipts


def required_workloads(rows: list[dict]) -> list[str]:
    problems = []
    expected_stream = {"copy", "scale", "add", "triad"}
    expected_spatter = set(range(5))
    expected_xrage = {"gather", "scatter-ordered", "scatter-parallel"}
    for sm in (1, 2, 4):
        for loader in ("logical", "direct-hashed"):
            selected = [r for r in rows if r.get("sm_count") == sm and r.get("loader") == loader]
            stream = {r.get("operation") for r in selected if r.get("family") == "stream"}
            spatter = {r.get("case") for r in selected if r.get("family") == "spatter"}
            xrage = {r.get("operation") for r in selected if r.get("family") == "xrage"}
            gemm = [r for r in selected if r.get("family") == "fp8_gemm"]
            if stream != expected_stream:
                problems.append(f"{sm}sm/{loader}: STREAM={sorted(stream)}")
            if spatter != expected_spatter:
                problems.append(f"{sm}sm/{loader}: Spatter={sorted(spatter)}")
            if xrage != expected_xrage:
                problems.append(f"{sm}sm/{loader}: xRAGE={sorted(xrage)}")
            if len(gemm) != 1 or gemm[0].get("vpu") is not False:
                problems.append(f"{sm}sm/{loader}: expected one no-VPU FP8 GEMM")
    return problems


def make_plan(build_path: Path, asset_path: Path, asset_root: Path,
              out: Path, warmups: int, repetitions: int,
              allow_incomplete: bool) -> dict:
    build = json.loads(build_path.read_text())
    if build.get("hardware_rebuilt") is not False or build.get("no_vpu") is not True:
        raise ValueError("build manifest does not assert no hardware rebuild and no VPU")
    rows = build.get("workloads", [])
    problems = required_workloads(rows)
    if problems and not allow_incomplete:
        raise ValueError("build matrix is incomplete:\n  " + "\n  ".join(problems))
    assets, receipts = load_assets(asset_path, asset_root)
    jobs = []
    for target in assets["targets"]:
        matched = [r for r in rows if r["sm_count"] == target["sm_count"]
                   and r["loader"] == target["loader"]]
        if not matched:
            raise ValueError(f"{target['id']}: no matching ELF configurations")
        for workload in matched:
            elf = Path(workload["elf"]).resolve()
            if not elf.is_file() or sha256(elf) != workload["elf_sha256"]:
                raise ValueError(f"ELF missing or changed: {elf}")
            for sample in range(warmups + repetitions):
                phase = "warmup" if sample < warmups else "measure"
                repetition = sample if phase == "warmup" else sample - warmups
                job_id = f"{target['id']}--{workload['name']}--{phase}-{repetition}"
                jobs.append({
                    "id": job_id, "target_id": target["id"], "target": target,
                    "workload": workload, "phase": phase, "repetition": repetition,
                    "result_dir": str((out / "jobs" / job_id).resolve()),
                })
    plan = {
        "schema": "radiance-fpga-1a-campaign-plan-v1",
        "build_manifest": str(build_path.resolve()),
        "build_manifest_sha256": sha256(build_path),
        "asset_manifest": str(asset_path.resolve()),
        "asset_manifest_sha256": sha256(asset_path),
        "asset_root": str(asset_root.resolve()),
        "asset_receipts": receipts,
        "warmups": warmups, "repetitions": repetitions,
        "jobs": jobs, "incomplete_build_problems": problems,
    }
    write_json(out / "plan.json", plan)
    return plan


def context(job: dict, asset_root: Path, result_dir: Path) -> dict[str, object]:
    target, workload = job["target"], job["workload"]
    values: dict[str, object] = {
        "elf": workload["elf"], "out": str(result_dir), "target_id": target["id"],
        "memory": target["memory"], "sm_count": target["sm_count"],
        "loader": target["loader"], "repetition": job["repetition"],
        "phase": job["phase"], "bdf": target.get("bdf", ""),
        "bitstream_config": target.get("bitstream_config", ""),
    }
    for field in PATH_FIELDS:
        values[field] = str((asset_root / target[field]).resolve()) if target.get(field) else ""
    values["stdout"] = str(result_dir / "stdout.log")
    values["stderr"] = str(result_dir / "stderr.log")
    return values


def expand(argv: list[str], values: dict[str, object]) -> list[str]:
    try:
        return [arg.format_map(values) for arg in argv]
    except KeyError as error:
        raise ValueError(f"unknown run_argv placeholder: {error.args[0]}") from error


def execute_job(job: dict, asset_root: Path, receipts: dict,
                rerun_failed: bool, reuse_loaded: bool) -> tuple[bool, bool]:
    """Execute one job; return (did_run, target_image_is_known_loaded)."""
    result_dir = Path(job["result_dir"])
    result_file = result_dir / "result.json"
    if result_file.is_file():
        old = json.loads(result_file.read_text())
        if old.get("status") == "passed" or not rerun_failed:
            return False, reuse_loaded
    result_dir.mkdir(parents=True, exist_ok=True)
    values = context(job, asset_root, result_dir)
    target = job["target"]
    argv_template = target.get("reuse_argv") if reuse_loaded else None
    argv = expand(argv_template or target["run_argv"], values)
    env = None
    if target.get("env"):
        import os
        env = os.environ.copy()
        env.update({key: str(value).format_map(values)
                    for key, value in target["env"].items()})
    request = {"job": job, "argv": argv, "asset_receipts": receipts,
               "reused_loaded_bitstream": bool(argv_template)}
    write_json(result_dir / "request.json", request)
    started = time.time()
    with (result_dir / "stdout.log").open("w") as stdout, \
            (result_dir / "stderr.log").open("w") as stderr:
        completed = subprocess.run(argv, cwd=asset_root, env=env,
                                   stdout=stdout, stderr=stderr, check=False)
    output = (result_dir / "stdout.log").read_text(errors="replace") + "\n" + \
             (result_dir / "stderr.log").read_text(errors="replace")
    cycles = CYCLES.findall(output)
    core_cycles = [
        {"cluster": int(cluster), "core": int(core), "start": int(start),
         "end": int(end), "cycles": int(duration)}
        for cluster, core, start, end, duration in CORE_CYCLES.findall(output)
    ]
    expected_cores = {
        (cluster, core)
        for cluster in range(int(target["sm_count"]))
        for core in range(2)
    }
    observed_cores = {(row["cluster"], row["core"]) for row in core_cycles}
    timing_complete = (len(core_cycles) == len(expected_cores) and
                       observed_cores == expected_cores and
                       all(row["end"] >= row["start"] and
                           row["cycles"] == row["end"] - row["start"]
                           for row in core_cycles))
    passed = (bool(PASS.search(output)) and not FAIL.search(output) and
              completed.returncode == 0)
    if target.get("require_per_sm_timing"):
        passed = passed and bool(cycles) and timing_complete
    result = {
        "schema": "radiance-fpga-1a-job-result-v1", "id": job["id"],
        "status": "passed" if passed else "failed", "returncode": completed.returncode,
        "kernel_cycles": int(cycles[-1]) if cycles else None,
        "core_kernel_cycles": core_cycles,
        "per_sm_timing_complete": timing_complete,
        "reused_loaded_bitstream": bool(argv_template),
        "wall_seconds": round(time.time() - started, 6),
    }
    parser_argv = target.get("parser_argv")
    if parser_argv:
        parsed = subprocess.run(expand(parser_argv, values), cwd=asset_root,
                                capture_output=True, text=True, check=False)
        (result_dir / "parser.stdout").write_text(parsed.stdout)
        (result_dir / "parser.stderr").write_text(parsed.stderr)
        if parsed.returncode == 0:
            try:
                result["parsed"] = json.loads(parsed.stdout)
            except json.JSONDecodeError:
                result["parser_error"] = "parser stdout is not JSON"
        else:
            result["parser_error"] = f"parser exited {parsed.returncode}"
    write_json(result_file, result)
    print(f"{job['id']}: {result['status']}", flush=True)
    image_loaded = reuse_loaded or "=== Bring-up done ===" in output
    return True, image_loaded


def execute(plan_path: Path, rerun_failed: bool, parallel_targets: bool = False) -> None:
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "radiance-fpga-1a-campaign-plan-v1":
        raise ValueError("unsupported plan schema")
    if sha256(Path(plan["build_manifest"])) != plan["build_manifest_sha256"]:
        raise ValueError("build manifest changed after planning")
    if sha256(Path(plan["asset_manifest"])) != plan["asset_manifest_sha256"]:
        raise ValueError("asset manifest changed after planning")
    asset_root = Path(plan["asset_root"])
    for label, receipt in plan["asset_receipts"].items():
        path = inside(asset_root, receipt["path"], label)
        if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
            raise ValueError(f"asset changed after planning: {label}")

    grouped: dict[str, list[dict]] = {}
    for job in plan["jobs"]:
        grouped.setdefault(job["target_id"], []).append(job)

    # Corigine programming disconnects a card and performs a host-wide PCIe
    # rescan.  A second target must not be programmed while a FireSim driver is
    # active on an already-programmed card: the rescan can invalidate the live
    # driver's mappings and destabilize the host.  Keep model-only campaigns
    # parallel-capable, but make FPGA target groups strictly sequential.
    if parallel_targets and any(
            jobs[0]["target"].get("kind") == "firesim"
            for jobs in grouped.values()):
        raise ValueError(
            "parallel FireSim targets are unsafe: Corigine programming performs "
            "a host-wide PCIe rescan; run target groups sequentially"
        )

    def run_target(jobs: list[dict]) -> None:
        image_loaded = False
        for job in jobs:
            _, image_loaded = execute_job(
                job, asset_root, plan["asset_receipts"], rerun_failed, image_loaded)

    if parallel_targets and len(grouped) > 1:
        bdfs = [jobs[0]["target"].get("bdf") for jobs in grouped.values()
                if jobs[0]["target"].get("kind") == "firesim"]
        if any(not bdf for bdf in bdfs) or len(bdfs) != len(set(bdfs)):
            raise ValueError("parallel FireSim targets require distinct nonempty BDFs")
        with ThreadPoolExecutor(max_workers=len(grouped)) as pool:
            futures = [pool.submit(run_target, jobs) for jobs in grouped.values()]
            for future in futures:
                future.result()
    else:
        for jobs in grouped.values():
            run_target(jobs)


def summarize(plan_path: Path, out: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    groups: dict[tuple, list[dict]] = {}
    missing = []
    for job in plan["jobs"]:
        result_path = Path(job["result_dir"]) / "result.json"
        if not result_path.is_file():
            missing.append(job["id"])
            continue
        result = json.loads(result_path.read_text())
        key = (job["target_id"], job["workload"]["name"])
        groups.setdefault(key, []).append({**result, "phase": job["phase"]})
    rows = []
    for (target_id, workload), results in sorted(groups.items()):
        samples = [r["kernel_cycles"] for r in results
                   if r["phase"] == "measure" and r["status"] == "passed"
                   and r.get("kernel_cycles") is not None]
        rows.append({
            "target_id": target_id, "workload": workload,
            "measure_passes": len(samples),
            "median_kernel_cycles": statistics.median(samples) if samples else None,
            "min_kernel_cycles": min(samples) if samples else None,
            "max_kernel_cycles": max(samples) if samples else None,
            "all_results_passed": all(r["status"] == "passed" for r in results),
        })
    report = {"schema": "radiance-fpga-1a-campaign-summary-v1",
              "rows": rows, "missing_jobs": missing}
    write_json(out, report)
    print(f"summarized {len(rows)} target/workload groups; {len(missing)} jobs missing")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--assets", type=Path, required=True)
    inv.add_argument("--out", type=Path, required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("--build-manifest", type=Path, required=True)
    planning.add_argument("--asset-manifest", type=Path, required=True)
    planning.add_argument("--assets", type=Path, required=True)
    planning.add_argument("--out", type=Path, required=True)
    planning.add_argument("--warmups", type=int, default=1)
    planning.add_argument("--repetitions", type=int, default=3)
    planning.add_argument("--allow-incomplete-build", action="store_true")
    running = sub.add_parser("run")
    running.add_argument("--plan", type=Path, required=True)
    running.add_argument("--rerun-failed", action="store_true")
    running.add_argument("--parallel-targets", action="store_true",
                         help="run non-FireSim target groups concurrently; FPGA targets "
                              "are rejected because programming rescans host PCIe")
    summary = sub.add_parser("summarize")
    summary.add_argument("--plan", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "inventory":
            write_json(args.out, inventory(args.assets))
        elif args.command == "plan":
            if args.warmups < 0 or args.repetitions < 1:
                parser.error("warmups must be nonnegative and repetitions positive")
            plan = make_plan(args.build_manifest, args.asset_manifest, args.assets.resolve(),
                             args.out.resolve(), args.warmups, args.repetitions,
                             args.allow_incomplete_build)
            print(f"planned {len(plan['jobs'])} runs under {args.out.resolve()}")
        elif args.command == "run":
            execute(args.plan.resolve(), args.rerun_failed, args.parallel_targets)
        else:
            summarize(args.plan.resolve(), args.out.resolve())
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
