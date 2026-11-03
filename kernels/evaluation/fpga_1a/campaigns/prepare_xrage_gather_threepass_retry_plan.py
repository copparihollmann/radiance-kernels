#!/usr/bin/env python3
"""Prepare the xRAGE gather three-pass follow-up: retry the real failure, fill
the real gap, and run the parameterized fix that has never touched hardware.

THE PLAN THAT WOULD HAVE BEEN WRONG

runs/xrage-gather-robust-dual-1a1b-v1 points at the ELFs under
runs/xrage-gather-robust-end-full-v2, which carry ONLY `robust_end_drain`: no
serial fence, no spin-until-reset, no post-spin, no drain-pass parameter.  Those
are PRE-FIX two-pass builds.  When they ran, 2SM_DRAM_baseline failed by
reproducing its own 2026-10-24 failure bit-exactly (104,000,437 cycles,
output_index 241892-241895, source_index 483435).  That is a determinism
demonstration, not a new defect, and re-running those builds buys nothing.

WHAT ACTUALLY HAPPENED, ESTABLISHED BY HASH

  * The accepted production host ELF for guarded-profile-xrage-gather is

        runs/hpc-profile-guard-builds-v1/xrage-gather/2sm/
            kernel.accepted-device-profile-host.soc.elf
        sha256 e1e7e130ea151ba02272fd57b0802958ed6350f6c6f3387e350bfb294b93d658

    It is the ELF named by runs/hpc-v2-memory-models-1a-v1/build-manifest.json,
    by the compute-instrumented build manifest, and by every v2 plan.

  * On 2026-11-02 00:33 that exact ELF FAILED once on the v1 2SM_HBM image:

        runs/fpga-hpc-bottleneck-guarded-2sm-all-1a-v1/jobs/
            firesim-2sm-hbm--guarded-profile-xrage-gather--measure-0

    kernel_cycles 123,728,293, returncode 1, four consecutive zero outputs at
    output_index 5,632,092-5,632,095 (source_index 347165, 347228, 347167,
    347230), mismatch_count=4.  n=1.  It was never retried, and the sibling
    DRAM_v2/HBM_v2 cells all passed with the same ELF.
    5,632,092 x 8 = 1,408,023 x 32, so the run of four lands exactly on a
    32-byte boundary.

  * Its device payload was transplanted from
    runs/xrage-gather-blockcyclic1024-nowait-v1 (fused sha256 94a6e65b74...),
    built at source revision 9682f33537aa7fa95cc82715c188de5ebf4fc83c.

WHICH ELF IS "THE THREE-PASS BUILD" -- THE DISTINCTION THIS PLAN TURNS ON

Searching every build manifest under runs/ for `robust_drain_passes` == 3
together with robust_serial_fence / robust_spin_until_reset / robust_post_spin
returns exactly two artifacts, both in runs/xrage-gather-current-three-pass-v1:

    051957ffff4a2351bd3dba655f0d4efcb620c497d714bc834128550619a07244  direct-hashed
    d68cf428f85718e65b727c650d6eeb72d75a1624f0222857f8df500fdff617a9  logical

NEITHER is e1e7e130ea.  e1e7e130ea does not carry a `robust_drain_passes` field
at all, and its `robust_post_spin` is 0.

That is not a contradiction, and the README of
runs/xrage-gather-current-three-pass-v1 states the resolution directly:

    "The failed 9682f335 source already contained a hard-coded third drain
     pass.  Its result metadata did not expose a drain-pass count and its
     post-spin was zero."

So there are TWO three-pass gather builds, and they are different artifacts:

    e1e7e130ea  ACCEPTED, in production, three drain passes HARD-CODED in
                source 9682f335, post-spin 0, no drain-pass metadata.
                This is the ELF that failed at 00:33.

    051957ffff  the PARAMETERIZED fix: --robust-drain-passes 3 together with
                --robust-post-spin 20000, built at a0142f0752...,
                locally validated against Cyclotron over all 16,737,936 output
                words with a one-bit negative control, and NEVER RUN ON
                HARDWARE.  Its own README calls the inert FPGA rerun "the
                required final test".

Both carry the full checker: digest-checked, diagnostic_readback,
host_full_readback, occupancy 8, block-cyclic 1024.

WHAT THIS PLAN RUNS -- 6 jobs

  3  e1e7e130ea on the v1 2SM_HBM image, repetitions 0/1/2.
     The exact failing cell, retried, so a one-off is distinguishable from a
     reproducible defect.  Nothing else about the cell changes: same image,
     same runner, same asset receipts.

  2  051957ffff on the SAME v1 2SM_HBM cell, repetitions 0/1.
     The parameterized three-pass + 20,000-cycle post-spin fix, measured
     against the failure it was written for, on the one configuration where the
     failure has ever been seen.  This is its first hardware execution.

  1  the 4SM_DRAM gather cell that has never run.

ABOUT THAT 4SM_DRAM CELL -- STATED PLAINLY

There is no 4-SM build of either three-pass artifact.  The accepted family's
4-SM sibling (runs/xrage-gather-blockcyclic1024-nowait-v1, 4dbf9a2f57..., same
source 9682f335, same flags) HAS run on 4SM_DRAM and passed at 50,377,247
cycles, so that cell is not a gap.  The only gather cell on 4SM_DRAM that has
genuinely never produced a result is
firesim-4sm-dram--xrage-gather-4sm--measure-0 in
runs/xrage-gather-robust-dual-1a1b-v1, whose ELF is the PRE-FIX two-pass
0bd050ff0c...: robust_end_drain only, occupancy 4, no block-cyclic schedule, and
digest-checked WITHOUT diagnostic_readback or host_full_readback -- so a failure
there reports a digest mismatch and no output index.  That is the honest state
of the matrix.  This plan runs that cell, because it is the cell that has never
run, and labels in the plan exactly what it does and does not test.  No ELF is
rebuilt to manufacture a better one.

ORDERING, AND WHY IT MATTERS HERE MORE THAN USUALLY

campaign.execute_job() is fail-closed: a failed job raises SafetyAbort, which
ends that CARD's group.  One of the six jobs is being run precisely because it
may fail.  So the two coverage jobs are placed FIRST on their cards and the
retries LAST:

    1a:00.0   parameterized-fix rep 0  ->  accepted rep 0  ->  accepted rep 2
    1b:00.0   4SM_DRAM coverage        ->  parameterized-fix rep 1 -> accepted rep 1

A failure of accepted rep 0 therefore costs only accepted rep 2, never the fix
measurement and never the coverage job, and accepted rep 1 still runs on the
other card.  To collect a dropped repetition afterwards, re-invoke with
`--include-job <that job id>` and WITHOUT --rerun-failed: the failed
result.json is then preserved as evidence instead of being archived over.

SAFETY

Runs nothing.  Touches no FPGA, no /dev/xdma*, no /sys/bus/pci.  Read-only with
respect to /scratch/nicolas and to every existing result.json, plan.json, asset
manifest and build manifest -- including the live
runs/xrage-gather-robust-dual-1a1b-v1, from which only the ELF path is read.
Writes only into runs/xrage-gather-threepass-retry-v1.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path("/scratch/agustin/radiance-fpga-1a")
CAMPAIGN = ROOT / "src/radiance-kernels/kernels/evaluation/fpga_1a/campaign.py"
ASSET_ROOT = ROOT / "artifacts/nicolas-fpga-assets-v1-hardened"
SOURCE_ASSETS = ASSET_ROOT / "assets.json"

OUT = ROOT / "runs/xrage-gather-threepass-retry-v1"
ASSETS = OUT / "assets-dual-bdf.json"
BUILD_MANIFEST = OUT / "build-manifest.json"

BDFS = ["1a:00.0", "1b:00.0"]
HBM = "firesim-2sm-hbm"
FOURSM = "firesim-4sm-dram"

FAILURE = {
    "job": str(
        ROOT / "runs/fpga-hpc-bottleneck-guarded-2sm-all-1a-v1/jobs"
        / "firesim-2sm-hbm--guarded-profile-xrage-gather--measure-0"
    ),
    "when": "2026-11-02T00:33 local",
    "bitstream_config": "2SM_HBM",
    "kernel_cycles": 123728293,
    "returncode": 1,
    "mismatch_count": 4,
    "output_index_range": [5632092, 5632095],
    "source_indices": [347165, 347228, 347167, 347230],
    "shape": "four consecutive all-zero 64-bit outputs",
    "alignment_note": (
        "5,632,092 x 8 = 1,408,023 x 32: the run of four begins exactly on a "
        "32-byte boundary, which is what a lost final-drain write burst would "
        "look like."
    ),
    "prior_samples": "n=1; never retried",
}

# (output name, source build manifest, source workload name, expected ELF sha256)
SOURCES = [
    (
        "xrage-gather-accepted-hardcoded-threepass-2sm",
        ROOT / "runs/hpc-v2-memory-models-1a-v1/build-manifest.json",
        "guarded-profile-xrage-gather",
        "e1e7e130ea151ba02272fd57b0802958ed6350f6c6f3387e350bfb294b93d658",
    ),
    (
        "xrage-gather-parameterized-threepass-2sm",
        ROOT / "runs/xrage-gather-current-three-pass-v1/direct-hashed-manifest.json",
        "direct-hashed",
        "051957ffff4a2351bd3dba655f0d4efcb620c497d714bc834128550619a07244",
    ),
    (
        "xrage-gather-robust-end-drain-only-4sm",
        ROOT / "runs/xrage-gather-robust-end-full-v2/direct-hashed-manifest.json",
        "xrage-gather-4sm",
        "0bd050ff0c3e3a1705fa799edf1dd8c0f29bc4f3483e851d554d34402dd2d60c",
    ),
    # Required filler, never selected.  make_plan() refuses a target with no
    # matching ELF, and load_assets() refuses an asset manifest that leaves any
    # bitstream under asset_root unrepresented -- so the 1SM_DRAM image must
    # have both a target and a 1-SM workload to match it.  This is the 1-SM
    # sibling of the accepted ELF's own device build (same source revision
    # 9682f335, occupancy 8, block-cyclic 1024, full readback).  No job in
    # SELECTION uses it, so nothing in this campaign runs on 1SM_DRAM.
    (
        "xrage-gather-1sm-unselected-filler",
        ROOT / "runs/xrage-gather-blockcyclic1024-nowait-v1/direct-hashed-manifest.json",
        "xrage-gather-blockcyclic1024-nowait--1sm",
        "b2a28613e8d6e7c3c0570bbb4421ec73037968e43d2313f270a55bd91663b9c3",
    ),
]

ACCEPTED = SOURCES[0][0]
PARAMETERIZED = SOURCES[1][0]
COVERAGE = SOURCES[2][0]

# (workload name, target id, repetition, bdf).  Coverage first on each card,
# retries last, because execute_job() is fail-closed per card.
SELECTION = [
    (PARAMETERIZED, HBM, 0, "1a:00.0"),
    (COVERAGE, FOURSM, 0, "1b:00.0"),
    (ACCEPTED, HBM, 0, "1a:00.0"),
    (PARAMETERIZED, HBM, 1, "1b:00.0"),
    (ACCEPTED, HBM, 2, "1a:00.0"),
    (ACCEPTED, HBM, 1, "1b:00.0"),
]
REPETITIONS = 3


def load_campaign():
    spec = importlib.util.spec_from_file_location("radiance_campaign", CAMPAIGN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_build_manifest() -> dict:
    """Compose a FRESH build manifest from three existing, sealed ones.

    Every row is copied verbatim and then re-verified against the ELF on disk,
    so the identity of each artifact is asserted here rather than assumed.  No
    existing manifest is read for anything but its workload rows, and none is
    written.
    """
    rows = []
    provenance = []
    for name, manifest_path, source_name, expected in SOURCES:
        manifest = json.loads(manifest_path.read_text())
        matches = [w for w in manifest["workloads"] if w["name"] == source_name]
        if len(matches) != 1:
            raise SystemExit(
                f"{manifest_path}: {len(matches)} workloads named "
                f"{source_name!r}, expected 1"
            )
        row = dict(matches[0])
        if row["elf_sha256"] != expected:
            raise SystemExit(
                f"{source_name}: manifest records {row['elf_sha256']}, this "
                f"script was written against {expected}"
            )
        elf = Path(row["elf"])
        if not elf.is_file():
            raise SystemExit(f"{source_name}: ELF missing: {elf}")
        on_disk = sha256_file(elf)
        if on_disk != expected:
            raise SystemExit(
                f"{source_name}: ELF on disk hashes {on_disk}, expected {expected}"
            )
        if row.get("family") != "xrage" or row.get("operation") != "gather":
            raise SystemExit(f"{source_name}: not an xRAGE gather workload")
        if not row.get("correctness"):
            raise SystemExit(f"{source_name}: declares no correctness check")
        row["name"] = name
        row["source_build_manifest"] = str(manifest_path)
        row["source_build_manifest_sha256"] = sha256_file(manifest_path)
        row["source_workload_name"] = source_name
        row["source_revision"] = manifest.get("current_source_revision")
        rows.append(row)
        provenance.append({
            "name": name,
            "elf": str(elf),
            "elf_sha256": expected,
            "sm_count": row["sm_count"],
            "loader": row["loader"],
            "correctness": row.get("correctness"),
            "diagnostic_readback": bool(row.get("diagnostic_readback")),
            "host_full_readback": bool(row.get("host_full_readback")),
            "muon_warps_per_core": row.get("muon_warps_per_core"),
            "gather_block_cyclic_elements": row.get("gather_block_cyclic_elements"),
            "robust_end_drain": row.get("robust_end_drain"),
            "robust_serial_fence": row.get("robust_serial_fence"),
            "robust_spin_until_reset": row.get("robust_spin_until_reset"),
            "robust_post_spin": row.get("robust_post_spin"),
            "robust_drain_passes": row.get("robust_drain_passes"),
            "source_build_manifest": str(manifest_path),
            "source_revision": manifest.get("current_source_revision"),
        })

    return {
        "schema": "radiance-xrage-gather-threepass-retry-build-v1",
        "hardware_rebuilt": False,
        "no_vpu": True,
        "scope": (
            "three already-built xRAGE gather artifacts, composed into one "
            "manifest so a single plan can address them; nothing is compiled"
        ),
        "artifact_identity": provenance,
        "workloads": rows,
    }, provenance


def build_assets() -> dict:
    spec = json.loads(SOURCE_ASSETS.read_text())
    # single_fpga_bdf would reject any target not on 1a:00.0.  Replace it with
    # an explicit allowlist so the constraint stays visible; per-job assignment
    # below is what actually decides the card.
    spec.pop("single_fpga_bdf", None)
    spec["allowed_fpga_bdfs"] = list(BDFS)
    profiling = spec.get("profiling")
    if isinstance(profiling, dict):
        profiling.pop("single_fpga_bdf", None)
        profiling["allowed_fpga_bdfs"] = list(BDFS)
    for target in spec["targets"]:
        if target.get("kind") == "firesim":
            target["bdf"] = BDFS[0]          # advisory; per-job assignment wins
    spec["dual_bdf_rationale"] = (
        "Per-job card assignment.  Programming is host-wide serialized by "
        "scripts/hostwide_flash_lock.py; steady-state simulation overlaps.  "
        "Bitstreams, drivers and runtime scripts are the sealed v1-hardened "
        "copies, unchanged and still hash-verified at execute time."
    )
    spec["source"] = (
        spec.get("source", "")
        + " | dual-BDF split prepared 2026-11-03 for the xRAGE gather "
          "three-pass follow-up"
    ).strip(" |")
    return spec


def preflight(campaign, plan: dict) -> dict:
    """Repeat, read-only, every hash check execute() makes before it runs."""
    problems: list[str] = []
    if sha256_file(Path(plan["build_manifest"])) != plan["build_manifest_sha256"]:
        problems.append("build manifest hash does not match the plan")
    if sha256_file(Path(plan["asset_manifest"])) != plan["asset_manifest_sha256"]:
        problems.append("asset manifest hash does not match the plan")
    asset_root = Path(plan["asset_root"])
    for label, receipt in sorted(plan["asset_receipts"].items()):
        path = campaign.inside(asset_root, receipt["path"], label)
        if path.stat().st_size != receipt["bytes"]:
            problems.append(f"{label}: size changed")
        elif sha256_file(path) != receipt["sha256"]:
            problems.append(f"{label}: sha256 changed")
    for job in plan["jobs"]:
        elf = Path(job["workload"]["elf"])
        if not elf.is_file():
            problems.append(f"{job['id']}: ELF missing")
        elif sha256_file(elf) != job["workload"]["elf_sha256"]:
            problems.append(f"{job['id']}: ELF hash changed")
    return {
        "schema": "radiance-plan-preflight-v1",
        "receipts_rehashed": len(plan["asset_receipts"]),
        "elfs_rehashed": len(plan["jobs"]),
        "build_manifest": plan["build_manifest"],
        "asset_manifest": plan["asset_manifest"],
        "problems": problems,
        "status": "pass" if not problems else "fail",
    }


def main() -> None:
    campaign = load_campaign()
    if (OUT / "plan.json").exists():
        raise SystemExit(
            f"{OUT / 'plan.json'} already exists; this campaign builds into a "
            "fresh directory and never rewrites a sealed plan"
        )
    OUT.mkdir(parents=True, exist_ok=True)

    build, provenance = build_build_manifest()
    BUILD_MANIFEST.write_text(json.dumps(build, indent=1, sort_keys=True) + "\n")
    ASSETS.write_text(json.dumps(build_assets(), indent=1, sort_keys=True) + "\n")

    plan = campaign.make_plan(
        build_path=BUILD_MANIFEST,
        asset_path=ASSETS,
        asset_root=ASSET_ROOT,
        out=OUT,
        warmups=0,
        repetitions=REPETITIONS,
        allow_incomplete=True,
    )

    available = {
        (job["workload"]["name"], job["target_id"], job["repetition"]): job
        for job in plan["jobs"]
    }
    jobs = []
    for name, target_id, repetition, bdf in SELECTION:
        key = (name, target_id, repetition)
        job = available.get(key)
        if job is None:
            raise SystemExit(f"plan has no job for {key}")
        job = json.loads(json.dumps(job))        # deep copy
        target = dict(job["target"])
        target["bdf"] = bdf
        job["target"] = target
        if target.get("require_per_sm_timing") is not True:
            raise SystemExit(f"{job['id']}: require_per_sm_timing is not true")
        if not job["workload"].get("correctness"):
            raise SystemExit(f"{job['id']}: workload declares no correctness check")
        jobs.append(job)

    if len(jobs) != len(SELECTION):
        raise SystemExit(f"selected {len(jobs)} jobs, expected {len(SELECTION)}")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise SystemExit("duplicate job ids")
    if len({job["result_dir"] for job in jobs}) != len(jobs):
        raise SystemExit("duplicate result dirs")

    # Fail-closed ordering contract: on each card, every job that is NOT a
    # retry of the known-failing cell must precede every job that is.
    for bdf in BDFS:
        on_card = [job for job in jobs if job["target"]["bdf"] == bdf]
        retry_from = next(
            (index for index, job in enumerate(on_card)
             if job["workload"]["name"] == ACCEPTED), len(on_card))
        if any(job["workload"]["name"] != ACCEPTED
               for job in on_card[retry_from:]):
            raise SystemExit(
                f"{bdf}: a non-retry job is scheduled after a retry of the "
                "known-failing cell; a SafetyAbort would drop it"
            )

    plan["jobs"] = jobs
    plan["failure_under_investigation"] = FAILURE
    plan["artifact_identity"] = provenance
    plan["three_pass_artifact_finding"] = (
        "There are TWO three-pass gather builds and they are different "
        "artifacts.  e1e7e130ea (ACCEPTED, in production, the ELF that failed) "
        "gets its three drain passes HARD-CODED from source revision "
        "9682f33537aa7fa95cc82715c188de5ebf4fc83c; it exposes no "
        "robust_drain_passes field and its robust_post_spin is 0.  "
        "051957ffff (runs/xrage-gather-current-three-pass-v1) is the "
        "PARAMETERIZED fix: --robust-drain-passes 3 with --robust-post-spin "
        "20000, built at a0142f07524321f70fa28e4f5fa087b56e0a101d, validated "
        "locally against Cyclotron over all 16,737,936 output words with a "
        "one-bit negative control, and never executed on hardware.  A search of "
        "every build manifest under runs/ for robust_drain_passes == 3 returns "
        "only 051957ffff and its logical shadow d68cf428f8, so the accepted ELF "
        "is three-pass in behaviour but is NOT the parameterized artifact."
    )
    plan["job_order_rationale"] = (
        "campaign.execute_job() is fail-closed: a failed job raises SafetyAbort "
        "and ends that CARD's group.  One of these six jobs exists because it "
        "may fail, so the two coverage jobs run FIRST on their cards and the "
        "retries of the known-failing cell run LAST.  A failure of accepted "
        "rep 0 on 1a costs only accepted rep 2; the fix measurement, the "
        "4SM_DRAM coverage job and accepted rep 1 on 1b all still run.  BDFs "
        "are assigned per job, as in "
        "scripts/prepare_roofline_dual_bdf_plan.py, so the retried cell is "
        "sampled on BOTH cards and a reproduction cannot be a one-card artefact."
    )
    plan["replication_rationale"] = (
        "n=3 on the failing cell, which is the point of the run: the failure is "
        "n=1 and has never been retried, so nothing yet distinguishes a one-off "
        "from a reproducible defect.  This host is otherwise bit-deterministic "
        "for a fixed (bitstream, ELF, plusargs) triple -- 29 of 31 replicate "
        "groups returned identical kernel_cycles, CV=0 -- which is exactly why "
        "three identical passing repetitions would be informative and three "
        "identical failures would be conclusive.  n=2 on the parameterized fix "
        "over the same cell; n=1 on the coverage cell, which is a coverage "
        "question and not a reliability one."
    )
    plan["coverage_cell_caveat"] = (
        "xrage-gather-robust-end-drain-only-4sm is 0bd050ff0c from "
        "runs/xrage-gather-robust-end-full-v2: robust_end_drain ONLY, occupancy "
        "4, no block-cyclic schedule, digest-checked WITHOUT "
        "diagnostic_readback or host_full_readback.  It is included because "
        "firesim-4sm-dram--xrage-gather-4sm--measure-0 is the only gather cell "
        "on 4SM_DRAM that has never produced a result, so it is the coverage "
        "gap.  It is NOT a test of either three-pass artifact, and a failure "
        "there would report a digest mismatch with no output index.  The "
        "accepted family's own 4-SM sibling (4dbf9a2f57, same source 9682f335, "
        "same flags) already ran on 4SM_DRAM and passed at 50,377,247 cycles, "
        "and neither three-pass artifact has a 4-SM build; none was created "
        "for this plan."
    )
    plan["resume_guidance"] = (
        "If a job fails, the campaign ends that card's group and the remaining "
        "jobs on that card are not attempted.  Two ways to finish the run, and "
        "they produce different bookkeeping:\n"
        "  (1) `--include-job <job id>` WITHOUT --rerun-failed runs only the "
        "      dropped repetition and leaves the failed cell's result.json in "
        "      place, so each repetition keeps its own result and "
        "      campaign.py summarize counts them separately.  Prefer this.\n"
        "  (2) re-invoking the whole plan with --rerun-failed re-runs the "
        "      failed job in the SAME result directory.  No evidence is lost -- "
        "      archive_prior_attempt() moves every prior artifact, logs and "
        "      memory-profile tree included, to attempts/attempt-NNN/ first -- "
        "      but result.json then holds only the LATEST attempt, so "
        "      summarize() under-counts the repetitions actually executed.  "
        "      Read attempts/ before reporting n for this cell.\n"
        "Either way, the question this plan answers is how many of the "
        "attempts on the accepted ELF reproduced the four-zero shape at "
        "output_index 5,632,092-5,632,095, so count attempts, not result files."
    )
    plan["overnight_driver_note"] = (
        "runs/takeover-2026-11-02/overnight.sh polls for this plan by path and "
        "launches it with --parallel-targets --rerun-failed, re-checking each "
        "pass until 07:00.  That is resume mode (2) above: a failing accepted-"
        "ELF cell will be retried on every pass, each attempt preserved under "
        "attempts/attempt-NNN/, with result.json holding the last one.  The "
        "coverage-first ordering keeps that retry loop from starving the "
        "4SM_DRAM job and the parameterized-fix jobs, which complete on the "
        "first pass."
    )
    plan["live_campaign_note"] = (
        "runs/xrage-gather-robust-dual-1a1b-v1 was executing when this plan was "
        "prepared and may itself produce a result for its own 4SM_DRAM job.  "
        "That job writes into its own result directory; this plan reads only "
        "the ELF path from it and writes nothing there.  If that cell completes "
        "first, the coverage job here becomes a second independent sample."
    )
    plan["no_bitstream_rebuild"] = (
        "Every ELF already existed and was verified by hash at prepare time.  "
        "Same v1-hardened bitstreams, drivers and runtime scripts as the "
        "campaign that recorded the failure.  Nothing is built."
    )
    campaign.write_json(OUT / "plan.json", plan)

    check = preflight(campaign, json.loads((OUT / "plan.json").read_text()))
    (OUT / "preflight.json").write_text(
        json.dumps(check, indent=1, sort_keys=True) + "\n")
    if check["status"] != "pass":
        raise SystemExit("preflight failed:\n  " + "\n  ".join(check["problems"]))

    by_bdf: dict[str, list[str]] = {}
    for job in jobs:
        by_bdf.setdefault(job["target"]["bdf"], []).append(job["id"])
    summary = {
        "schema": "radiance-xrage-gather-threepass-retry-prepare-v1",
        "plan": str(OUT / "plan.json"),
        "jobs": len(jobs),
        "jobs_per_bdf": {k: len(v) for k, v in sorted(by_bdf.items())},
        "card_order": {k: v for k, v in sorted(by_bdf.items())},
        "preflight": check["status"],
        "artifact_identity": provenance,
        "order": [f"{job['id']} @{job['target']['bdf']}" for job in jobs],
        "not_run": "this script prepares the plan only; it never executes a job",
    }
    (OUT / "prepare-summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "artifact_identity"}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
