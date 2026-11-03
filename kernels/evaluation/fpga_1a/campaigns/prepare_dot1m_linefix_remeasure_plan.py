#!/usr/bin/env python3
"""Plan the dot-1m (stream_dot) hardware re-measurement with the line fix in.

Why this campaign exists
------------------------
`stream_dot_partial` used to be a 16-byte, 64-byte-aligned array holding one
float32 per SM, written as `stream_dot_partial[block_id]`.  Every SM's partial
therefore lived in ONE 64-byte line: at 4 SM, SM0/1/2/3 wrote bytes 0/4/8/12 of
the line at 0x10805140.  Commit 46bc878 ("dot: give each SM its own 64-byte line
for its reduction partial") pads the array to one line per SM and indexes it at
a 16-float stride.

Every existing dot-1m FPGA result was measured with the pre-fix device ELF.
Those results are not wrong and nothing about them is deleted or edited here:
they remain valid evidence for the ELF that actually ran.  They just no longer
describe the source.  This plan produces the hardware evidence for the ELF the
source now builds.

What is NOT claimed
-------------------
`accepted_device_byte_exact` cannot hold against the old accepted payload: the
data layout legitimately changed, so the device ELF legitimately changed.  The
correctness contract is the kernel's own full host check --
`exact-device-order-partials-host-final-scalar-and-guards` -- and that is
unchanged: the expected partial words, the FNV digest and the ordered final
scalar are bit-identical at 1, 2 and 4 SM.

Read-only with respect to /scratch/nicolas.  Creating this plan touches no FPGA
and launches no hardware job.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path("/scratch/agustin/radiance-fpga-1a")
CAMPAIGN = ROOT / "src/radiance-kernels/kernels/evaluation/fpga_1a/campaign.py"

OUT = ROOT / "runs/dot1m-linefix-remeasure-v1"
BUILD_MANIFEST = OUT / "build/direct-hashed-manifest.json"

# The same hardened copy of the legacy v1 bitstream/runtime set that
# scripts/prepare_roofline_dual_bdf_plan.py uses.  Its 1SM_DRAM / 2SM_DRAM /
# 4SM_DRAM bitstreams and drivers are byte-identical to the ones the accepted
# dot-1m results were measured on (artifacts/nicolas-fpga-assets-v1); only the
# runner is hardened.
ASSET_ROOT = ROOT / "artifacts/nicolas-fpga-assets-v1-hardened"
SOURCE_ASSETS = ASSET_ROOT / "assets.json"
DUAL_ASSETS = OUT / "assets-dual-bdf.json"

BDFS = ["1a:00.0", "1b:00.0"]
REPETITIONS = 3

# The three cells that let the fix be compared against the existing dot-1m
# record one for one.  The asset manifest must keep every target it had -- the
# campaign loader refuses an assets.json that leaves a bitstream in the asset
# root unrepresented -- so the 2SM_DRAM_baseline and 2SM_HBM cells are dropped
# here, after planning, rather than from the manifest.
SELECTED_TARGETS = ("firesim-4sm-dram", "firesim-2sm-dram", "firesim-1sm-dram")

FIX_COMMIT = "46bc878eb6d063c7942c6fe7c1e53218fed8d052"

PRIOR_RESULTS = {
    "firesim-1sm-dram": (
        ROOT / "runs/fpga-stream-dot-babelstream-f32-1m-bank-blockcyclic2048-"
               "candidate-1a-v1/jobs/firesim-1sm-dram--stream-dot-babelstream-"
               "f32-1m-bank-blockcyclic2048-spin-post20k--1sm--measure-0/"
               "result.json"),
    "firesim-2sm-dram": (
        ROOT / "runs/fpga-stream-dot-babelstream-f32-1m-bank-blockcyclic2048-"
               "candidate-1a-v1/jobs/firesim-2sm-dram--stream-dot-babelstream-"
               "f32-1m-bank-blockcyclic2048-spin-post20k--2sm--measure-0/"
               "result.json"),
    "firesim-4sm-dram": (
        ROOT / "runs/fpga-stream-dot-babelstream-f32-1m-bank-blockcyclic2048-"
               "candidate-1a-v1/jobs/firesim-4sm-dram--stream-dot-babelstream-"
               "f32-1m-bank-blockcyclic2048-spin-post20k--4sm--measure-0/"
               "result.json"),
}


def load_campaign():
    spec = importlib.util.spec_from_file_location("radiance_campaign", CAMPAIGN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_dual_assets() -> dict:
    spec = json.loads(SOURCE_ASSETS.read_text())
    spec.pop("single_fpga_bdf", None)
    spec["allowed_fpga_bdfs"] = list(BDFS)
    profiling = spec.get("profiling")
    if isinstance(profiling, dict):
        profiling.pop("single_fpga_bdf", None)
        profiling["allowed_fpga_bdfs"] = list(BDFS)
    spec["dual_bdf_rationale"] = (
        "Per-job BDF assignment. Programming is host-wide serialized by "
        "scripts/hostwide_flash_lock.py; steady-state simulation overlaps."
    )
    # Target-level BDF is advisory only - the per-job assignment below wins.
    for target in spec["targets"]:
        if target.get("kind") == "firesim":
            target["bdf"] = BDFS[0]
    return spec


def prior_cycles(path: Path) -> int | None:
    try:
        return json.loads(path.read_text()).get("kernel_cycles")
    except OSError:
        return None


def main() -> None:
    campaign = load_campaign()
    OUT.mkdir(parents=True, exist_ok=True)

    spec = build_dual_assets()
    DUAL_ASSETS.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")

    plan = campaign.make_plan(
        build_path=BUILD_MANIFEST,
        asset_path=DUAL_ASSETS,
        asset_root=ASSET_ROOT,
        out=OUT,
        warmups=0,
        repetitions=REPETITIONS,
        allow_incomplete=True,
    )

    jobs = [j for j in plan["jobs"] if j["target_id"] in SELECTED_TARGETS]
    if len(jobs) != len(SELECTED_TARGETS) * REPETITIONS:
        raise SystemExit(f"expected {len(SELECTED_TARGETS) * REPETITIONS} jobs, "
                         f"got {len(jobs)}")

    # Ordering.  4 SM is the SM count at which the defect actually manifested
    # -- four SMs writing bytes 0/4/8/12 of one line -- so it is the most
    # valuable point if the run is truncated, and it goes first.  Within that,
    # one full 4/2/1 SM round is completed before any repetition is added, so a
    # truncation after three jobs still yields the complete 1/2/4 SM comparison
    # at n=1, which is exactly the sample size of the record being superseded.
    jobs.sort(key=lambda j: (j["repetition"], -j["target"]["sm_count"]))

    # Alternate cards so both stay busy and a truncated run leaves a balanced
    # result set. Alternation is global so an early stop cannot leave one card
    # systematically ahead.
    for index, job in enumerate(jobs):
        target = dict(job["target"])
        target["bdf"] = BDFS[index % len(BDFS)]
        job["target"] = target

    # Per-job provenance: what each cell supersedes, and what may not be claimed.
    for job in jobs:
        workload = dict(job["workload"])
        prior = PRIOR_RESULTS[job["target_id"]]
        workload["supersedes_result"] = str(prior)
        workload["supersedes_kernel_cycles"] = prior_cycles(prior)
        workload["accepted_device_byte_exact"] = False
        workload["accepted_device_byte_exact_rationale"] = (
            f"The device ELF legitimately changed: commit {FIX_COMMIT[:7]} pads "
            "stream_dot_partial to one 64-byte line per SM and indexes it at a "
            "16-float stride, so the data layout and the guard addresses move. "
            "Byte-exactness against the old accepted payload is therefore not "
            "the right contract. Correctness is the kernel's own full host "
            "check - every per-SM partial word, the FNV digest, the ordered "
            "final scalar and both guard bands - and its expected values are "
            "bit-identical to the accepted ones."
        )
        workload["line_fix_commit"] = FIX_COMMIT
        job["workload"] = workload

    plan["jobs"] = jobs
    plan["job_order_rationale"] = (
        "These measurements SUPERSEDE the pre-fix dot-1m measurements at "
        "1SM_DRAM / 2SM_DRAM / 4SM_DRAM (runs/fpga-stream-dot-babelstream-f32-"
        "1m-bank-blockcyclic2048-candidate-1a-v1/jobs/, and the profiled reruns "
        "of the same ELFs in runs/fpga-hpc-bottleneck-medium-profile-1a-v1/). "
        f"Commit {FIX_COMMIT} ('dot: give each SM its own 64-byte line for its "
        "reduction partial') pads stream_dot_partial from a single shared "
        "64-byte line to one line per SM, so the pre-fix results were measured "
        "with a device ELF that the current source no longer produces. "
        "Nothing existing is deleted, edited or invalidated: those result.json "
        "files remain valid evidence for the ELF that actually ran. "
        "accepted_device_byte_exact CANNOT hold against the old accepted "
        "payload, and is recorded false on every job here: the data layout "
        "legitimately changed, which is the whole point of the fix. The "
        "correctness contract is the kernel's own full host check - "
        "'exact-device-order-partials-host-final-scalar-and-guards' - whose "
        "expected digests are UNCHANGED (1 SM 0f359ab682d35266, 2 SM "
        "c8eb6d25dcfd422d, 4 SM 025818571ba57725), so a PASS here is the same "
        "numerical statement the accepted results made. "
        "ORDER: 4 SM first. Four SMs writing bytes 0/4/8/12 of one line is "
        "where the defect actually manifested, so 4SM_DRAM is the most "
        "valuable point if the run is truncated. One complete 4/2/1 SM round "
        "runs before any repetition is added, so stopping after three jobs "
        "still yields the full 1/2/4 SM comparison at n=1 - the same sample "
        "size as the record being superseded. Jobs alternate 1a:00.0 / "
        "1b:00.0; campaign.py groups FireSim jobs by card and runs the two "
        "groups concurrently, while programming is serialized host-wide."
    )
    plan["supersedes"] = {
        "commit": FIX_COMMIT,
        "reason": "stream_dot_partial cache-line false-sharing fix; the device "
                  "ELF and the guard addresses change, so the pre-fix dot-1m "
                  "hardware results no longer describe the source.",
        "prior_results": {k: str(v) for k, v in PRIOR_RESULTS.items()},
        "prior_results_remain_valid": True,
        "accepted_device_byte_exact": False,
    }
    plan["excluded_cells_note"] = (
        "2SM_DRAM_v2 and 2SM_HBM_v2 are NOT in this plan. They live in a "
        "different asset root (artifacts/nicolas-fpga-assets-v2-candidates) "
        "and a campaign plan carries exactly one asset_root, so they cannot "
        "share a plan with the 1/2/4 SM v1 DRAM cells. The hpc-v2 dot-1m pair "
        "is also not reproducible from committed source: it ran "
        "runs/hpc-profile-guard-builds-v1/dot-1m/2sm/"
        "kernel.accepted-device-profile-host.soc.elf, a host built with "
        "-DNIGHTLY_PROFILE_GUARD_CYCLES=1000000ULL from an UNCOMMITTED local "
        "patch to lib/include/nightly/host.h in "
        "worktrees/hpc-profile-revisions-v1/ff2b613bcfc6. Re-measuring those "
        "two cells needs that patch committed first; a plain host would not be "
        "comparable to the hpc-v2 record. The 2SM_DRAM_baseline and 2SM_HBM v1 "
        "cells are omitted for budget, not for correctness."
    )
    plan["host_verification"] = str(OUT / "verify")
    campaign.write_json(OUT / "plan.json", plan)

    summary = {
        "schema": "radiance-dot1m-linefix-remeasure-prepare-v1",
        "plan": str(OUT / "plan.json"),
        "build_manifest": str(BUILD_MANIFEST),
        "asset_root": str(ASSET_ROOT),
        "jobs": len(jobs),
        "repetitions": REPETITIONS,
        "cells": len(SELECTED_TARGETS),
        "job_order": [f"{j['id']} @{j['target']['bdf']}" for j in jobs],
        "jobs_per_bdf": {
            bdf: sum(1 for j in jobs if j["target"]["bdf"] == bdf)
            for bdf in BDFS
        },
    }
    (OUT / "prepare-summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
