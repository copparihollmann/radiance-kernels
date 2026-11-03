#!/usr/bin/env python3
"""Prepare the missing TUNED-HBM arm of the compute-instrumented v2 campaign.

WHAT IS ALREADY MEASURED

  runs/hpc-v2-compute-instrumented-v1   42/42 passed.  The same 21 accepted HPC
                                        kernels, re-hosted with the
                                        compute-instrumentation commit
                                        a8b6bcdd70f8412d9d4e64f057825746320e62da,
                                        on 2SM_DRAM_v2 and 2SM_HBM_v2 with the
                                        FASED HARDWARE DEFAULTS (untuned).
                                        Device-reported useful_ops,
                                        executed_ops, peak_ops_per_cycle,
                                        compute_util_bp, compute_window_cycles
                                        and the per-kernel detail.

  runs/hpc-v2-hbm-tuned-1a1b-v1         21/21 passed.  2SM_HBM_v2 with the TUNED
                                        JEDEC 2000 MT/s profile.  Cycles only --
                                        built on the ACCEPTED host ELFs, which
                                        carry no compute counters.

WHAT IS MISSING, AND WHY IT IS THE INTERESTING CELL

Tuning recovered ~13-16 points of cycles on HBM2.  The FASED side already says
HOW: per-request SERVICE TIME fell.  Bytes are unchanged on 19 of 21 kernels,
occupancy is DOWN on 21 of 21, and implied latency drops 65.4 -> 49.2 cycles.
What no measurement yet says is what that did to COMPUTE utilisation -- whether
the recovered cycles show up as the SMs actually issuing more useful work per
cycle, or merely as the same work finishing sooner.  Answering that needs the
compute-instrumented host ELFs run against the tuned memory model, which is
exactly the one arm of the 2x2 (untuned/tuned x DRAM/HBM) that does not exist.

WHAT THIS PLAN IS

  21 jobs.  2SM_HBM_v2 only.  The 21 compute-instrumented host ELFs reused
  BYTE-FOR-BYTE from runs/hpc-v2-compute-instrumented-v1 -- the build manifest
  there is consumed as-is and nothing is rebuilt -- driven with the TUNED
  plusarg vector lifted verbatim out of runs/hpc-v2-hbm-tuned-1a1b-v1/plan.json.

CREDIT FOR THE TUNED VECTOR

The tuned profile is NOT this campaign's work.  It is the HARDWARE OWNER's
own HBM2 2000 MT/s profile, recovered from the raw_output of his run 800:
43 explicit FASED runtime configuration registers written to each of the 5
memory-model bridges (215 `+mm_<reg>_<bridge>=<value>` settings) plus the 5
`+mm_useHardwareDefaultRuntimeSettings_<bridge>` opt-in flags, 220 plusargs in
all.  This script copies that string verbatim -- it never re-derives, reorders
or re-serialises it -- and asserts the copy is byte-identical to the anchor's
before planning.  Any claim this campaign makes about tuned HBM2 rests on his
profile and must say so.

ORDERING

The three ablation kernels run FIRST:

    stream-triad   spatter-0   spmv-jds

They are the kernels that runs/hbm-tuning-ablation-v1 decomposes with the full
leave-one-out / leave-one-in register-group ablation, so they are the only three
for which a compute-utilisation number lands in an already-decomposed context:
the ablation says which register group bought the cycles, and these jobs say
what the compute pipeline did while it happened.  A truncated run must not lose
them.  The remaining 18 keep the tuned anchor's own information order so the
tail of this run is directly comparable to the tail of that one.

BDFs alternate 1a:00.0 / 1b:00.0 per job, exactly as
scripts/prepare_roofline_dual_bdf_plan.py does.  Alternation is global rather
than per-phase, so an early stop cannot leave one card systematically ahead,
and no kernel's result can be an artefact of one card.

n=1.  This host is bit-deterministic for a fixed (bitstream, ELF, plusargs)
triple: 29 of 31 replicate groups returned identical kernel_cycles, CV=0.

SAFETY

Runs nothing.  Touches no FPGA, no /dev/xdma*, no /sys/bus/pci.  Read-only with
respect to /scratch/nicolas and to every existing result.json, plan.json, asset
manifest and build manifest.  Writes only into
runs/hpc-v2-compute-instrumented-tuned-hbm-v1, which must not already exist as a
populated run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path("/scratch/agustin/radiance-fpga-1a")
CAMPAIGN = ROOT / "src/radiance-kernels/kernels/evaluation/fpga_1a/campaign.py"
ASSET_ROOT = ROOT / "artifacts/nicolas-fpga-assets-v2-candidates"

# Reused verbatim.  Not copied, not rewritten: make_plan hashes this file into
# the plan and execute() re-hashes it fail-closed, so the plan is bound to the
# same 21 instrumented ELFs the measured untuned arm used.
BUILD_MANIFEST = ROOT / "runs/hpc-v2-compute-instrumented-v1/build-manifest.json"
SOURCE_ASSETS = ROOT / "runs/hpc-v2-compute-instrumented-v1/assets-dual-bdf.json"
UNTUNED_PLAN = ROOT / "runs/hpc-v2-compute-instrumented-v1/plan.json"
TUNED_PLAN = ROOT / "runs/hpc-v2-hbm-tuned-1a1b-v1/plan.json"

OUT = ROOT / "runs/hpc-v2-compute-instrumented-tuned-hbm-v1"
ASSETS = OUT / "assets-tuned-hbm.json"

TARGET = "firesim-2sm-hbm-v2-quadchannel"
DRAM_TARGET = "firesim-2sm-dram-v2-quadrank"
BDFS = ["1a:00.0", "1b:00.0"]
BRIDGES = 5

# The three kernels runs/hbm-tuning-ablation-v1 decomposes register-group by
# register-group.  They lead the order so a truncated run keeps them.
ABLATION_KERNELS = [
    "guarded-profile-stream-triad",
    "guarded-profile-spatter-0",
    "guarded-profile-spmv-jds",
]

EXPECTED_WORKLOADS = 21
EXPECTED_PLUSARGS = 220
EXPECTED_REGISTERS = 43
TUNED_VECTOR_SHA256 = (
    "0b10c885df165bf6b56b11170762869af02e350bf59fdd1aa393d3b8872c79a0"
)

COMPUTE_COUNTERS = [
    "useful_ops", "executed_ops", "peak_ops_per_cycle",
    "compute_util_bp", "compute_window_cycles",
]


def load_campaign():
    spec = importlib.util.spec_from_file_location("radiance_campaign", CAMPAIGN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_owner_tuned_vector() -> tuple[str, dict[str, str], list[str]]:
    """Lift the owner's tuned plusarg string out of the tuned anchor's plan.

    Returned verbatim.  The structural assertions below only DESCRIBE it; the
    string that goes into this plan is the same object, never a rebuild of it.
    """
    plan = json.loads(TUNED_PLAN.read_text())
    vectors = {job["target"]["env"]["EXTRA_PLUSARGS"] for job in plan["jobs"]}
    if len(vectors) != 1:
        raise SystemExit(
            f"{TUNED_PLAN} carries {len(vectors)} distinct EXTRA_PLUSARGS "
            "vectors; the tuned arm assumes exactly one"
        )
    vector = next(iter(vectors))
    tokens = vector.split()
    if len(tokens) != EXPECTED_PLUSARGS:
        raise SystemExit(
            f"tuned vector has {len(tokens)} plusargs, expected "
            f"{EXPECTED_PLUSARGS}"
        )

    settings: dict[str, dict[int, str]] = {}
    flags: list[str] = []
    for token in tokens:
        match = re.fullmatch(
            r"\+mm_(?P<reg>.+)_(?P<bridge>\d+)=(?P<value>.+)", token)
        if match:
            settings.setdefault(match["reg"], {})[int(match["bridge"])] = \
                match["value"]
        elif re.fullmatch(r"\+mm_useHardwareDefaultRuntimeSettings_\d+", token):
            flags.append(token)
        else:
            raise SystemExit(f"unrecognised tuned plusarg: {token!r}")

    if len(flags) != BRIDGES:
        raise SystemExit(
            f"tuned vector carries {len(flags)} hardware-default opt-in flags, "
            f"expected {BRIDGES}; without them any register the command line "
            "does not set has nothing to fall back to and the FASED driver "
            "aborts at start-up"
        )
    if len(settings) != EXPECTED_REGISTERS:
        raise SystemExit(
            f"tuned vector sets {len(settings)} registers, expected "
            f"{EXPECTED_REGISTERS}"
        )
    flat: dict[str, str] = {}
    for register, by_bridge in sorted(settings.items()):
        if sorted(by_bridge) != list(range(BRIDGES)):
            raise SystemExit(
                f"{register}: set on bridges {sorted(by_bridge)}, expected "
                f"0-{BRIDGES - 1}"
            )
        values = set(by_bridge.values())
        if len(values) != 1:
            raise SystemExit(
                f"{register}: value differs across bridges: {sorted(values)}"
            )
        flat[register] = values.pop()

    if sha256_text(vector) != TUNED_VECTOR_SHA256:
        raise SystemExit(
            "the tuned vector in the anchor plan no longer hashes to the "
            f"profile this script was written against:\n  anchor  "
            f"{sha256_text(vector)}\n  expected {TUNED_VECTOR_SHA256}\n"
            "Re-read the owner's run 800 before planning."
        )
    return vector, flat, flags


def build_assets(vector: str) -> dict:
    """The compute-instrumented asset manifest with the HBM arm tuned.

    Both v2 targets stay present: load_assets() refuses a manifest that leaves
    any bitstream under asset_root unrepresented.  The DRAM_v2 target is carried
    unchanged -- still on hardware defaults -- and every job it would generate is
    dropped below, so nothing in this run touches it.
    """
    spec = json.loads(SOURCE_ASSETS.read_text())
    seen = []
    for target in spec["targets"]:
        if target["id"] != TARGET:
            continue
        env = target.setdefault("env", {})
        before = env.get("EXTRA_PLUSARGS")
        if before is None:
            raise SystemExit(f"{TARGET}: source manifest has no EXTRA_PLUSARGS")
        env["EXTRA_PLUSARGS"] = vector
        target["memory_model_runtime_settings"] = (
            "tuned JEDEC HBM2 2000 MT/s, 43 registers x 5 bridges, lifted "
            "verbatim from the hardware owner's run 800 "
            "(runs/hpc-v2-hbm-tuned-1a1b-v1/plan.json)"
        )
        target["tuned_profile_credit"] = (
            "The 43-register HBM2 2000 MT/s runtime profile is the hardware "
            "owner's, recovered from his run 800 raw_output.  This campaign "
            "copies it byte-for-byte and claims no part of its derivation."
        )
        target["untuned_extra_plusargs"] = before
        seen.append(target["id"])
    if seen != [TARGET]:
        raise SystemExit(f"expected to tune exactly {TARGET}, tuned {seen}")

    dram = [t for t in spec["targets"] if t["id"] == DRAM_TARGET]
    if len(dram) != 1:
        raise SystemExit(f"expected exactly one {DRAM_TARGET} target")
    if "useHardwareDefaultRuntimeSettings" not in \
            dram[0].get("env", {}).get("EXTRA_PLUSARGS", ""):
        raise SystemExit(
            f"{DRAM_TARGET} is no longer on hardware defaults; this run must "
            "not silently change the DDR3 comparison baseline"
        )

    spec["source"] = (
        spec.get("source", "")
        + " | tuned-HBM compute arm prepared 2026-11-03: owner's 43-register "
          "HBM2 2000 MT/s profile applied to the HBM_v2 target only"
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

    vector, registers, flags = read_owner_tuned_vector()
    spec = build_assets(vector)
    ASSETS.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")

    plan = campaign.make_plan(
        build_path=BUILD_MANIFEST,
        asset_path=ASSETS,
        asset_root=ASSET_ROOT,
        out=OUT,
        warmups=0,
        repetitions=1,
        allow_incomplete=True,
    )

    # Keep only the HBM_v2 arm.  The DRAM_v2 half of the 2x2 is already measured
    # at runs/hpc-v2-compute-instrumented-v1 and is not re-run here.
    jobs = [job for job in plan["jobs"] if job["target_id"] == TARGET]
    if len(jobs) != EXPECTED_WORKLOADS:
        raise SystemExit(
            f"selected {len(jobs)} {TARGET} jobs, expected {EXPECTED_WORKLOADS}"
        )
    for job in jobs:
        if job["target"]["env"]["EXTRA_PLUSARGS"] != vector:
            raise SystemExit(f"{job['id']}: tuned vector did not reach the job")
        if job["target"].get("require_per_sm_timing") is not True:
            raise SystemExit(f"{job['id']}: require_per_sm_timing is not true")
        if not job["workload"].get("compute_instrumented_host"):
            raise SystemExit(
                f"{job['id']}: host ELF is not the compute-instrumented build"
            )
        if not job["workload"].get("correctness"):
            raise SystemExit(f"{job['id']}: workload declares no correctness check")

    # The 21 kernels must be exactly the 21 the untuned instrumented arm ran,
    # so the two arms are the same cells and nothing silently dropped out.
    untuned = json.loads(UNTUNED_PLAN.read_text())
    untuned_names = sorted({
        job["workload"]["name"] for job in untuned["jobs"]
        if job["target_id"] == TARGET
    })
    if sorted(job["workload"]["name"] for job in jobs) != untuned_names:
        raise SystemExit(
            "the tuned arm's kernels differ from the untuned instrumented arm's"
        )
    untuned_elf = {
        job["workload"]["name"]: job["workload"]["elf_sha256"]
        for job in untuned["jobs"] if job["target_id"] == TARGET
    }
    reused = []
    for job in jobs:
        name = job["workload"]["name"]
        if job["workload"]["elf_sha256"] != untuned_elf[name]:
            raise SystemExit(
                f"{name}: host ELF differs from the untuned instrumented arm; "
                "this run must reuse those ELFs, not rebuild them"
            )
        reused.append(name)

    # Order: the three decomposed ablation kernels first, then the tuned
    # anchor's own information order for the remaining 18.
    anchor_order = [
        job["workload"]["name"] for job in json.loads(TUNED_PLAN.read_text())["jobs"]
    ]
    missing_in_anchor = [n for n in untuned_names if n not in anchor_order]
    if missing_in_anchor:
        raise SystemExit(
            f"tuned anchor does not order these kernels: {missing_in_anchor}"
        )
    unknown_head = [n for n in ABLATION_KERNELS if n not in untuned_names]
    if unknown_head:
        raise SystemExit(f"ablation kernels not in this build: {unknown_head}")
    tail = [n for n in anchor_order if n not in ABLATION_KERNELS]
    order = {name: index for index, name in enumerate(ABLATION_KERNELS + tail)}
    jobs.sort(key=lambda job: order[job["workload"]["name"]])

    # Alternate cards.  Global alternation: a truncated run leaves the cards
    # balanced and no kernel's number belongs to one card.
    for index, job in enumerate(jobs):
        target = dict(job["target"])
        target["bdf"] = BDFS[index % len(BDFS)]
        job["target"] = target

    plan["jobs"] = jobs
    plan["tuned_profile_note"] = (
        "2SM_HBM_v2 driven with the TUNED JEDEC HBM2 2000 MT/s profile: 43 "
        "FASED runtime configuration registers written to each of 5 bridges "
        "(215 explicit settings) plus the 5 "
        "+mm_useHardwareDefaultRuntimeSettings_<bridge> opt-in flags, 220 "
        "plusargs in all.  The profile is the HARDWARE OWNER's, recovered from "
        "his run 800 raw_output, and is copied here byte-for-byte from "
        "runs/hpc-v2-hbm-tuned-1a1b-v1/plan.json.  This campaign claims no part "
        "of its derivation and any result reported from it must credit him."
    )
    plan["tuned_profile_provenance"] = {
        "anchor_plan": str(TUNED_PLAN),
        "extra_plusargs_sha256": sha256_text(vector),
        "plusargs": len(vector.split()),
        "registers": len(registers),
        "bridges": BRIDGES,
        "hardware_default_optin_flags": sorted(flags),
        "byte_identical_to_anchor": True,
        "registers_and_values": dict(sorted(registers.items())),
    }
    plan["experiment_question"] = (
        "Tuning recovered ~13-16 points of HBM2 cycles, and the FASED counters "
        "say it did so by lowering per-request SERVICE TIME: bytes unchanged on "
        "19/21 kernels, occupancy down on 21/21, implied latency 65.4 -> 49.2 "
        "cycles.  This arm supplies the missing axis -- what the recovery did "
        "to COMPUTE utilisation -- by running the compute-instrumented host "
        "ELFs against the tuned model.  With it the 2x2 "
        "(untuned/tuned x DDR3_v2/HBM2_v2) is complete on both the memory and "
        "the compute side."
    )
    plan["device_compute_counters"] = COMPUTE_COUNTERS
    plan["job_order_rationale"] = (
        "The three ablation kernels run FIRST -- stream-triad, spatter-0, "
        "spmv-jds -- because runs/hbm-tuning-ablation-v1 decomposes exactly "
        "those three with the full leave-one-out / leave-one-in register-group "
        "ablation.  A compute-utilisation number for them lands in the richest "
        "available context: the ablation says WHICH register group bought the "
        "cycles, these jobs say what the compute pipeline was doing while it "
        "did.  The remaining 18 keep the tuned anchor's own information order, "
        "so the tail of this run is directly comparable to the tail of that "
        "one.  BDFs alternate 1a/1b globally, as in "
        "scripts/prepare_roofline_dual_bdf_plan.py, so an early stop leaves the "
        "cards balanced and no kernel's result is a one-card artefact."
    )
    plan["replication_rationale"] = (
        "n=1.  This host is bit-deterministic for a fixed (bitstream, ELF, "
        "plusargs) triple: 29 of 31 replicate groups returned identical "
        "kernel_cycles, CV=0.  Replication would re-measure the same integer."
    )
    plan["reused_artifacts"] = {
        "build_manifest": str(BUILD_MANIFEST),
        "build_manifest_sha256": plan["build_manifest_sha256"],
        "host_elfs": "reused byte-for-byte; nothing was rebuilt",
        "instrumented_host_elf_count": len(reused),
        "compute_instrumentation_commit":
            "a8b6bcdd70f8412d9d4e64f057825746320e62da",
    }
    plan["anchors"] = {
        "untuned_dram_v2_and_hbm_v2_compute":
            str(ROOT / "runs/hpc-v2-compute-instrumented-v1"),
        "tuned_hbm_v2_cycles_only":
            str(ROOT / "runs/hpc-v2-hbm-tuned-1a1b-v1"),
        "register_group_ablation": str(ROOT / "runs/hbm-tuning-ablation-v1"),
        "note": (
            "No anchor job is re-run here.  The DDR3_v2 arm and the untuned "
            "HBM2 arm stay the comparison baselines exactly as measured."
        ),
    }
    plan["no_bitstream_rebuild"] = (
        "All 43 tuned settings are FASED RUNTIME configuration registers "
        "written over the control bus at driver start-up.  Same bitstream, same "
        "driver, same ELFs, byte-identical asset receipts.  Nothing is built."
    )
    campaign.write_json(OUT / "plan.json", plan)

    check = preflight(campaign, json.loads((OUT / "plan.json").read_text()))
    (OUT / "preflight.json").write_text(
        json.dumps(check, indent=1, sort_keys=True) + "\n")
    if check["status"] != "pass":
        raise SystemExit("preflight failed:\n  " + "\n  ".join(check["problems"]))

    by_bdf: dict[str, int] = {}
    for job in jobs:
        by_bdf[job["target"]["bdf"]] = by_bdf.get(job["target"]["bdf"], 0) + 1
    summary = {
        "schema": "radiance-compute-instrumented-tuned-hbm-prepare-v1",
        "plan": str(OUT / "plan.json"),
        "jobs": len(jobs),
        "target": TARGET,
        "jobs_per_bdf": dict(sorted(by_bdf.items())),
        "tuned_extra_plusargs_sha256": sha256_text(vector),
        "tuned_plusargs": len(vector.split()),
        "tuned_registers": len(registers),
        "preflight": check["status"],
        "order": [
            f"{job['workload']['name'].removeprefix('guarded-profile-')}"
            f" @{job['target']['bdf']}" for job in jobs
        ],
        "not_run": "this script prepares the plan only; it never executes a job",
    }
    (OUT / "prepare-summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
