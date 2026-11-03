#!/usr/bin/env python3
"""Pre-register the STREAM layout-offset experiment on 2SM_DRAM_v2.

THE QUESTION
------------
Is STREAM's row-buffer thrashing a property of STREAM, or an artifact of where
the linker happened to put its arrays?

On 2SM_DRAM_v2 the FASED rankPower counters give a row-conflict rate
numACT / (numCASR + numCASW) of 0.5226 for Copy and 0.6490 for Triad, against
0.0446 for spatter-0 and ~0.0039 for an ideal dense stream through an 8 KB DDR3
page.  A perfectly sequential 1 Mi-element stream should have excellent row
locality.  It does not.

THE DIAGNOSIS, VERIFIED (not assumed) FROM THE DEPLOYED ARTIFACT
----------------------------------------------------------------
llvm-nm on runs/hpc-profile-guard-builds-v1/stream-triad/2sm/kernel.radiance.elf
-- the device ELF the live 2SM_DRAM_v2 campaign actually runs -- gives:

    stream_b (in)   client 0x10004000   slice-local 0x04001000   bank 0 rank 0
    stream_c (in)   client 0x10404000   slice-local 0x04101000   bank 0 rank 0
    stream_a (out)  client 0x10805140   slice-local 0x04201440   bank 0 rank 0

Composing the two maps that the RTL actually implements -- the 4-slice,
32 B-unit GPU address hash (radiance AddressHashParams; parameters pinned in
kernels/common.mk:200-203) and the DDR3 default map (midas DramCommon.scala:
118-133, bank = addr[15:13], rank = addr[17:16]) -- the two input arrays are
4 MiB apart in client space, which is exactly 1 MiB apart in slice-local space,
which is an exact multiple of the 256 KiB bank+rank period.  Measured with
kernels/evaluation/fpga_1a/campaigns/stream_layout_bank_map.py, the two inputs
share a bank AND a rank at 100.00% of element indices and the output joins them
at 86.52%.  Every read<->writeback alternation is a forced PRE+ACT, tRP 14 +
tRCD 14 = 28 cycles (DramCommon.scala:78,81, hardware defaults).

THE OFFSETS, AND WHY 8 KB WOULD NOT HAVE WORKED
-----------------------------------------------
The hash sends consecutive 32 B units round-robin across the 4 slices and keeps
each slice dense, so a client-space shift of N bytes is only an N/4 shift in the
slice-local space the DDR3 map decodes.  The "shift one array base by 8 KB"
suggestion is a 2 KiB slice-local shift -- a quarter of a bank -- and the same
decoder says it would only take the worst pair from 1.0000 to 0.7500.  The two
arms here are chosen in slice-local terms:

    bank32k        +32 KiB/array ->  +8 KiB slice-local = +1 bank, same rank
    bankrank288k  +288 KiB/array -> +72 KiB slice-local = +1 bank AND +1 rank

Both measure 0.0000 same-bank-and-rank over every element index, for every pair
of arrays the kernel streams.  bank32k is the minimal fix; bankrank288k also
separates the ranks, so even a bank-group or scheduler effect inside one rank
cannot survive it.

WHY THE BASELINE IS AN ARM
--------------------------
The unmodified layout is rebuilt from the same source revision, with the same
host, and measured in the same batch on the same two cards.  The 2026-11-02
record is the sanity anchor, not the comparison: comparing a new build against a
campaign from another day would confound the layout change with everything else
that moved.

WHY 2SM_DRAM_v2 ONLY
--------------------
rankPower_*_numACT / numCASR / numCASW exist only on the DDR3 bridges.  The HBM
target in the same asset manifest has no rankPower counters, so the row-conflict
metric is undefined there and the experiment is meaningless on it.  The HBM
target stays in the asset manifest (the campaign loader refuses a manifest that
leaves a bitstream in the asset root unrepresented) and is dropped from the job
list afterwards.

Creating this plan touches no FPGA and launches no hardware job.  Read-only with
respect to /scratch/nicolas.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path("/scratch/agustin/radiance-fpga-1a")
CAMPAIGN = ROOT / "src/radiance-kernels/kernels/evaluation/fpga_1a/campaign.py"

OUT = ROOT / "runs/stream-layout-offset-v1"
BUILD_MANIFEST = OUT / "build/direct-hashed-manifest.json"
GATE = OUT / "evidence/gate.json"

# The same asset root the live hpc-v2 campaign runs: it is the only one that
# carries the 2SM_DRAM_v2 bitstream and the profiling runner.
ASSET_ROOT = ROOT / "artifacts/nicolas-fpga-assets-v2-candidates"
SOURCE_ASSETS = ASSET_ROOT / "assets.json"
DUAL_ASSETS = OUT / "assets-dual-bdf.json"

BDFS = ["1a:00.0", "1b:00.0"]
REPETITIONS = 1
TARGET = "firesim-2sm-dram-v2-quadrank"
DROPPED_TARGET = "firesim-2sm-hbm-v2-quadchannel"

# Measured 2026-11-02 on this exact target, from
# analysis/unified-roofline-2026-11-02/fased_metrics.json
# (campaign runs/hpc-v2-memory-models-dual-1a1b-v1).  These are the numbers the
# prediction below is registered against.
RECORD = {
    "stream-copy": {
        "result": str(ROOT / "runs/hpc-v2-memory-models-dual-1a1b-v1/jobs/"
                             "firesim-2sm-dram-v2-quadrank--guarded-profile-"
                             "stream-copy--measure-0/result.json"),
        "act_per_cas": 0.5226173189995897,
        "act": 240786, "cas": 460731,
        "kernel_cycles": 5284012,
        "gpu_total_bytes": 14743392,
        "gpu_bytes_per_cycle": 2.7902,
        "mean_outstanding_reads_total": 3.0578,
        "reads": 305606,
        "mean_read_latency_cycles": 52.9,
        "all_pre_share_mean": 0.5067,
    },
    "stream-triad": {
        "result": str(ROOT / "runs/hpc-v2-memory-models-dual-1a1b-v1/jobs/"
                             "firesim-2sm-dram-v2-quadrank--guarded-profile-"
                             "stream-triad--measure-0/result.json"),
        "act_per_cas": 0.6489651944230833,
        "act": 388198, "cas": 598180,
        "kernel_cycles": 7514770,
        "gpu_total_bytes": 19141760,
        "gpu_bytes_per_cycle": 2.5472,
        "mean_outstanding_reads_total": 3.3558,
        "reads": 442850,
        "mean_read_latency_cycles": 56.9,
        "all_pre_share_mean": 0.5463,
    },
}

# Most informative first.  Triad has the worst measured row-conflict rate
# (0.6490) and three live arrays, so the Triad baseline/bank32k pair is the
# decisive comparison; it is jobs 0 and 1, which land on different cards and
# therefore complete concurrently and first.
ORDER = [
    ("stream-triad-layout-baseline-2sm", "control: the defect, rebuilt in-batch"),
    ("stream-triad-layout-bank32k-2sm", "minimal fix on the worst kernel"),
    ("stream-copy-layout-baseline-2sm", "control: the defect, rebuilt in-batch"),
    ("stream-copy-layout-bank32k-2sm", "minimal fix on the two-array kernel"),
    ("stream-triad-layout-bankrank288k-2sm", "bank AND rank separation, worst kernel"),
    ("stream-copy-layout-bankrank288k-2sm", "bank AND rank separation, two-array kernel"),
]

PREDICTION = {
    "registered": "2026-11-03, before any hardware job in this plan has run",
    "metric": "dram_act_per_cas = sum(rankPower_r_numACT) / "
              "sum(rankPower_r_numCASR + rankPower_r_numCASW) over the four GPU "
              "bridges and their four ranks, differenced across the ROI window, "
              "exactly as analysis/unified-roofline-2026-11-02/extract_fased.py "
              "computes it for the record.",
    "reference_record": RECORD,
    "static_layout_measurement": {
        "method": "kernels/evaluation/fpga_1a/campaigns/stream_layout_bank_map.py",
        "same_bank_and_rank_fraction_over_live_array_pairs": {
            "baseline_copy": {"stream_a->stream_c": 0.865234375},
            "baseline_triad": {"stream_b->stream_c": 1.0,
                               "stream_b->stream_a": 0.865234375,
                               "stream_c->stream_a": 0.865234375},
            "bank32k_all_pairs": 0.0,
            "bankrank288k_all_pairs": 0.0,
            "rejected_8KiB_client_shift_triad_worst_pair": 0.75,
        },
    },
    "P1_batch_is_comparable": {
        "claim": "The rebuilt baseline arms reproduce the 2026-11-02 record "
                 "within +/-0.08 absolute ACT/CAS.",
        "pass_window": {"stream-copy": [0.44, 0.60], "stream-triad": [0.57, 0.73]},
        "if_violated": "The batch is NOT comparable to the record; read nothing "
                       "else from it until the difference is explained. The "
                       "within-batch baseline-vs-offset comparison may still be "
                       "read, because it is internally controlled.",
    },
    "P2_diagnosis_CONFIRMED_if": {
        "claim": "Both offset arms drop ACT/CAS to <= 0.10 for both operations, "
                 "landing near the 0.0039 ideal (1 activation per 256 column "
                 "accesses in an 8 KB page) and at or below spatter-0's measured "
                 "0.0446.",
        "expected_value": "0.004 to 0.03",
        "threshold": 0.10,
        "reasoning": "If every row conflict comes from three dense streams "
                     "sharing one bank of one rank, separating the banks leaves "
                     "each stream its own open row and the only activations left "
                     "are the genuine 8 KiB page crossings.",
    },
    "P3_diagnosis_REFUTED_if": {
        "claim": "The offset arms stay above 0.80x their own in-batch baseline "
                 "arm (copy > 0.42, triad > 0.52) even though the static decode "
                 "says the arrays no longer share a bank or a rank at any element "
                 "index.",
        "meaning": "Then the thrash is not base-address aliasing. The remaining "
                   "candidates would be intra-array: the four L2 slices' "
                   "independent request streams interleaving inside one channel, "
                   "FR-FCFS write-drain turnarounds, or refresh. The static "
                   "decode in this plan would stand and the DIAGNOSIS would fall.",
    },
    "P4_partial_if": {
        "claim": "ACT/CAS lands between 0.10 and 0.80x baseline.",
        "meaning": "Base-address aliasing is one contributor among several. "
                   "Report the measured share; do not claim the fix.",
    },
    "P5_rank_separation_is_not_worse": {
        "claim": "bankrank288k <= bank32k + 0.01 ACT/CAS for each operation.",
        "if_violated": "A rank-specific cost (rank-to-rank bus turnaround) "
                       "outweighs the extra isolation; prefer bank32k.",
    },
    "P6_traffic_invariance_control": {
        "claim": "GPU total bytes and total CAS are unchanged within +/-1% "
                 "across all three layouts of the same operation "
                 "(copy ~14,743,392 B / 460,731 CAS; triad ~19,141,760 B / "
                 "598,180 CAS).",
        "reasoning": "Moving a base address relocates traffic, it does not "
                     "create or destroy it. A change here means the arm differs "
                     "in something other than layout and is not comparable.",
    },
    "P7_cycles": {
        "claim": "If P2 holds, kernel_cycles fall by 1.15x to 1.70x, centred on "
                 "1.38x for Copy and 1.47x for Triad.",
        "point_estimates": {
            "stream-copy": {"from": 5284012, "to": 3830000, "speedup": 1.38},
            "stream-triad": {"from": 7514770, "to": 5110000, "speedup": 1.47},
        },
        "derivation": "Little's law on the measured baseline. Mean read latency "
                      "L = mean_outstanding_reads * kernel_cycles / reads = "
                      "3.0578*5284012/305606 = 52.9 cycles (Copy) and "
                      "3.3558*7514770/442850 = 56.9 cycles (Triad). A fraction f "
                      "of accesses pays tRP+tRCD = 28 cycles, so a conflict-free "
                      "latency is L - 28f = 38.3 (Copy) and 38.7 (Triad). At "
                      "unchanged concurrency, cycles scale with latency: 52.9/38.3 "
                      "= 1.38x and 56.9/38.7 = 1.47x. Cross-check: spatter-0, "
                      "whose ACT/CAS is already 0.0446, measures L = 30.2 cycles "
                      "on the same target, so a conflict-free STREAM latency near "
                      "38 cycles is in the right range and is not an extrapolation "
                      "past anything observed.",
        "weaker_than_P2_because": "all_pre_share_mean is 0.51-0.55, i.e. every "
                                  "rank sits fully precharged more than half the "
                                  "ROI, so the channel is not saturated and the "
                                  "concurrency the queues sustain may rise or "
                                  "fall once the latency changes.",
        "null_result_that_would_still_be_informative": "ACT/CAS collapses (P2 "
                                                       "holds) but cycles improve "
                                                       "by less than 1.05x. That "
                                                       "would say the row "
                                                       "conflicts were never on "
                                                       "the critical path and the "
                                                       "limiter is request "
                                                       "concurrency, not DRAM.",
    },
    "P8_correctness": {
        "claim": "Every job prints PASS. The device-side full host digest check "
                 "matches the UNCHANGED accepted digests: stream-copy "
                 "49fabae987622325, stream-triad d463e7f745222325.",
        "reasoning": "Moving a base address cannot change arithmetic. A FAIL, or "
                     "a changed digest, invalidates that arm outright; it does "
                     "not become evidence about layout.",
    },
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
    for target in spec["targets"]:
        if target.get("kind") != "firesim":
            continue
        # Both v2 images carry REAL FASED timing models, and their driver aborts
        # at startup unless each of the five bridges is fully specified or
        # explicitly opted in to its hardware defaults.  This is byte-for-byte
        # what runs/hpc-v2-memory-models-dual-1a1b-v1/assets-dual-bdf.json
        # carries (scripts/prepare_v2_dual_bdf_plan.py:86-89), and it has to be,
        # because the record this experiment is registered against was measured
        # with the DDR3 model at its datasheet-derived reset values.  A different
        # runtime setting would change tRP/tRCD and make the comparison void.
        target.setdefault("env", {})["EXTRA_PLUSARGS"] = " ".join(
            f"+mm_useHardwareDefaultRuntimeSettings_{i}" for i in range(5)
        )
        target["memory_model_runtime_settings"] = "hardware defaults (all 5 bridges)"
        # Target-level BDF is advisory only - the per-job assignment below wins.
        target["bdf"] = BDFS[0]
    return spec


def main() -> None:
    campaign = load_campaign()
    OUT.mkdir(parents=True, exist_ok=True)

    gate = json.loads(GATE.read_text())
    if gate.get("status") != "pass" or gate.get("digests_unchanged") is not True:
        raise SystemExit("build gate did not pass; refusing to plan hardware work")

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

    by_name = {}
    for job in plan["jobs"]:
        if job["target_id"] != TARGET:
            continue
        by_name.setdefault(job["workload"]["name"], []).append(job)

    jobs = []
    for name, rationale in ORDER:
        candidates = by_name.get(name, [])
        if len(candidates) != REPETITIONS:
            raise SystemExit(f"expected {REPETITIONS} job(s) for {name}, "
                             f"found {len(candidates)}")
        for job in candidates:
            workload = dict(job["workload"])
            workload["arm_rationale"] = rationale
            gate_row = next(a for a in gate["artifacts"]
                            if a["name"] == name and a["loader"] == "direct-hashed")
            workload["array_offset_bytes"] = gate_row["array_offset_bytes"]
            workload["array_offset_slice_local_bytes"] = \
                gate_row["array_offset_slice_local_bytes"]
            workload["static_same_bank_and_rank_fraction"] = \
                gate_row["worst_live_pair_same_bank_and_rank_fraction"]
            workload["array_bank_rank_decode"] = gate_row["arrays"]
            workload["expected_digest_matches_accepted"] = True
            workload["accepted_device_byte_exact"] = False
            workload["accepted_device_byte_exact_rationale"] = (
                "The device ELF legitimately changed: the array base addresses "
                "moved, which is the whole point. Byte-exactness against the "
                "accepted payload is therefore not the right contract. The "
                "contract is the kernel's own full host check over every FP32 "
                "output word plus both guard bands, whose expected digest is "
                "BIT-IDENTICAL to the accepted one "
                f"({gate_row['expected_digest']})."
            )
            job = dict(job)
            job["workload"] = workload
            jobs.append(job)

    if len(jobs) > 8:
        raise SystemExit(f"job budget is 8, plan has {len(jobs)}")

    # Alternate cards so both stay busy. campaign.py groups FireSim jobs by card
    # and runs the two groups concurrently, so with this ordering the decisive
    # Triad baseline/bank32k pair is the first job on each card and the two
    # complete together, before anything else.
    for index, job in enumerate(jobs):
        target = dict(job["target"])
        target["bdf"] = BDFS[index % len(BDFS)]
        job["target"] = target

    plan["jobs"] = jobs
    plan["prediction"] = PREDICTION
    plan["job_order_rationale"] = (
        "Six jobs: {baseline, bank32k, bankrank288k} x {Triad, Copy}, all on "
        f"{TARGET}. ORDER: Triad first, because Triad has the worst measured "
        "row-conflict rate (0.6490) and three live arrays, so Triad "
        "baseline-vs-bank32k is the decisive comparison. Those two are jobs 0 "
        "and 1 and land on 1a:00.0 and 1b:00.0 respectively, so they run "
        "concurrently and finish first: a run truncated after one job per card "
        "still answers the question. Copy baseline-vs-bank32k follows as the "
        "two-array replication, and the bankrank288k pair last because it is a "
        "stronger version of a fix the bank32k pair has already demonstrated or "
        "refuted. Jobs alternate 1a:00.0 / 1b:00.0 globally so an early stop "
        "cannot leave one card systematically ahead."
    )
    plan["excluded_target_note"] = (
        f"{DROPPED_TARGET} is in the asset manifest but carries no jobs. HBM has "
        "no rankPower counters, so dram_act_per_cas -- the entire metric of this "
        "experiment -- does not exist there. The target is left in the manifest "
        "because the campaign loader refuses an assets.json that leaves a "
        "bitstream in the asset root unrepresented."
    )
    plan["baseline_is_an_arm_note"] = (
        "The unmodified layout is an ARM of this batch, not a cross-campaign "
        "comparison. It is rebuilt from the same revision with the same host "
        "flags and measured on the same two cards in the same batch. The "
        "2026-11-02 numbers in prediction.reference_record are the sanity anchor "
        "(prediction P1) and are NOT the control."
    )
    plan["build_gate"] = {
        "gate": str(GATE),
        "status": gate["status"],
        "artifact_count": gate["artifact_count"],
        "digests_unchanged": gate["digests_unchanged"],
        "functional_model": {
            "summary": gate["functional_model"]["summary"],
            "passed": gate["functional_model"]["passed"],
            "result_count": gate["functional_model"]["result_count"],
        },
    }
    campaign.write_json(OUT / "plan.json", plan)

    summary = {
        "schema": "radiance-stream-layout-offset-prepare-v1",
        "plan": str(OUT / "plan.json"),
        "build_manifest": str(BUILD_MANIFEST),
        "asset_root": str(ASSET_ROOT),
        "target": TARGET,
        "jobs": len(jobs),
        "repetitions": REPETITIONS,
        "job_order": [f"{j['id']} @{j['target']['bdf']}" for j in jobs],
        "jobs_per_bdf": {bdf: sum(1 for j in jobs if j["target"]["bdf"] == bdf)
                         for bdf in BDFS},
        "prediction_registered": PREDICTION["registered"],
    }
    (OUT / "prepare-summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
