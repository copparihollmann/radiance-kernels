#!/usr/bin/env python3
"""Compute destination-cache-line ownership for all 21 hpc-v2 HPC kernels.

Why this exists
---------------
`runs/hpc-v2-memory-models-dual-1a1b-v1/plan.json` carries 21 distinct HPC
workloads.  Before this script, three of them recorded partition-safety
evidence (btree, pathfinder, spmv-jds) and even those recorded it as manifest
fields copied from the generator, not as a computed owner map.  Eleven carried
`"correctness": "digest-checked"` and nothing else.

A digest catches wrong values.  It does not catch two SMs writing one 64-byte
line, because the line can be shared and still end up holding the right bytes
on the run you measured.  That is how the xRAGE Scatter defect survived every
digest it was ever checked against
(reports/DEFECT-xrage-4sm-cache-line-2026-11-02.md).

This driver runs one computed line-ownership audit per kernel at 1, 2 and 4
SMs, using the schedule actually compiled into that kernel, and a paired
`unsafe` control that must fail.  A detector that cannot fail proves nothing,
so a kernel is only reported as covered when its control failed.

What it does NOT do
-------------------
It audits the OUTPUT ownership map of each kernel's device partition.  It does
not check numerical results; those remain the job of each family's own artifact
audit and oracle.  Families whose output partition this script cannot derive
from the source and the build manifest are reported as `undetermined`, with the
missing input named, rather than being passed by default.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DENSE = _load("audit_dense_contiguous_line_ownership")
GATHER = _load("audit_spatter_gather_output_line_ownership")
SCATTER = _load("audit_spatter_scatter_task_line_ownership")
XRAGE = _load("audit_xrage_destination_line_ownership")

SM_COUNTS = (1, 2, 4)
SPATTER_SUITE = (REPO / "kernels" / "spatter" / "inputs" / "standard-suite" /
                 "basic-tests" / "gpu-stream.json")

# Each entry names the output array a kernel's SMs write, the schedule that is
# compiled into its device source, and the sizes recorded in the hpc-v2 build
# manifest.  `unsafe` is the control: the same output under the naive schedule
# that mu_contiguous_partition exists to replace.
DENSE_KERNELS = {
    "stream-copy":  dict(elements=1 << 20, element_bytes=4, granule=16,
                         output="stream_c[i]"),
    "stream-scale": dict(elements=1 << 20, element_bytes=4, granule=16,
                         output="stream_b[i]"),
    "stream-add":   dict(elements=1 << 20, element_bytes=4, granule=16,
                         output="stream_c[i]"),
    "stream-triad": dict(elements=1 << 20, element_bytes=4, granule=16,
                         output="stream_a[i]"),
    "stencil3d":    dict(elements=262144, element_bytes=4, granule=16,
                         output="stencil3d_output[linear]"),
    "nearn-1m":     dict(elements=1 << 20, element_bytes=4, granule=16,
                         output="nearn_distances[i]"),
    "hotspot3d":    dict(elements=4096, element_bytes=4, granule=16,
                         repeat_stride=4096,
                         output="output[z * plane + column]"),
    "gemv-bf16-256x4096": dict(elements=256, element_bytes=2, granule=1,
                               output="gemv_y[row]"),
    "pathfinder":   dict(elements=4096, element_bytes=4, granule=16,
                         output="final cost row[col]"),
    "btree-point":  dict(elements=262144, element_bytes=16, granule=4,
                         output="point output record[q]"),
    "btree-range":  dict(elements=131072, element_bytes=16, granule=4,
                         output="range output record[q]"),
    "spmv-jds":     dict(elements=146689, element_bytes=4,
                         schedule="block-cyclic", chunk_elements=4080,
                         output="spmv_out[row]"),
    "dot-1m-inputs": dict(elements=1 << 20, element_bytes=4,
                          schedule="block-cyclic", chunk_elements=2048,
                          output="stream_dot_a/b[i] (read-only chunks)"),
}

# stream_dot is listed separately because its output array is sized by the SM
# count rather than fixed: the reduction emits one float32 partial per SM.  It
# used to be a packed `.zero 16` array written as `stream_dot_partial[block_id]`
# -- every SM in one 64-byte line, which this driver reported as a defect.  The
# array is now padded to one 64-byte line per SM (`.zero 64 * sms`) and written
# as `stream_dot_partial[block_id * 16]`, so the owner map is `stride_floats`
# elements per SM under a contiguous partition at that same granule.  The
# unsafe control is the old packed array, which must still fail at 2 and 4 SMs.
PER_SM_SCALAR = {
    "dot-1m-partials": dict(element_bytes=4, stride_floats=16,
                            output="stream_dot_partial[block_id * 16]"),
}

# LBM does not use mu_contiguous_partition: it reads a build-time
# kGroupBoundaries[] table and writes a PADDED grid through
# `grid_index(x, y, z, entry) = MARGIN + entry + 20 * (x + 40y + 1280z)`.
# Padding means the group index is an indirection onto the address, which is
# the same shape as the xRAGE defect, so the addresses are enumerated rather
# than argued.  Constants are the static_asserts in kernels/lbm/kernel.cpp and
# the generated config: 32x32x8 active in a 40x32x12 padded grid, 20 floats per
# cell, 4 cells per group, margin 51200 floats.
LBM = dict(size=(32, 32, 8), padded=(40, 32, 12), cell_entries=20,
           margin_floats=51200, group_cells=4, groups=2048)

SPATTER_GATHER = {
    "spatter-0": dict(case=0, schedule="contiguous", granule=8),
    "spatter-4": dict(case=4, schedule="contiguous", granule=8),
}
SPATTER_SCATTER = {
    "spatter-1": dict(case=1, granule=16),
    "spatter-2": dict(case=2, granule=16),
    "spatter-3": dict(case=3, granule=16),
}


def lbm_write_addresses() -> list[int]:
    """Every float index LBM writes, in the order the device writes them.

    Groups are the unit of ownership, so the list is group-major and each group
    contributes `group_cells * cell_entries` consecutive entries.  The addresses
    themselves come from the padded `grid_index` formula, not from the group
    index, so a padded grid that broke group/line alignment would show up here.
    """
    size_x, size_y, _ = LBM["size"]
    padded_x, padded_y, _ = LBM["padded"]
    entries = LBM["cell_entries"]
    addresses = []
    for group in range(LBM["groups"]):
        for lane in range(LBM["group_cells"]):
            cell = group * LBM["group_cells"] + lane
            x = cell % size_x
            yz = cell // size_x
            y = yz % size_y
            z = yz // size_y
            base = x + y * padded_x + z * padded_x * padded_y
            for entry in range(entries):
                addresses.append(LBM["margin_floats"] + entry + entries * base)
    return addresses


def _combine(checks: list[dict]) -> dict:
    """Fold per-SM-count checks from separate audits into one report.

    Kernels whose output array is sized by the SM count need one audit per SM
    count, so their checks arrive as several single-check reports.
    """
    return {
        "status": "pass" if all(c["status"] == "pass" for c in checks)
                  else "fail",
        "checks": checks,
        "partition_cache_line_safe":
            all(c["cache_line_exclusive"] for c in checks),
        "partition_complete": all(c["coverage_exactly_once"] for c in checks),
        "partition_overlap": sum(c["doubly_owned_elements"] for c in checks),
    }


def _verdict(safe: dict, unsafe: dict | None) -> str:
    if safe["status"] != "pass":
        return "defect"
    if unsafe is None:
        return "exclusive-no-control"
    if unsafe["status"] == "pass":
        return "control-did-not-fire"
    return "exclusive"


def _row(name: str, output: str, schedule: str, safe: dict,
         unsafe: dict | None) -> dict:
    return {
        "kernel": name,
        "output": output,
        "schedule": schedule,
        "sm_counts": list(SM_COUNTS),
        "cache_line_exclusive": safe.get("partition_cache_line_safe"),
        "coverage_exactly_once": safe.get("partition_complete"),
        "overlap": safe.get("partition_overlap"),
        "split_lines_per_sm_count":
            {str(c["sm_count"]): c["split_line_count"] for c in safe["checks"]},
        "unsafe_control_status": unsafe["status"] if unsafe else None,
        "unsafe_control_split_lines":
            {str(c["sm_count"]): c["split_line_count"]
             for c in unsafe["checks"]} if unsafe else None,
        "verdict": _verdict(safe, unsafe),
        "evidence": "computed-owner-map",
    }


def run(spatter_suite: Path, xrage_suite: Path | None,
        xrage_destinations: Path | None) -> dict:
    rows = []

    for name, spec in DENSE_KERNELS.items():
        spec = dict(spec)
        output = spec.pop("output")
        schedule = spec.pop("schedule", "contiguous")
        safe = DENSE.audit(sm_counts=SM_COUNTS, schedule=schedule,
                           label=name, **spec)
        unsafe = DENSE.audit(spec["elements"], spec["element_bytes"],
                             SM_COUNTS, "round-robin",
                             label=f"{name}-unsafe-control")
        rows.append(_row(name, output,
                         f"{schedule}(granule={spec.get('granule')}, "
                         f"chunk={spec.get('chunk_elements')})",
                         safe, unsafe))

    for name, spec in PER_SM_SCALAR.items():
        stride = spec["stride_floats"]
        checks, control_checks = [], []
        for sms in SM_COUNTS:
            # The padded device array: stride floats per SM, owned contiguously
            # at a granule of exactly one 64-byte line.
            checks.extend(DENSE.audit(sms * stride, spec["element_bytes"],
                                      (sms,), "contiguous", granule=stride,
                                      label=name)["checks"])
            # The control is the packed array this replaced: one float per SM,
            # all in line 0.  It must still fail at 2 and 4 SMs.
            control_checks.extend(
                DENSE.audit(sms, spec["element_bytes"], (sms,), "round-robin",
                            label=f"{name}-unsafe-control")["checks"])
        rows.append(_row(name, spec["output"],
                         f"one 64-byte line per SM "
                         f"(contiguous, granule={stride} floats)",
                         _combine(checks), _combine(control_checks)))

    for name, spec in SPATTER_GATHER.items():
        sys.path.insert(0, str(REPO / "kernels" / "spatter"))
        from run import normalize  # noqa: E402
        raw = json.loads(spatter_suite.read_text())
        case = normalize(raw[spec["case"]], spatter_suite.parent)
        owners = case["length"] * case["wrap"]
        common = dict(length=case["length"], count=case["count"],
                      wrap=case["wrap"], sm_counts=SM_COUNTS)
        safe = GATHER.audit(owners, schedule=spec["schedule"],
                            granule=spec["granule"], label=name, **common)
        unsafe = GATHER.audit(owners, schedule="round-robin",
                              label=f"{name}-unsafe-control", **common)
        rows.append(_row(name, "spatter_dense[owner]",
                         f"{spec['schedule']}(granule={spec['granule']})",
                         safe, unsafe))

    for name, spec in SPATTER_SCATTER.items():
        sys.path.insert(0, str(REPO / "kernels" / "spatter"))
        from run import normalize  # noqa: E402
        raw = json.loads(spatter_suite.read_text())
        case = normalize(raw[spec["case"]], spatter_suite.parent)
        destinations = SCATTER.destinations_for_case(case)
        safe = SCATTER.audit(destinations, SM_COUNTS, spec["granule"],
                             label=name)
        unsafe = SCATTER.audit(destinations, SM_COUNTS, spec["granule"],
                               schedule="round-robin",
                               label=f"{name}-unsafe-control")
        rows.append(_row(name, f"{case['output']}[pattern[j] + delta * i]",
                         f"contiguous-task(granule={spec['granule']})",
                         safe, unsafe))

    if xrage_suite is not None:
        sys.path.insert(0, str(REPO / "kernels" / "spatter"))
        from run import normalize  # noqa: E402
        raw = json.loads(xrage_suite.read_text())
        case = normalize(raw[0], xrage_suite.parent)
        owners = case["length"] * case["wrap"]
        common = dict(length=case["length"], count=case["count"],
                      wrap=case["wrap"], sm_counts=SM_COUNTS)
        safe = GATHER.audit(owners, schedule="block-cyclic",
                            chunk_elements=1024, label="xrage-gather", **common)
        unsafe = GATHER.audit(owners, schedule="round-robin",
                              label="xrage-gather-unsafe-control", **common)
        rows.append(_row("xrage-gather", "spatter_dense[owner]",
                         "block-cyclic(chunk=1024)", safe, unsafe))
    else:
        rows.append({"kernel": "xrage-gather", "verdict": "undetermined",
                     "missing": "--xrage-suite (artifacts/xrage-inputs-v1/"
                                "xrage-suite.json)"})

    if xrage_destinations is not None:
        from array import array
        raw = array("I")
        raw.frombytes(xrage_destinations.read_bytes())
        if sys.byteorder != "little":
            raw.byteswap()
        destinations = list(raw)
        safe = XRAGE.audit(destinations, SM_COUNTS, 8, 8, 64, "line-aligned")
        unsafe = XRAGE.audit(destinations, SM_COUNTS, 8, 8, 64, "rank")
        rows.append(_row("xrage-scatter-ordered",
                         "spatter_sparse[group_destinations[group]]",
                         "destination-line-aligned-group-ranks", safe, unsafe))
    else:
        rows.append({"kernel": "xrage-scatter-ordered",
                     "verdict": "undetermined",
                     "missing": "--xrage-destinations (artifacts/"
                                "xrage-inputs-v1/xrage-scatter.u32le)"})

    rows.append({
        "kernel": "gemm-bf16-512", "verdict": "undetermined",
        "output": "device-tiled MXFP8 GEMM output",
        "missing": "the hpc-v2 workload records no output partition and no "
                   "kernel source directory for this family, only "
                   "b_copy_policy=cluster-parity and shape 512x512x512.  "
                   "lib/mxgemm/mxgemm_lib.hpp::mxgemm() takes threadblock_id "
                   "and never reads it, so the SM-to-output-tile map lives in "
                   "whichever kernels/gemm_* wrapper the build used.  That "
                   "wrapper, or the built ELF's C_raw store map, is needed "
                   "before an owner map can be computed.",
    })
    # kGroupBoundaries[] is {0, 512, 1024, 1536, 2048} at 4 SM and
    # {0, 1024, 2048} at 2 SM, which is exactly a contiguous partition of the
    # 163840 writes at a granule of one group (80 floats).  Partitioning the
    # write list that way reproduces the device table without trusting it.
    lbm_addresses = lbm_write_addresses()
    lbm_granule = LBM["group_cells"] * LBM["cell_entries"]
    safe = SCATTER.audit(lbm_addresses, SM_COUNTS, lbm_granule,
                         payload_bytes=4, label="lbm")
    unsafe = SCATTER.audit(lbm_addresses, SM_COUNTS, lbm_granule,
                           payload_bytes=4, schedule="round-robin",
                           label="lbm-unsafe-control")
    rows.append(_row("lbm", "lbm_output_raw[grid_index(x, y, z, entry)]",
                     f"contiguous-group(granule={lbm_granule} floats)",
                     safe, unsafe))

    defects = [r for r in rows if r["verdict"] == "defect"]
    return {
        "schema": "radiance-hpc-v2-line-ownership-matrix-v1",
        "sm_counts": list(SM_COUNTS),
        "line_bytes": 64,
        "kernels": len(rows),
        "rows": rows,
        "defects": [r["kernel"] for r in defects],
        "undetermined": [r["kernel"] for r in rows
                         if r["verdict"] in ("undetermined",
                                             "manifest-boundaries-only")],
        "status": "fail" if defects else "pass",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spatter-suite", type=Path, default=SPATTER_SUITE)
    ap.add_argument("--xrage-suite", type=Path)
    ap.add_argument("--xrage-destinations", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    report = run(args.spatter_suite, args.xrage_suite, args.xrage_destinations)
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        sys.stdout.write(text)

    for row in report["rows"]:
        print(f"{row['kernel']:<24} {row['verdict']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
