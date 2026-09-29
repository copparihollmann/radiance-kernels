#!/usr/bin/env python3
"""Build and run one Spatter JSON configuration on Radiance/Muon."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from plan import plan_for

ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
MASK64 = (1 << 64) - 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
KINDS = {"gather": 0, "scatter": 1, "gs": 2, "multigather": 3, "multiscatter": 4}
ARRAYS = (
    "pattern", "pattern_gather", "pattern_scatter", "sparse", "dense",
    "sparse_gather", "sparse_scatter", "group_offsets", "group_tasks",
)


def positive(value: object, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def parse_pattern(value: object, delta: int) -> tuple[list[int], int]:
    if isinstance(value, list):
        if not value:
            raise ValueError("empty pattern")
        return [positive(v, "pattern index", True) for v in value], delta
    if not isinstance(value, str):
        raise ValueError("pattern must be an array or Spatter pattern string")
    parts = value.upper().split(":")
    if parts[0] == "UNIFORM" and len(parts) in (3, 4):
        length = positive(int(parts[1]), "uniform length")
        stride = positive(int(parts[2]), "uniform stride")
        if len(parts) == 4:
            delta = length * stride if parts[3] == "NR" else positive(int(parts[3]), "uniform delta")
        return list(range(0, length * stride, stride)), delta
    if parts[0] == "MS1" and len(parts) == 4:
        length = positive(int(parts[1]), "MS1 length")
        positions = [positive(int(v), "gap location", True) for v in parts[2].split(",")]
        gaps = [positive(int(v), "gap", True) for v in parts[3].split(",")]
        if len(gaps) not in (1, len(positions)):
            raise ValueError("MS1 gap count does not match gap locations")
        result, current, gap_id = [], -1, 0
        for j in range(length):
            if gap_id < len(positions) and positions[gap_id] == j:
                current += gaps[0 if len(gaps) == 1 else gap_id]
                gap_id += 1
            else:
                current += 1
            result.append(current)
        if min(result) < 0:
            raise ValueError("MS1 generated negative index")
        return result, delta
    if parts[0] == "LAPLACIAN" and len(parts) == 4:
        dimension = positive(int(parts[1]), "Laplacian dimension")
        order = positive(int(parts[2]), "Laplacian order")
        size = positive(int(parts[3]), "Laplacian size")
        offsets = [(j + 1) * size**i for i in range(dimension) for j in range(order)]
        center = max(offsets)
        return [center - v for v in reversed(offsets)] + [center] + [center + v for v in offsets], 1
    if re.fullmatch(r"[0-9]+(,[0-9]+)*", value):
        return [int(v) for v in value.split(",")], delta
    raise ValueError(f"unsupported Spatter pattern: {value[:80]}")


def normalize(raw: dict) -> dict:
    for key in ("atomic-writes", "boundary", "compress", "shared-memory"):
        if raw.get(key):
            raise ValueError(f"{key} is not supported by the Radiance mapping")
    kind_name = str(raw.get("kernel", "Gather")).lower()
    if kind_name not in KINDS:
        raise ValueError(f"unsupported kernel: {kind_name}")
    collision_policy = raw.get("collision-policy", "parallel")
    if collision_policy not in ("parallel", "ordered"):
        raise ValueError("collision-policy must be parallel or ordered")
    if collision_policy == "ordered" and kind_name not in ("scatter", "gs", "multiscatter"):
        raise ValueError("ordered collision-policy needs a scatter destination")
    count = positive(raw.get("count", 1024), "count")
    local_work_size = raw.get("local-work-size")
    if local_work_size is not None:
        local_work_size = positive(local_work_size, "local-work-size")
    wrap = positive(raw.get("wrap", 1), "wrap")
    gather_final_wrap = positive(raw.get("gather-final-wrap", 0),
                                 "gather-final-wrap", True)
    if gather_final_wrap and kind_name != "gs":
        raise ValueError("gather-final-wrap is supported only for GS")
    delta = positive(raw.get("delta", 8), "delta", True)
    delta_gather = positive(raw.get("delta-gather", 8), "delta-gather", True)
    delta_scatter = positive(raw.get("delta-scatter", 8), "delta-scatter", True)
    pattern, delta = parse_pattern(raw["pattern"], delta) if "pattern" in raw else ([], delta)
    gather, delta_gather = (
        parse_pattern(raw["pattern-gather"], delta_gather)
        if "pattern-gather" in raw else ([], delta_gather)
    )
    scatter, delta_scatter = (
        parse_pattern(raw["pattern-scatter"], delta_scatter)
        if "pattern-scatter" in raw else ([], delta_scatter)
    )
    pattern_size = raw.get("pattern-size")
    if pattern_size is not None:
        n = positive(pattern_size, "pattern-size")
        pattern = pattern[:n]
        gather = gather[:n]
        scatter = scatter[:n]
    if kind_name in ("gather", "scatter") and not pattern:
        raise ValueError("gather/scatter needs pattern")
    if kind_name == "gs" and (not gather or len(gather) != len(scatter)):
        raise ValueError("GS requires equal nonempty gather/scatter patterns")
    if kind_name == "multigather" and (not pattern or not gather or max(gather) >= len(pattern)):
        raise ValueError("MultiGather inner indices must address pattern")
    if kind_name == "multiscatter" and (not pattern or not scatter or max(scatter) >= len(pattern)):
        raise ValueError("MultiScatter inner indices must address pattern")
    length = len(gather) if kind_name in ("gs", "multigather") else (
        len(scatter) if kind_name == "multiscatter" else len(pattern)
    )
    if count * length > 0xFFFFFFFF:
        raise ValueError("count * pattern length exceeds RV32 task range")
    if kind_name == "gs":
        src_length = max(gather) + delta_gather * (count - 1) + 1
        dst_length = max(scatter) + delta_scatter * (count - 1) + 1
        source, output = "sparse_gather", "sparse_scatter"
    elif kind_name in ("gather", "multigather"):
        src_length = max(pattern) + delta * (count - 1) + 1
        dst_length = length * wrap
        source, output = "sparse", "dense"
    else:
        src_length = length * wrap
        dst_length = max(pattern) + delta * (count - 1) + 1
        source, output = "dense", "sparse"
    payload_tag = positive(raw.get("source-tag", source_tag(source)), "source-tag")
    if payload_tag > 0xFFFFFFFF:
        raise ValueError("source-tag exceeds the supported range")
    if max(src_length, dst_length) * 8 > 0x70000000:
        raise ValueError("array exceeds the current GPU DRAM address window")
    pattern_words = sum(max(1, len(table)) for table in (pattern, gather, scatter))
    schedule_words = (dst_length + 1 + count * length
                      if collision_policy == "ordered" else 2)
    embedded_bytes = ((src_length + dst_length) * 8 +
                      (pattern_words + schedule_words) * 4)
    # RV32 code, runtime data, guards, and alignment occupy the same window.
    if embedded_bytes + 0x100000 > 0x70000000:
        raise ValueError("input, output, patterns, and schedule exceed GPU DRAM address window")
    case = dict(
        kind=kind_name, pattern=pattern, gather=gather, scatter=scatter,
        count=count, wrap=wrap, delta=delta, delta_gather=delta_gather,
        delta_scatter=delta_scatter, gather_final_wrap=gather_final_wrap,
        length=length, src_length=src_length,
        dst_length=dst_length, source=source, output=output,
        payload_tag=payload_tag,
        collision_policy=collision_policy,
        requested_local_work_size=local_work_size,
    )
    case["_plan"] = plan_for(case)
    return case


def payload(tag: int, index: int) -> int:
    x = (index + tag * 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def fnv(values) -> int:
    h = FNV_OFFSET
    for value in values:
        h = ((h ^ (value & 0xFFFFFFFF)) * FNV_PRIME) & MASK64
        h = ((h ^ (value >> 32)) * FNV_PRIME) & MASK64
    return h


def source_tag(source: str) -> int:
    return {"sparse": 1, "dense": 2, "sparse_gather": 3}[source]


def destination(case: dict, i: int, j: int) -> int:
    return case["_plan"].write.at(case, i, j)


def source_index(case: dict, i: int, j: int) -> int:
    return case["_plan"].read.at(case, i, j)


def reference(case: dict) -> tuple[int | None, bool]:
    """Return serial output digest and whether destination writes overlap."""
    length, count = case["length"], case["count"]
    tag = case["payload_tag"]
    if case["kind"] in ("gather", "multigather"):
        def values():
            for r in range(case["wrap"]):
                last = r + ((count - 1 - r) // case["wrap"]) * case["wrap"] if r < count else None
                for j in range(length):
                    yield payload(tag, source_index(case, last, j)) if last is not None else 0
        return fnv(values()), False
    seen = bytearray(case["dst_length"])
    overlap = False
    # xRAGE 9 has repeated destinations. Its output is intentionally exploratory.
    output = array("Q", [0]) * case["dst_length"]
    for i in range(count):
        for j in range(length):
            dst = destination(case, i, j)
            if seen[dst]:
                overlap = True
            seen[dst] = 1
            if not overlap or case["collision_policy"] == "ordered":
                output[dst] = payload(tag, source_index(case, i, j))
    return (None if overlap and case["collision_policy"] == "parallel" else fnv(output)), overlap


def sample_digest(case: dict, samples: int = 64) -> int:
    """Expected digest over evenly spaced output positions for RTL readback."""
    n = min(samples, case["dst_length"])
    positions = [i * (case["dst_length"] - 1) // (n - 1) for i in range(n)] if n > 1 else [0]
    if case["kind"] in ("gather", "multigather"):
        values = []
        for pos in positions:
            r, j = divmod(pos, case["length"])
            if r >= case["count"]:
                values.append(0)
            else:
                last = r + ((case["count"] - 1 - r) // case["wrap"]) * case["wrap"]
                values.append(payload(case["payload_tag"], source_index(case, last, j)))
        return fnv(values)
    selected = {pos: 0 for pos in positions}
    for i in range(case["count"]):
        for j in range(case["length"]):
            dst = destination(case, i, j)
            if dst in selected:
                selected[dst] = payload(case["payload_tag"], source_index(case, i, j))
    return fnv(selected[pos] for pos in positions)


def write_u32(path: Path, values: list[int]) -> None:
    with path.open("wb") as stream:
        if not values:
            stream.write(b"\0\0\0\0")
        for start in range(0, len(values), 65536):
            chunk = values[start:start + 65536]
            if any(v > 0xFFFFFFFF for v in chunk):
                raise ValueError("pattern index exceeds RV32 range")
            stream.write(struct.pack(f"<{len(chunk)}I", *chunk))


def write_source(path: Path, length: int, tag: int) -> None:
    with path.open("wb") as stream:
        for start in range(0, length, 32768):
            chunk = [payload(tag, i) for i in range(start, min(length, start + 32768))]
            stream.write(struct.pack(f"<{len(chunk)}Q", *chunk))


def write_destination_groups(case: dict) -> None:
    """Stable counting sort of transfers by destination, preserving every write.

    Each destination is owned by one GPU lane. Writes to a repeated address
    retain source order, so two 32-bit stores cannot tear against another
    lane's write to the same 64-bit value.
    """
    destinations = case["dst_length"]
    tasks = case["count"] * case["length"]
    counts = array("I", [0]) * destinations
    for task in range(tasks):
        i, j = divmod(task, case["length"])
        counts[destination(case, i, j)] += 1
    offsets = array("I", [0]) * (destinations + 1)
    for index, count in enumerate(counts):
        offsets[index + 1] = offsets[index] + count
    cursor = offsets[:destinations]
    ordered = array("I", [0]) * tasks
    for task in range(tasks):
        i, j = divmod(task, case["length"])
        dst = destination(case, i, j)
        ordered[cursor[dst]] = task
        cursor[dst] += 1
    write_u32(GENERATED / "group_offsets.bin", offsets)
    write_u32(GENERATED / "group_tasks.bin", ordered)


def prepare(case: dict, suite: Path, case_id: int) -> dict:
    GENERATED.mkdir(exist_ok=True)
    write_u32(GENERATED / "pattern.bin", case["pattern"])
    write_u32(GENERATED / "pattern_gather.bin", case["gather"])
    write_u32(GENERATED / "pattern_scatter.bin", case["scatter"])
    write_source(GENERATED / "source.bin", case["src_length"], case["payload_tag"])
    if case["collision_policy"] == "ordered":
        write_destination_groups(case)
    else:
        write_u32(GENERATED / "group_offsets.bin", [])
        write_u32(GENERATED / "group_tasks.bin", [])
    expected, overlap = reference(case)
    exploratory = overlap and case["collision_policy"] == "parallel"
    probe_index = destination(case, 0, 0) if exploratory else 0
    readback_samples = min(64, case["dst_length"]) if case["dst_length"] > 1024 and not exploratory else 0
    expected_sample = sample_digest(case, readback_samples) if readback_samples else 0
    lines = [
        "#pragma once",
        f"#define SPATTER_KIND {KINDS[case['kind']]}",
        "#define SPATTER_CHAIN 0",
        f"#define SPATTER_PATTERN_LENGTH {case['length']}u",
        f"#define SPATTER_COUNT {case['count']}u",
        f"#define SPATTER_WRAP {case['wrap']}u",
        f"#define SPATTER_DELTA {case['delta']}u",
        f"#define SPATTER_DELTA_GATHER {case['delta_gather']}u",
        f"#define SPATTER_DELTA_SCATTER {case['delta_scatter']}u",
        f"#define SPATTER_GATHER_FINAL_WRAP {case['gather_final_wrap']}u",
        f"#define SPATTER_ORDERED_COLLISIONS {int(case['collision_policy'] == 'ordered')}",
        f"#define SPATTER_OUTPUT_LENGTH {case['dst_length']}u",
        f"#define SPATTER_EXPECTED_DIGEST 0x{(expected or 0):016x}ULL",
        f"#define SPATTER_READBACK_SAMPLES {readback_samples}u",
        f"#define SPATTER_EXPECTED_SAMPLE_DIGEST 0x{expected_sample:016x}ULL",
        f"#define SPATTER_EXPLORATORY_PROBE_INDEX {probe_index}u",
        f"#define SPATTER_EXPLORATORY {int(exploratory)}",
    ]
    (GENERATED / "config.h").write_text("\n".join(lines) + "\n")
    names = {
        "pattern": "pattern.bin", "pattern_gather": "pattern_gather.bin",
        "pattern_scatter": "pattern_scatter.bin",
        "group_offsets": "group_offsets.bin", "group_tasks": "group_tasks.bin",
        case["source"]: "source.bin",
    }
    asm = ['.section .data,"aw",@progbits']
    for name in ARRAYS:
        if name == case["output"]:
            continue
        asm.extend([".balign 64", f".globl spatter_{name}", f"spatter_{name}:"])
        asm.append(f'.incbin "generated/{names[name]}"' if name in names else ".long 0")
    asm.extend([
        '.section .bss,"aw",@nobits', '.balign 64',
        '.globl spatter_guard_before', 'spatter_guard_before:', '.zero 64',
        '.globl spatter_' + case["output"], 'spatter_' + case["output"] + ':',
        f'.zero {case["dst_length"] * 8}',
        '.globl spatter_guard_after', 'spatter_guard_after:', '.zero 64',
    ])
    (ROOT / "data.S").write_text("\n".join(asm) + "\n")
    address_plan = case["_plan"].as_dict()
    (GENERATED / "plan.json").write_text(json.dumps(address_plan, indent=2) + "\n")
    dataset_hash = hashlib.sha256(suite.read_bytes()).hexdigest()
    manifest = {
        "suite": str(suite.resolve()), "suite_sha256": dataset_hash, "case": case_id,
        "kind": case["kind"], "pattern_length": case["length"],
        "address_plan": address_plan,
        "count": case["count"], "wrap": case["wrap"],
        "source_tag": case["payload_tag"],
        "requested_local_work_size": case["requested_local_work_size"],
        "muon_warps_per_core": 4,
        "source_elements": case["src_length"], "output_elements": case["dst_length"],
        "logical_payload_bytes": case["count"] * case["length"] * 16,
        "expected_digest": None if expected is None else f"{expected:016x}",
        "expected_sample_digest": f"{expected_sample:016x}" if readback_samples else None,
        "readback_samples": readback_samples,
        "exploratory_probe_index": probe_index if overlap else None,
        "destination_overlap": overlap,
        "collision_policy": case["collision_policy"],
        "status": "prepared",
    }
    (GENERATED / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_command(argv: list[str], log: Path, cwd: Path | None = None,
                env: dict[str, str] | None = None) -> None:
    with log.open("w") as output:
        subprocess.run(argv, cwd=cwd, env=env, stdout=output,
                       stderr=subprocess.STDOUT, check=True)


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def source_hash() -> str:
    digest = hashlib.sha256()
    for name in ("run.py", "plan.py", "Makefile", "kernel.cpp", "spatter_ops.hpp", "host.cpp",
                 "emit_symbols.py", "cyclotron-no-trace.patch"):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--case", type=int, default=0, help="zero-based JSON configuration index")
    parser.add_argument("--count", type=int,
                        help="replace the suite repetition count for a clearly labeled scaled run")
    parser.add_argument("--collision-policy", choices=("parallel", "ordered"),
                        help="ordered serializes transfers per destination for complete scatter checks")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--sim-only", action="store_true",
                        help="run an already built ELF from --out without regenerating inputs")
    parser.add_argument("--simv", type=Path, help="VCS simulator for the pinned Radiance checkout")
    parser.add_argument("--max-cycles", type=int, default=2_000_000_000)
    parser.add_argument("--gpu-mhz", type=float,
                        help="measured GPU clock for converting cycles to GB/s")
    parser.add_argument("--record-trace", action="store_true",
                        help="retain Cyclotron's SQLite instruction/memory trace")
    args = parser.parse_args()
    if args.gpu_mhz is not None and args.gpu_mhz <= 0:
        parser.error("--gpu-mhz must be positive")
    if args.max_cycles <= 0:
        parser.error("--max-cycles must be positive")
    if args.count is not None and args.count <= 0:
        parser.error("--count must be positive")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.sim_only:
        if args.prepare_only or args.build_only or args.simv is None:
            parser.error("--sim-only requires --simv and excludes --prepare-only/--build-only")
        if not (out / "result.json").exists() or not (out / "kernel.soc.elf").exists():
            parser.error("--sim-only requires result.json and kernel.soc.elf in --out")
        manifest = json.loads((out / "result.json").read_text())
        if manifest["suite"] != str(args.suite.resolve()) or manifest["case"] != args.case:
            parser.error("--suite and --case must match the existing build")
    else:
        suite = json.loads(args.suite.read_text())
        if not isinstance(suite, list) or not 0 <= args.case < len(suite):
            parser.error("suite must be a nonempty JSON list and --case must select an entry")
        original_count = suite[args.case].get("count", 1024)
        raw_case = dict(suite[args.case])
        if args.collision_policy is not None:
            raw_case["collision-policy"] = args.collision_policy
        if args.count is not None:
            raw_case["count"] = args.count
        case = normalize(raw_case)
        manifest = prepare(case, args.suite, args.case)
        manifest["original_count"] = original_count
        manifest["count_overridden"] = args.count is not None and args.count != original_count
        manifest["spatter_revision"] = git_revision(args.suite.parent)
        manifest["kernel_revision"] = git_revision(ROOT)
        manifest["kernel_source_sha256"] = source_hash()
        manifest["radiance_revision"] = (
            git_revision(args.simv.resolve().parents[2] / "generators/radiance")
            if args.simv else None
        )
        manifest["cyclotron_revision"] = (
            git_revision(args.simv.resolve().parents[2] / "generators/radiance/cyclotron")
            if args.simv else None
        )
        manifest["simulator"] = "VCS" if args.simv else None
        manifest["cyclotron_trace_requested"] = args.record_trace
        manifest["correctness"] = (
            "guards-and-nonzero-exploratory" if manifest["destination_overlap"] and
            manifest["collision_policy"] == "parallel" else
            "sample-checked" if manifest["readback_samples"] else "digest-checked"
        )
        (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        return 0
    try:
        if not args.sim_only:
            run_command(["make", "kernel.soc.elf"], out / "build.log", ROOT)
            shutil.copy2(ROOT / "kernel.soc.elf", out / "kernel.soc.elf")
            manifest["status"] = "built"
        if not args.build_only:
            if args.simv is None:
                parser.error("--simv is required unless --prepare-only or --build-only is used")
            elf = out / "kernel.soc.elf"
            manifest.pop("error", None)
            manifest.pop("gpu_cycles", None)
            manifest.pop("gpu_mhz", None)
            manifest.pop("payload_gbps", None)
            manifest["radiance_revision"] = git_revision(
                args.simv.resolve().parents[2] / "generators/radiance")
            manifest["cyclotron_revision"] = git_revision(
                args.simv.resolve().parents[2] / "generators/radiance/cyclotron")
            manifest["simulator"] = "VCS"
            manifest["cyclotron_trace_requested"] = args.record_trace
            manifest["status"] = "running"
            (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
            sim_env = os.environ.copy()
            if not args.record_trace:
                sim_env["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
            run_command([
                str(args.simv.resolve()), "+permissive",
                "+gpu_finish_keeps_sim=1", f"+max-cycles={args.max_cycles}",
                f"+trace-db={out / 'trace.sqlite'}",
                f"+loadmem={elf}", "+permissive-off", str(elf),
            ], out / "vcs.log", out, sim_env)
            log = (out / "vcs.log").read_text(errors="replace")
            cycles = [int(v) for v in re.findall(r"\bCycles:\s*(\d+)", log)]
            manifest["gpu_cycles"] = max(cycles) if cycles else None
            if args.gpu_mhz is not None:
                manifest["gpu_mhz"] = args.gpu_mhz
            if cycles and args.gpu_mhz is not None:
                manifest["payload_gbps"] = round(
                    manifest["logical_payload_bytes"] * args.gpu_mhz * 1e6
                    / (max(cycles) * 1e9), 6
                )
            # With +verbose disabled VCS does not print its PASS line. The
            # host publishes HTIF status 1/3 only after validation.
            manifest["status"] = (
                "failed" if not cycles or "*** FAILED ***" in log else
                "exploratory" if manifest["destination_overlap"] and
                manifest["collision_policy"] == "parallel" else "passed"
            )
    except subprocess.CalledProcessError as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{error.cmd[0]} exited {error.returncode}"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["error"] = "interrupted by signal"
        (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return 130
    (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] in ("built", "passed", "exploratory") else 1


if __name__ == "__main__":
    raise SystemExit(main())
