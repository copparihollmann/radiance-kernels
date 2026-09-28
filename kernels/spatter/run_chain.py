#!/usr/bin/env python3
"""Build a two-stage Gather -> Scatter ELF with a materialized dense array."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from run import (GENERATED, ROOT, destination, fnv, normalize, payload,
                 source_hash, source_index, write_source, write_u32)


def selected(path: Path, index: int) -> dict:
    suite = json.loads(path.read_text())
    if not isinstance(suite, list) or not 0 <= index < len(suite):
        raise ValueError(f"{path}: case {index} is outside the suite")
    return normalize(suite[index])


def gather_slot(gather: dict, slot: int) -> int:
    residue, j = divmod(slot, gather["length"])
    if residue >= gather["count"]:
        return 0
    last = residue + ((gather["count"] - 1 - residue) // gather["wrap"]) * gather["wrap"]
    return payload(gather["payload_tag"], source_index(gather, last, j))


def expected_output(gather: dict, scatter: dict) -> array:
    output = array("Q", [0]) * scatter["dst_length"]
    seen = bytearray(scatter["dst_length"])
    for i in range(scatter["count"]):
        for j in range(scatter["length"]):
            dst = destination(scatter, i, j)
            if seen[dst]:
                raise ValueError("chain Scatter destinations overlap; use a non-overlapping case")
            seen[dst] = 1
            output[dst] = gather_slot(gather, source_index(scatter, i, j))
    return output


def prepare(gather: dict, scatter: dict, suites: tuple[Path, Path],
            cases: tuple[int, int], out: Path) -> dict:
    if gather["kind"] != "gather" or scatter["kind"] != "scatter":
        raise ValueError("chain needs Gather followed by Scatter")
    if gather["dst_length"] != scatter["src_length"]:
        raise ValueError("chain intermediate length does not match Scatter input")
    embedded = (gather["src_length"] + gather["dst_length"] +
                scatter["dst_length"]) * 8 + 4 * (len(gather["pattern"]) +
                len(scatter["pattern"]))
    if embedded + 0x100000 > 0x70000000:
        raise ValueError("chain source, intermediate, and output exceed GPU DRAM window")
    GENERATED.mkdir(exist_ok=True)
    write_u32(GENERATED / "pattern.bin", [])
    write_u32(GENERATED / "pattern_gather.bin", gather["pattern"])
    write_u32(GENERATED / "pattern_scatter.bin", scatter["pattern"])
    write_u32(GENERATED / "group_offsets.bin", [])
    write_u32(GENERATED / "group_tasks.bin", [])
    write_source(GENERATED / "source.bin", gather["src_length"], gather["payload_tag"])
    output = expected_output(gather, scatter)
    expected = fnv(output)
    samples = min(64, len(output)) if len(output) > 1024 else 0
    positions = [i * (len(output) - 1) // (samples - 1) for i in range(samples)] if samples else []
    expected_sample = fnv(output[i] for i in positions) if samples else None
    config = [
        "#pragma once", "#define SPATTER_KIND 2", "#define SPATTER_CHAIN 1",
        f"#define SPATTER_GATHER_LENGTH {gather['length']}u",
        f"#define SPATTER_GATHER_COUNT {gather['count']}u",
        f"#define SPATTER_GATHER_WRAP {gather['wrap']}u",
        f"#define SPATTER_GATHER_DELTA {gather['delta']}u",
        f"#define SPATTER_SCATTER_LENGTH {scatter['length']}u",
        f"#define SPATTER_SCATTER_COUNT {scatter['count']}u",
        f"#define SPATTER_SCATTER_WRAP {scatter['wrap']}u",
        f"#define SPATTER_SCATTER_DELTA {scatter['delta']}u",
        f"#define SPATTER_OUTPUT_LENGTH {scatter['dst_length']}u",
        f"#define SPATTER_EXPECTED_DIGEST 0x{expected:016x}ULL",
        f"#define SPATTER_READBACK_SAMPLES {samples}u",
        f"#define SPATTER_EXPECTED_SAMPLE_DIGEST 0x{(expected_sample or 0):016x}ULL",
        "#define SPATTER_EXPLORATORY 0",
        "#define SPATTER_EXPLORATORY_PROBE_INDEX 0u",
    ]
    (GENERATED / "config.h").write_text("\n".join(config) + "\n")
    asm = [
        '.section .data,"aw",@progbits',
        '.balign 64', '.globl spatter_pattern_gather', 'spatter_pattern_gather:',
        '.incbin "generated/pattern_gather.bin"',
        '.balign 64', '.globl spatter_pattern_scatter', 'spatter_pattern_scatter:',
        '.incbin "generated/pattern_scatter.bin"',
        '.balign 64', '.globl spatter_sparse_gather', 'spatter_sparse_gather:',
        '.incbin "generated/source.bin"',
        '.balign 64', '.globl spatter_dense', 'spatter_dense:',
        f'.zero {gather["dst_length"] * 8}',
        '.section .bss,"aw",@nobits', '.balign 64',
        '.globl spatter_guard_before', 'spatter_guard_before:', '.zero 64',
        '.globl spatter_sparse_scatter', 'spatter_sparse_scatter:',
        f'.zero {scatter["dst_length"] * 8}',
        '.globl spatter_guard_after', 'spatter_guard_after:', '.zero 64',
    ]
    (ROOT / "data.S").write_text("\n".join(asm) + "\n")
    suite_hash = hashlib.sha256()
    for suite in suites:
        suite_hash.update(suite.read_bytes())
    kernel_hash = hashlib.sha256()
    kernel_hash.update(source_hash().encode())
    kernel_hash.update(Path(__file__).read_bytes())
    manifest = {
        "suite": [str(path.resolve()) for path in suites],
        "suite_sha256": suite_hash.hexdigest(), "case": list(cases),
        "kind": "gather-scatter-chain", "pattern_length": scatter["length"],
        "count": scatter["count"], "source_elements": gather["src_length"],
        "intermediate_elements": gather["dst_length"],
        "output_elements": scatter["dst_length"],
        "logical_payload_bytes": 16 * (gather["count"] * gather["length"] +
                                        scatter["count"] * scatter["length"]),
        "expected_digest": f"{expected:016x}",
        "expected_sample_digest": f"{expected_sample:016x}" if samples else None,
        "readback_samples": samples, "destination_overlap": False,
        "collision_policy": "parallel",
        "address_plan": {"stages": [gather["_plan"].as_dict(),
                                      scatter["_plan"].as_dict()],
                         "intermediate": "spatter_dense", "barrier": True},
        "kernel_source_sha256": kernel_hash.hexdigest(),
        "correctness": "sample-checked" if samples else "digest-checked",
        "status": "prepared", "simulator": None,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gather_suite", type=Path)
    parser.add_argument("gather_case", type=int)
    parser.add_argument("scatter_suite", type=Path)
    parser.add_argument("scatter_case", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    gather = selected(args.gather_suite, args.gather_case)
    scatter = selected(args.scatter_suite, args.scatter_case)
    out = args.out.resolve()
    manifest = prepare(gather, scatter,
                       (args.gather_suite, args.scatter_suite),
                       (args.gather_case, args.scatter_case), out)
    if args.prepare_only:
        return
    with (out / "build.log").open("w") as log:
        subprocess.run(["make", "kernel.soc.elf"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    shutil.copy2(ROOT / "kernel.soc.elf", out / "kernel.soc.elf")
    manifest["status"] = "built"
    (out / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
