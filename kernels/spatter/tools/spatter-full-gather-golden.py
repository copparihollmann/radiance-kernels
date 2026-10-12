#!/usr/bin/env python3
"""Run the pinned upstream serial Gather at its unsplit original count."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

MASK64 = (1 << 64) - 1
UPSTREAM_REVISION = "ec8923711f8dc21eedff7189f12b02eb06845d2f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_file(path: Path) -> str:
    digest = 0xCBF29CE484222325
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 65536), b""):
            if len(block) % 8:
                raise ValueError("partial 64-bit output word")
            for (value,) in struct.iter_unpack("<Q", block):
                for word in (value & 0xFFFFFFFF, value >> 32):
                    digest = ((digest ^ word) * 0x100000001B3) & MASK64
    return f"{digest:016x}"


def assemble_source(plan: dict, root: Path, output: Path) -> None:
    end = 0
    with output.open("wb+") as combined:
        for index, chunk in enumerate(plan["chunks"]):
            path = root / ("first" if index == 0 else "last" if index == len(plan["chunks"]) - 1
                           else f"chunk-{index}") / "source.bin"
            if path.stat().st_size != chunk["source_bytes"]:
                raise ValueError(f"source size differs from plan: {path}")
            base = chunk["source_index_base"]
            if base > end:
                raise ValueError("source windows have a gap; cannot assemble this plan")
            overlap = end - base
            if overlap > chunk["source_elements"]:
                raise ValueError("source windows are not ordered")
            with path.open("rb") as part:
                if overlap:
                    combined.seek(base * 8)
                    if combined.read(overlap * 8) != part.read(overlap * 8):
                        raise ValueError(f"source windows disagree in overlap: {path}")
                combined.seek(0, 2)
                shutil.copyfileobj(part, combined, 1024 * 1024)
            end = base + chunk["source_elements"]
    if output.stat().st_size != plan["original_source_bytes"]:
        raise ValueError("assembled source differs from original source length")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--chunk-golden-root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["kind"] != "gather" or plan["wrap"] != 1:
        raise ValueError("the full-source oracle requires wrap-1 Gather")
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream.resolve()), "rev-parse", "HEAD"],
        text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"upstream revision differs: {revision}")
    root = args.chunk_golden_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = out / "source.bin"
    assemble_source(plan, root, source)
    first = root / "first"
    last = root / "last"
    for chunk in (first, last):
        result = json.loads((chunk / "result.json").read_text())
        if result["status"] != "passed" or result["upstream_revision"] != revision:
            raise ValueError(f"unverified chunk upstream run: {chunk}")
    for name in ("pattern.bin", "gather.bin", "scatter.bin"):
        if (first / name).read_bytes() != (last / name).read_bytes():
            raise ValueError(f"chunk patterns differ: {name}")
    output = out / "output.bin"
    command = [str(args.driver.resolve()), "gather", str(plan["original_count"]), "1",
               str(plan["delta"]), "8", "8", str(plan["original_source_elements"]),
               str(plan["pattern_length"]), str(first / "pattern.bin"),
               str(first / "gather.bin"), str(first / "scatter.bin"),
               str(source), str(output)]
    with (out / "upstream.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    actual = digest_file(output)
    if actual != plan["expected_full_digest"] or actual != plan["chunks"][-1]["expected_digest"]:
        raise ValueError(f"upstream original-count digest {actual} disagrees with the chunk plan")
    record = {
        "upstream_revision": revision,
        "suite_sha256": plan["suite_sha256"],
        "plan_sha256": sha256(args.plan),
        "upstream_driver_sha256": sha256(args.driver),
        "original_count": plan["original_count"],
        "pattern_length": plan["pattern_length"],
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256(source),
        "pattern_sha256": sha256(first / "pattern.bin"),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256(output),
        "complete_output_digest": actual,
        "status": "passed",
    }
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"upstream original-count Gather passed: {actual}")


if __name__ == "__main__":
    main()
