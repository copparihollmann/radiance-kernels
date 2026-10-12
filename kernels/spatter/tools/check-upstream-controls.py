#!/usr/bin/env python3
"""Run a pinned Spatter case with changed inputs to test the golden check."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spatter_upstream_golden", Path(__file__).with_name("spatter-upstream-golden.py")
)
GOLDEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GOLDEN)
FIELDS = ("variant", "source_sha256", "pattern_sha256", "output_sha256",
          "output_digest", "matches_golden")


def changed_copy(source: Path, destination: Path, a: int, b: int | None = None) -> None:
    data = bytearray(source.read_bytes())
    if b is None:
        data[a] ^= 1
    else:
        data[a:a + 4], data[b:b + 4] = data[b:b + 4], data[a:a + 4]
    if data == source.read_bytes():
        raise ValueError("control did not change its input")
    destination.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "evaluation/upstream-control-results.csv")
    parser.add_argument("--write", action="store_true",
                        help="replace the recorded control results")
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != GOLDEN.UPSTREAM_REVISION:
        raise ValueError(f"upstream revision differs: {revision}")

    name = "smoke-1"  # two-repetition Scatter, with distinct source values
    build = json.loads((ROOT / "runs" / name / "result.json").read_text())
    model = json.loads((ROOT / "runs/model" / name / "result.json").read_text())
    suite = ROOT / "smoke.json"
    if GOLDEN.sha256(suite) != build["suite_sha256"]:
        raise ValueError("smoke suite differs from the built ELF")
    case = GOLDEN.validate_case(build, json.loads(suite.read_text())[build["case"]])
    saved = ROOT / "golden-runs" / name
    saved_output = saved / "output.bin"
    saved_record = json.loads((saved / "result.json").read_text())
    if saved_record["upstream_source_sha256"] != GOLDEN.upstream_source_hash(upstream):
        raise ValueError("upstream source differs from the recorded golden run")
    saved_digest = GOLDEN.fnv_file(saved_output)
    if (model["status"] != "passed" or saved_digest != build["expected_digest"] or
            saved_digest != model["output_digest"]):
        raise ValueError("baseline golden and Radiance model differ")

    with tempfile.TemporaryDirectory() as scratch:
        temporary = Path(scratch)
        driver = temporary / "upstream-golden"
        GOLDEN.compile_driver(upstream, driver)
        source_control = temporary / "source-flip.bin"
        pattern_control = temporary / "pattern-swap.bin"
        changed_copy(saved / "source.bin", source_control, 0)
        changed_copy(saved / "pattern.bin", pattern_control, 0, 4)
        variants = (
            ("baseline", saved / "source.bin", saved / "pattern.bin"),
            ("source-bit-flip", source_control, saved / "pattern.bin"),
            ("pattern-swap", saved / "source.bin", pattern_control),
        )
        records = []
        for variant, source, pattern in variants:
            output = temporary / f"{variant}.bin"
            subprocess.run([
                str(driver), case["kind"], str(case["count"]), str(case["wrap"]),
                str(case["delta"]), str(case["delta_gather"]),
                str(case["delta_scatter"]), str(case["src_length"]),
                str(case["dst_length"]), str(pattern), str(saved / "gather.bin"),
                str(saved / "scatter.bin"), str(source), str(output),
            ], check=True)
            same = output.read_bytes() == saved_output.read_bytes()
            if same != (variant == "baseline"):
                raise ValueError(f"unexpected upstream control result: {variant}")
            records.append(dict(
                variant=variant, source_sha256=GOLDEN.sha256(source),
                pattern_sha256=GOLDEN.sha256(pattern),
                output_sha256=GOLDEN.sha256(output),
                output_digest=GOLDEN.fnv_file(output),
                matches_golden=str(same).lower(),
            ))

        output_control = temporary / "output-flip.bin"
        changed_copy(saved_output, output_control, 0)
        records.append(dict(
            variant="output-bit-flip", source_sha256=GOLDEN.sha256(saved / "source.bin"),
            pattern_sha256=GOLDEN.sha256(saved / "pattern.bin"),
            output_sha256=GOLDEN.sha256(output_control),
            output_digest=GOLDEN.fnv_file(output_control), matches_golden="false",
        ))
        if any(row["output_digest"] == saved_digest for row in records[1:]):
            raise ValueError("changed input or output escaped the digest check")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
        if args.write:
            args.output.write_text(buffer.getvalue())
        elif args.output.read_text() != buffer.getvalue():
            raise ValueError("control results differ from the recorded CSV")
    print("baseline matched; source, pattern, and output controls all differed")


if __name__ == "__main__":
    main()
