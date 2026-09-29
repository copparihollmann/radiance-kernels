#!/usr/bin/env python3
"""Check Radiance output against unmodified upstream Spatter serial kernels.

The C++ driver links hpcgarage/spatter Configuration.cc and Timer.cc at the
pinned revision. It consumes the same deterministic 64-bit source payload and
address patterns as the Radiance ELF, then writes upstream's complete output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import FNV_OFFSET, FNV_PRIME, MASK64, normalize, write_source  # noqa: E402

UPSTREAM_REVISION = "ec8923711f8dc21eedff7189f12b02eb06845d2f"
FIELDS = (
    "run", "kind", "pattern_length", "count", "collision_policy",
    "upstream_revision", "upstream_source_sha256", "pattern_validation", "suite_sha256",
    "radiance_elf_sha256", "source_bytes", "source_sha256",
    "golden_output_bytes", "golden_output_sha256", "golden_digest",
    "radiance_expected_digest", "cyclotron_digest", "status",
)
ARTIFACT_FIELDS = ("path", "bytes", "sha256")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upstream_source_hash(upstream: Path) -> str:
    digest = hashlib.sha256()
    for name in ("Configuration.cc", "Configuration.hh", "Timer.cc", "Timer.hh",
                 "PatternParser.cc", "PatternParser.hh", "AlignedAllocator.hh",
                 "SpatterTypes.hh"):
        digest.update(name.encode())
        digest.update((upstream / "src/Spatter" / name).read_bytes())
    digest.update((ROOT / "tools/upstream-golden.cc").read_bytes())
    return digest.hexdigest()


def compile_driver(upstream: Path, output: Path) -> None:
    command = ["c++", "-std=c++17", "-O2", "-I", str(upstream / "src/Spatter"),
               str(ROOT / "tools/upstream-golden.cc"),
               str(upstream / "src/Spatter/Configuration.cc"),
               str(upstream / "src/Spatter/Timer.cc"),
               str(upstream / "src/Spatter/PatternParser.cc"), "-o", str(output)]
    subprocess.run(command, check=True)


def write_pattern(path: Path, indices: list[int]) -> None:
    with path.open("wb") as output:
        for begin in range(0, len(indices), 65536):
            block = indices[begin:begin + 65536]
            output.write(struct.pack(f"<{len(block)}I", *block))


def fnv_file(path: Path) -> str:
    digest = FNV_OFFSET
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 65536), b""):
            if len(block) % 8:
                raise ValueError(f"golden output has a partial 64-bit word: {path}")
            for (value,) in struct.iter_unpack("<Q", block):
                digest = ((digest ^ (value & 0xFFFFFFFF)) * FNV_PRIME) & MASK64
                digest = ((digest ^ (value >> 32)) * FNV_PRIME) & MASK64
    return f"{digest:016x}"


def validate_case(build: dict, raw: dict) -> dict:
    raw = dict(raw)
    raw["count"] = build["count"]
    raw["collision-policy"] = build.get("collision_policy", "parallel")
    case = normalize(raw)
    for field, value in (("kind", case["kind"]), ("count", case["count"]),
                         ("pattern_length", case["length"]),
                         ("source_elements", case["src_length"]),
                         ("output_elements", case["dst_length"]),
                         ("source_tag", case["payload_tag"])):
        if field in build and build[field] != value:
            raise ValueError(f"build differs from input suite at {field}")
    if case["gather_final_wrap"]:
        raise ValueError("fused composition has no direct upstream Spatter kernel")
    if build["destination_overlap"] and case["collision_policy"] != "ordered":
        raise ValueError("parallel overlapping Scatter has no deterministic golden output")
    return case


def check_patterns(raw: dict, case: dict, driver: Path, path: Path) -> str:
    checked = 0
    direct = 0
    for key, values, initial_delta, final_delta in (
        ("pattern", case["pattern"], raw.get("delta", 8), case["delta"]),
        ("pattern-gather", case["gather"], raw.get("delta-gather", 8),
         case["delta_gather"]),
        ("pattern-scatter", case["scatter"], raw.get("delta-scatter", 8),
         case["delta_scatter"]),
    ):
        if key not in raw:
            continue
        spec = raw[key]
        if isinstance(spec, list) and len(spec) > 1024:
            # Upstream reads JSON arrays directly, without a generator.
            # These exact integers are already in the pattern binary.
            direct += 1
            continue
        text = ",".join(map(str, spec)) if isinstance(spec, list) else spec
        parsed = path / f"upstream-{key}.bin"
        command = [str(driver), "--parse-pattern", text, str(initial_delta),
                   str(raw.get("pattern-size", 0)), str(parsed)]
        parsed_delta = int(subprocess.check_output(command, text=True).strip())
        expected = path / f"{key.removeprefix('pattern-') if key != 'pattern' else 'pattern'}.bin"
        if parsed.read_bytes() != expected.read_bytes() or parsed_delta != final_delta:
            raise ValueError(f"upstream parser differs on {key}")
        checked += 1
    return f"{checked} upstream parsed, {direct} direct JSON arrays"


def one_case(name: str, upstream: Path, driver: Path, source_root: Path,
             model_root: Path, output_root: Path, source_hash: str) -> dict:
    build_dir = source_root / name
    build = json.loads((build_dir / "result.json").read_text())
    model = json.loads((model_root / name / "result.json").read_text())
    if model["status"] != "passed":
        raise ValueError(f"{name}: complete Cyclotron check is required")
    suite = Path(build["suite"])
    if sha256(suite) != build["suite_sha256"]:
        raise ValueError(f"{name}: input suite changed")
    raw = json.loads(suite.read_text())[build["case"]]
    case = validate_case(build, raw)
    elf_hash = sha256(build_dir / "kernel.soc.elf")
    if elf_hash != model["elf_sha256"]:
        raise ValueError(f"{name}: build and model ELF differ")
    path = output_root / name
    path.mkdir(parents=True, exist_ok=True)
    write_pattern(path / "pattern.bin", case["pattern"])
    write_pattern(path / "gather.bin", case["gather"])
    write_pattern(path / "scatter.bin", case["scatter"])
    pattern_validation = check_patterns(raw, case, driver, path)
    write_source(path / "source.bin", case["src_length"], case["payload_tag"])
    output = path / "output.bin"
    command = [str(driver), case["kind"], str(case["count"]), str(case["wrap"]),
               str(case["delta"]), str(case["delta_gather"]),
               str(case["delta_scatter"]), str(case["src_length"]),
               str(case["dst_length"]), str(path / "pattern.bin"),
               str(path / "gather.bin"), str(path / "scatter.bin"),
               str(path / "source.bin"), str(output)]
    with (path / "upstream.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    actual = fnv_file(output)
    if actual != build["expected_digest"] or actual != model["output_digest"]:
        raise ValueError(f"{name}: upstream {actual}, Radiance expectation "
                         f"{build['expected_digest']}, Cyclotron {model['output_digest']}")
    row = {
        "run": name, "kind": case["kind"], "pattern_length": case["length"],
        "count": case["count"], "collision_policy": case["collision_policy"],
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_source_sha256": source_hash,
        "pattern_validation": pattern_validation,
        "suite_sha256": build["suite_sha256"], "radiance_elf_sha256": elf_hash,
        "source_bytes": (path / "source.bin").stat().st_size,
        "source_sha256": sha256(path / "source.bin"),
        "golden_output_bytes": output.stat().st_size,
        "golden_output_sha256": sha256(output),
        "golden_digest": actual,
        "radiance_expected_digest": build["expected_digest"],
        "cyclotron_digest": model["output_digest"], "status": "passed",
    }
    (path / "result.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--model-root", type=Path, default=ROOT / "runs/model")
    parser.add_argument("--output-root", type=Path, default=ROOT / "golden-runs")
    parser.add_argument("--table", type=Path,
                        default=ROOT / "evaluation/upstream-golden-results.csv")
    parser.add_argument("--artifact-table", type=Path,
                        default=ROOT / "evaluation/upstream-golden-artifacts.csv")
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"upstream revision {revision} differs from pinned {UPSTREAM_REVISION}")
    source_hash = upstream_source_hash(upstream)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    driver = output_root / "upstream-golden"
    stamp = output_root / "upstream-source.sha256"
    if not driver.is_file() or not stamp.is_file() or stamp.read_text().strip() != source_hash:
        compile_driver(upstream, driver)
        stamp.write_text(source_hash + "\n")
    rows = [one_case(name, upstream, driver, args.source_root.resolve(),
                     args.model_root.resolve(), output_root, source_hash)
            for name in args.names]
    args.table.parent.mkdir(parents=True, exist_ok=True)
    with args.table.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    artifacts = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            artifacts.append({"path": path.relative_to(output_root).as_posix(),
                              "bytes": path.stat().st_size, "sha256": sha256(path)})
    with args.artifact_table.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=ARTIFACT_FIELDS,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(artifacts)
    print(f"upstream Spatter and Radiance/Cyclotron agree on {len(rows)} cases")


if __name__ == "__main__":
    main()
