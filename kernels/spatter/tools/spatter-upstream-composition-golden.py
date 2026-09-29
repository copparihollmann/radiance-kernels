#!/usr/bin/env python3
"""Check full-size Gather→Scatter composition with upstream serial Spatter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run import fnv, normalize, write_source, write_u32  # noqa: E402
from run_chain import gather_slot  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "spatter_upstream_golden", TOOLS / "spatter-upstream-golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

FIELDS = (
    "run", "kind", "count", "pattern_length", "intermediate_elements",
    "upstream_revision", "upstream_source_sha256", "driver_script_sha256",
    "original_suite_sha256", "fused_suite_sha256",
    "chain_elf_sha256", "fused_elf_sha256",
    "source_bytes", "source_sha256", "gather_pattern_sha256",
    "scatter_pattern_sha256", "intermediate_bytes", "intermediate_sha256",
    "intermediate_digest", "golden_output_bytes", "golden_output_sha256",
    "golden_digest", "chain_expected_digest", "chain_model_digest",
    "fused_expected_digest", "fused_model_digest", "status",
)


def run_stage(driver: Path, kind: str, case: dict, pattern: Path,
              dummy: Path, source: Path, output: Path, log: Path) -> None:
    command = [str(driver), kind, str(case["count"]), str(case["wrap"]),
               str(case["delta"]), "0", "0", str(case["src_length"]),
               str(case["dst_length"]), str(pattern), str(dummy), str(dummy),
               str(source), str(output)]
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def checked_run(root: Path, name: str) -> tuple[dict, dict, str]:
    build_path = root / name
    model_path = root / "model" / name
    build = json.loads((build_path / "result.json").read_text())
    model = json.loads((model_path / "result.json").read_text())
    elf_hash = golden.sha256(build_path / "kernel.soc.elf")
    if (build["status"] != "built" or model["status"] != "passed" or
            model["correctness"] != "digest-checked" or
            model["elf_sha256"] != elf_hash or
            golden.sha256(model_path / "kernel.soc.elf") != elf_hash or
            build["expected_digest"] != model["output_digest"] or
            build["kernel_source_sha256"] != model["kernel_source_sha256"]):
        raise ValueError(f"{name}: build and complete model check differ")
    return build, model, elf_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--golden-root", type=Path, default=ROOT / "golden-runs")
    parser.add_argument("--table", type=Path,
                        default=ROOT / "evaluation/upstream-golden-composition-results.csv")
    parser.add_argument("--artifact-table", type=Path,
                        default=ROOT / "evaluation/upstream-golden-artifacts.csv")
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != golden.UPSTREAM_REVISION:
        raise ValueError(f"upstream revision {revision} differs from pinned source")
    source_hash = golden.upstream_source_hash(upstream)
    root = args.build_root.resolve()
    chain_name = "chain-gpu-stream-no-fence"
    fused_name = "composed-gpu-stream-current"
    chain_build, chain_model, chain_elf = checked_run(root, chain_name)
    fused_build, fused_model, fused_elf = checked_run(root, fused_name)

    suite_path = ROOT / "inputs/standard-suite/basic-tests/gpu-stream.json"
    fused_suite = ROOT / "inputs/composed-gpu-stream.json"
    suite = json.loads(suite_path.read_text())
    gather, scatter = normalize(suite[0]), normalize(suite[1])
    if (gather["kind"] != "gather" or scatter["kind"] != "scatter" or
            gather["dst_length"] != scatter["src_length"] or
            chain_build["case"] != [0, 1] or fused_build["case"] != 0 or
            chain_build["count"] != scatter["count"] or
            chain_build["source_elements"] != gather["src_length"] or
            chain_build["intermediate_elements"] != gather["dst_length"] or
            chain_build["output_elements"] != scatter["dst_length"]):
        raise ValueError("the full-size chain differs from the original two cases")
    suite_hash = hashlib.sha256(suite_path.read_bytes() * 2).hexdigest()
    fused_suite_hash = golden.sha256(fused_suite)
    if (chain_build["suite_sha256"] != suite_hash or
            fused_build["suite_sha256"] != fused_suite_hash):
        raise ValueError("build suite hashes differ from tracked inputs")

    golden_root = args.golden_root.resolve()
    golden_root.mkdir(parents=True, exist_ok=True)
    driver = golden_root / "upstream-golden"
    stamp = golden_root / "upstream-source.sha256"
    if not driver.is_file() or not stamp.is_file() or stamp.read_text().strip() != source_hash:
        golden.compile_driver(upstream, driver)
        stamp.write_text(source_hash + "\n")
    out = golden_root / "composition-fullsize"
    out.mkdir(parents=True, exist_ok=True)
    source, intermediate, output = (out / "source.bin", out / "intermediate.bin",
                                    out / "output.bin")
    gather_pattern, scatter_pattern, dummy = (
        out / "gather-pattern.bin", out / "scatter-pattern.bin", out / "dummy.bin")
    write_source(source, gather["src_length"], gather["payload_tag"])
    write_u32(gather_pattern, gather["pattern"])
    write_u32(scatter_pattern, scatter["pattern"])
    write_u32(dummy, [])
    run_stage(driver, "gather", gather, gather_pattern, dummy, source,
              intermediate, out / "gather.log")
    run_stage(driver, "scatter", scatter, scatter_pattern, dummy, intermediate,
              output, out / "scatter.log")

    intermediate_digest = golden.fnv_file(intermediate)
    expected_intermediate = f"{fnv(gather_slot(gather, j) for j in range(gather['dst_length'])):016x}"
    digest = golden.fnv_file(output)
    if (intermediate.stat().st_size != gather["dst_length"] * 8 or
            output.stat().st_size != scatter["dst_length"] * 8 or
            intermediate_digest != expected_intermediate or
            len({digest, chain_build["expected_digest"], chain_model["output_digest"],
                 fused_build["expected_digest"], fused_model["output_digest"]}) != 1):
        raise ValueError("upstream serial stages differ from Radiance composition")

    row = {
        "run": chain_name, "kind": "gather-scatter-chain",
        "count": str(scatter["count"]), "pattern_length": str(scatter["length"]),
        "intermediate_elements": str(gather["dst_length"]),
        "upstream_revision": golden.UPSTREAM_REVISION,
        "upstream_source_sha256": source_hash,
        "driver_script_sha256": golden.sha256(Path(__file__)),
        "original_suite_sha256": golden.sha256(suite_path),
        "fused_suite_sha256": fused_suite_hash,
        "chain_elf_sha256": chain_elf, "fused_elf_sha256": fused_elf,
        "source_bytes": str(source.stat().st_size), "source_sha256": golden.sha256(source),
        "gather_pattern_sha256": golden.sha256(gather_pattern),
        "scatter_pattern_sha256": golden.sha256(scatter_pattern),
        "intermediate_bytes": str(intermediate.stat().st_size),
        "intermediate_sha256": golden.sha256(intermediate),
        "intermediate_digest": intermediate_digest,
        "golden_output_bytes": str(output.stat().st_size),
        "golden_output_sha256": golden.sha256(output),
        "golden_digest": digest, "chain_expected_digest": chain_build["expected_digest"],
        "chain_model_digest": chain_model["output_digest"],
        "fused_expected_digest": fused_build["expected_digest"],
        "fused_model_digest": fused_model["output_digest"], "status": "passed",
    }
    (out / "result.json").write_text(json.dumps(row, indent=2) + "\n")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    with args.table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    artifacts = []
    for path in sorted(golden_root.rglob("*")):
        if path.is_file():
            artifacts.append({"path": path.relative_to(golden_root).as_posix(),
                              "bytes": path.stat().st_size,
                              "sha256": golden.sha256(path)})
    with args.artifact_table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=golden.ARTIFACT_FIELDS,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(artifacts)
    print(f"upstream serial Gather→Scatter and Radiance agree: digest={digest}")


if __name__ == "__main__":
    main()
