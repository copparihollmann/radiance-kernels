#!/usr/bin/env python3
"""Verify a completed no-reference Cyclotron run against upstream SpectFormer."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path

import numpy as np

from prepare import sha256
from reference import load_input
from run_reusable import scan_cyclotron_log, upstream_output


def verify(target: Path, input_path: Path, out: Path,
           simulator_root: Path) -> dict:
    target = target.resolve()
    out = out.resolve()
    build_path = target / "manifest.json"
    run_path = out / "result.json"
    build = json.loads(build_path.read_text())
    run = json.loads(run_path.read_text())
    if not build.get("runtime_input") or not build.get("device_elf_built"):
        raise ValueError("expected a built runtime-input program")
    if run.get("reference_checked") or run.get("status") not in (
            "passed", "completed_unverified"):
        raise ValueError("expected a completed no-reference run")
    if run.get("exit_code") != 0 or run.get("cycles_functional") is None:
        raise ValueError("run did not complete successfully")
    image_path = Path(build["source_image_manifest"])
    image = json.loads(image_path.read_text())
    if run["model"] != build["model"] or run["image_size"] != image["image_size"]:
        raise ValueError("run model scope differs from the build or export")
    pixels = load_input(input_path, image["image_size"])
    input_digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    weights_path = image_path.parent / "weights.bin"
    checks = (
        ("input", input_digest, run["input_sha256"]),
        ("weights", sha256(weights_path), build["weights_sha256"]),
        ("export weights", image["weights_sha256"], build["weights_sha256"]),
        ("run weights", run["weights_sha256"], build["weights_sha256"]),
        ("checkpoint", run["checkpoint_sha256"], image["checkpoint_sha256"]),
        ("ELF", sha256(target / "kernel.radiance.elf"), build["radiance_elf_sha256"]),
        ("run ELF", run["elf_sha256"], build["radiance_elf_sha256"]),
        ("simulator", sha256(simulator_root / "target/release/cyclotron"),
         run["simulator_sha256"]),
        ("config", sha256(out / "cyclotron-config.toml"), run["config_sha256"]),
        ("log", sha256(out / "cyclotron.log"), run["log_sha256"]),
    )
    for label, actual, recorded in checks:
        if actual != recorded:
            raise ValueError(f"{label} hash differs from the run receipt")
    events = scan_cyclotron_log(out / "cyclotron.log",
                                weights_path.stat().st_size, image["base_address"],
                                pixels.nbytes, build["input_address"])
    if (events["passed"] != 1 or events["weights"] != 1 or
            events["input"] != 1 or
            events["cycles"] != [run["cycles_functional"]]):
        raise ValueError("run log does not match completion or preload addresses")

    kind = build.get("output_kind", "logits")
    count = build.get("output_count", image["num_classes"])
    if run["output_kind"] != kind or run["output_count"] != count:
        raise ValueError("run output scope differs from the build")
    output_path = out / ("logits.bin" if kind == "logits" else "output.bin")
    if output_path.stat().st_size != count * 4:
        raise ValueError("device output has the wrong size")
    if (run.get("output_sha256") is not None and
            sha256(output_path) != run["output_sha256"]):
        raise ValueError("device output hash differs from the run receipt")
    actual = np.fromfile(output_path, dtype="<f4")
    expected = upstream_output(image, pixels, kind)
    if expected.size != count:
        raise ValueError("upstream output has the wrong size")
    reference_path = out / "verified-reference.bin"
    reference_path.write_bytes(expected.tobytes())
    finite = bool(np.isfinite(actual).all())
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    rtol, atol = build["logit_rtol"], build["logit_atol"]
    bad = (int(np.count_nonzero(np.abs(delta) > atol + rtol * np.abs(expected)))
           if finite else count)
    passed = finite and bad == 0
    record = {
        "model": build["model"],
        "status": "passed" if passed else "failed",
        "scope": kind,
        "output_count": count,
        "run_result_sha256": sha256(run_path),
        "build_manifest_sha256": sha256(build_path),
        "image_manifest_sha256": sha256(image_path),
        "input_sha256": input_digest,
        "checkpoint_sha256": image["checkpoint_sha256"],
        "weights_sha256": run["weights_sha256"],
        "elf_sha256": run["elf_sha256"],
        "simulator_sha256": run["simulator_sha256"],
        "output_sha256": sha256(output_path),
        "reference_sha256": sha256(reference_path),
        "reference_environment": {
            package: metadata.version(package) for package in
            ("torch", "timm", "numpy")
        },
        "all_finite": finite,
        "max_abs_error": float(np.max(np.abs(delta))) if finite else None,
        "rms_error": float(np.sqrt(np.mean(delta * delta))) if finite else None,
        "out_of_tolerance": bad,
        "rtol": rtol,
        "atol": atol,
        "top1_agrees": (bool(np.argmax(actual) == np.argmax(expected))
                         if finite and kind == "logits" else None),
        "cycles_functional": run["cycles_functional"],
        "reference_checked": True,
        "rtl_execution": False,
        "performance_measurement": False,
    }
    (out / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cyclotron-root", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.target, args.input, args.out, args.cyclotron_root)
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
