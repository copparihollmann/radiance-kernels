#!/usr/bin/env python3
"""Record native checks for the full-depth, reduced-size decoder programs."""

import argparse
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
MODELS = {"tinyllama": 22, "deepseek_r1_distill_qwen_1_5b": 28,
          "gemma_2_2b_it": 26}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def record(generated_root: Path) -> list[dict]:
    results = []
    for model, layers in MODELS.items():
        target = generated_root / model
        manifest = json.loads((target / "manifest.json").read_text())
        if (manifest["model"] != model or manifest["layers"] != layers
                or manifest["prefill_tokens"] != 1
                or manifest["decode_steps"] != 1
                or not manifest["device_check_all_stages"]
                or manifest["checkpoint_weights"]
                or manifest["full_model_dimensions"]):
            raise ValueError(f"{model}: unexpected build manifest")
        log = (target / "native.log").read_text().strip()
        match = re.fullmatch(r"errors=0 max_abs_error=([0-9.eE+-]+)", log)
        if not match:
            raise ValueError(f"{model}: native check did not pass: {log}")
        if manifest["native_verified_float_stages"] != manifest["stages"]:
            raise ValueError(f"{model}: some stages were not checked")
        results.append({
            "model": model,
            "scope": "full_layer_count_reduced_synthetic_decoder",
            "layers": layers,
            "prefill_tokens": 1,
            "decode_steps": 1,
            "stages": manifest["stages"],
            "checked_float_stages": manifest["native_verified_float_stages"],
            "checked_integer_stages": manifest["native_verified_integer_stages"],
            "native_max_abs_error": float(match.group(1)),
            "native_passed": True,
            "native_binary_sha256": digest(target / "native"),
            "generated_source_sha256": digest(target / "kernel.cpp"),
            "device_source_files_sha256": {
                name: digest(target / name)
                for name in manifest.get("device_source_files", ["kernel.cpp"])},
            "device_optimization": manifest.get("device_optimization", "O3"),
            "stages_per_device_object": manifest.get("stages_per_device_object", 0),
            "native_log_sha256": digest(target / "native.log"),
            "portable_ops_sha256": digest(HERE / "pipeline_math.hpp"),
            "reference_sha256": digest(HERE.parent / "evaluation/llm/reference.py"),
            "model_specs_sha256": manifest["source_spec_sha256"],
            "checkpoint_weights": False,
            "original_model_dimensions": False,
            "radiance_device_execution": False,
            "performance_measurement": False,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", type=Path,
                        default=HERE / "generated/full-depth")
    parser.add_argument("--out", type=Path,
                        default=HERE / "evaluation/full-depth-native-results.json")
    args = parser.parse_args()
    results = record(args.generated_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(f"{result['model']}: {result['stages']} stages, "
              f"max error {result['native_max_abs_error']:.8g}")


if __name__ == "__main__":
    main()
