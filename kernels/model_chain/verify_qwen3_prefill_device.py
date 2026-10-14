#!/usr/bin/env python3
"""Compare executed Qwen3 full-logit and last-prefill-logit device buffers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np


NAMES = ("prefill.final_norm", "prefill.lm_head", "decode0.token_ids",
         "decode0.lm_head", "decode1.token_ids", "decode1.lm_head")
OPTIMIZED_EXTRA = "prefill.last_hidden"
COUNTS = {"prefill.final_norm": 3 * 32, "prefill.lm_head": 3 * 64,
          "decode0.token_ids": 1, "decode0.lm_head": 64,
          "decode1.token_ids": 1, "decode1.lm_head": 64}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symbol(name: str) -> str:
    return "v_" + name.replace(".", "_")


def run(root: Path, label: str, simulator: Path, nm: Path,
        out_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    target = root / "qwen3_0_6b"
    elf = target / "kernel.radiance.elf"
    config = root / "cyclotron-config.toml"
    manifest = json.loads((target / "manifest.json").read_text())
    receipt = json.loads((target / "functional-result.json").read_text())
    policy = "last" if label == "last" else "all"
    if (manifest["model"] != "qwen3_0_6b" or manifest["layers"] != 2
            or manifest["prefill_tokens"] != 3 or manifest["decode_steps"] != 2
            or manifest["generation"] != "greedy"
            or manifest["input_token_ids"] != {"prefill.token_ids": [1, 2, 3]}
            or manifest["prefill_logits_policy"] != policy
            or receipt["status"] != "passed" or receipt["tohost"] != 0
            or receipt["prefill_logits_policy"] != policy
            or sha(elf) != receipt["device_elf_sha256"]):
        raise ValueError(f"{label}: build and completed run are not the matched control")
    names = NAMES + ((OPTIMIZED_EXTRA,) if label == "last" else ())
    counts = dict(COUNTS)
    if label == "last":
        counts["prefill.lm_head"] = 64
        counts[OPTIMIZED_EXTRA] = 32
    listing = subprocess.check_output([str(nm), "-n", str(elf)], text=True)
    addresses = {name: int(address, 16) for address, name in re.findall(
        r"(?m)^([0-9a-f]+) [BD] (\w+)$", listing)}
    if any(symbol(name) not in addresses for name in names):
        raise ValueError(f"{label}: output symbol missing from ELF")
    begin = min(addresses[symbol(name)] for name in names)
    end = max(addresses[symbol(name)] + counts[name] * 4 for name in names)
    dump_path = out_dir / f"{label}-gmem.bin"
    log_path = out_dir / f"{label}-cyclotron.log"
    env = os.environ.copy()
    env["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
    env["CYCLOTRON_DUMP_GMEM"] = f"0x{begin:x}:{end - begin}:{dump_path}"
    with log_path.open("w") as log:
        process = subprocess.run([str(simulator), str(config), "--binary-path",
                                  str(elf), "--gen-trace", "false"],
                                 cwd=simulator.parent.parent.parent, env=env,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 timeout=300, check=False)
    output = log_path.read_text()
    cycles = re.findall(r"simulation finished after (\d+) cycles", output)
    if (process.returncode or output.count("isa-test passed with tohost=0") != 1
            or len(cycles) != 1 or int(cycles[0]) != receipt["cycles_functional"]):
        raise RuntimeError(f"{label}: Cyclotron dump run did not match saved receipt")
    dump = dump_path.read_bytes()
    if len(dump) != end - begin:
        raise ValueError(f"{label}: incomplete GMEM dump")
    values = {}
    for name in names:
        offset = addresses[symbol(name)] - begin
        dtype = "<i4" if name.endswith("token_ids") else "<f4"
        values[name] = np.frombuffer(dump[offset:offset + counts[name] * 4],
                                     dtype=dtype).copy()
    return values, {"elf_sha256": sha(elf), "config_sha256": sha(config),
                    "log_sha256": sha(log_path), "gmem_sha256": sha(dump_path),
                    "functional_cycles": int(cycles[0])}


def check(all_root: Path, last_root: Path, simulator: Path, nm: Path,
          out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    original, all_meta = run(all_root, "all", simulator, nm, out_dir)
    selected, last_meta = run(last_root, "last", simulator, nm, out_dir)
    comparisons = {
        "prefill_final_hidden": (original["prefill.final_norm"][-32:],
                                 selected["prefill.last_hidden"]),
        "prefill_final_logits": (original["prefill.lm_head"][-64:],
                                  selected["prefill.lm_head"]),
        "decode0_logits": (original["decode0.lm_head"],
                           selected["decode0.lm_head"]),
        "decode1_logits": (original["decode1.lm_head"],
                           selected["decode1.lm_head"]),
    }
    errors = {name: float(np.max(np.abs(left - right)))
              for name, (left, right) in comparisons.items()}
    ids = {name: [int(original[name][0]), int(selected[name][0])]
           for name in ("decode0.token_ids", "decode1.token_ids")}
    wrong_row_delta = float(np.max(np.abs(
        original["prefill.lm_head"][:64] - selected["prefill.lm_head"])))
    if (any(error != 0.0 for error in errors.values())
            or any(pair[0] != pair[1] for pair in ids.values())
            or wrong_row_delta <= 1e-5
            or all_meta["config_sha256"] != last_meta["config_sha256"]):
        raise AssertionError("executed last-token mapping differs from full-logit control")
    return {
        "schema": "radiance.qwen3-prefill-device-comparison.v1",
        "status": "passed", "scope": "two-layer reduced synthetic FP32 Qwen3, 3 prefill and 2 greedy decode tokens",
        "backend": "cyclotron", "checked_float_elements": 32 + 3 * 64,
        "float_max_abs_error": errors, "greedy_token_ids_all_and_last": ids,
        "wrong_first_prefill_logit_row_max_abs_delta": wrong_row_delta,
        "all_logits": all_meta, "last_logits": last_meta,
        "simulator_sha256": sha(simulator), "nm_sha256": sha(nm),
        "rtl_execution": False, "performance_measurement": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-root", type=Path, required=True)
    parser.add_argument("--last-root", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--nm", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.all_root, args.last_root, args.simulator,
                   args.nm, args.out_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"matched {result['checked_float_elements']} executed float values")


if __name__ == "__main__":
    main()
