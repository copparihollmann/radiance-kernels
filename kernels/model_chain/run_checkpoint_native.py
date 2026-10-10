#!/usr/bin/env python3
"""Run a full-checkpoint decoder's generated Radiance stage C++ on the host.

The generated stage and verifier sources are the same ones linked into the
Radiance ELF. This checks the mapped arithmetic and stage order against every
generated NumPy golden tensor. It is not device, RTL, or performance evidence.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from compile_decoder import write_if_changed
from split_decoder_weights import image_segments, verify_image, verify_image_placement


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernels/evaluation/llm"))
from stitch import model_specs  # noqa: E402

MODELS = ("tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma_2_2b_it")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(target: Path, manifest: dict, image: dict, image_paths: list[Path]) -> Path:
    native = target / "native-check"
    native.mkdir(exist_ok=True)
    expected_sources = {"kernel.cpp", "model_data.cpp"}
    expected_sources.update(path.name for path in target.glob("stage_chunk_*.cpp"))
    expected_sources.update(path.name for path in target.glob("verify_chunk_*.cpp"))
    if set(manifest["device_source_files"]) != expected_sources:
        raise ValueError("native sources differ from the ELF build manifest")
    write_if_changed(native / "mu_intrinsics.h", "#pragma once\n#define __global\n"
                     "#define MU_NUM_THREADS 32\n#define MU_NUM_CORES 1\n")
    write_if_changed(native / "mu_schedule.h", "#pragma once\n")
    write_if_changed(native / "kernel_verify.h", """#pragma once
#include <cmath>
#include <cstdint>
extern uint32_t native_tohost;
inline bool mu_close(float actual, float expected, float rel, float abs_) {
  return std::isfinite(actual) &&
         std::fabs(actual - expected) <= rel * std::fabs(expected) + abs_;
}
inline void mu_tohost(uint32_t code) { native_tohost = code; }
""")
    stages = manifest["stages"]
    checks = len(manifest["verified_tensors"])
    chunks = sorted(target.glob("verify_chunk_*.cpp"))
    if (not chunks or checks != stages or
            checks != (manifest["native_verified_float_stages"] +
                       manifest["native_verified_integer_stages"])):
        raise ValueError("full device-stage verifier sources are missing")
    prototypes = "\n".join(
        [f"void stage_{i}(void*, uint32_t, uint32_t, uint32_t);"
         for i in range(stages)] +
        [f"bool {path.stem}();" for path in chunks])
    stage_entries = ", ".join(f"stage_{i}" for i in range(stages))
    verify_entries = ", ".join(path.stem for path in chunks)
    preloads = []
    for segment, path in zip(image_segments(image), image_paths):
        preloads.append(
            f"  if (!map_image({segment['gpu_base_address']}ull, "
            f"{segment['image_size_bytes']}ull, "
            f"{json.dumps(str(path.resolve()))})) return 2;")
    source = """#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
uint32_t native_tohost = 0;
""" + prototypes + "\n" + """
using Stage = void (*)(void*, uint32_t, uint32_t, uint32_t);
using Check = bool (*)();
static const Stage stages[] = {""" + stage_entries + "};\n" + \
             "static const Check checks[] = {" + verify_entries + "};\n" + """
static bool map_image(uint64_t base, uint64_t size, const char* path) {
  int file = open(path, O_RDONLY);
  if (file < 0) { perror(path); return false; }
  void* memory = mmap((void*)base, size, PROT_READ,
                      MAP_PRIVATE | MAP_FIXED_NOREPLACE, file, 0);
  close(file);
  if (memory == MAP_FAILED || (uint64_t)memory != base) {
    perror("mmap image"); return false;
  }
  return true;
}
int main() {
""" + "\n".join(preloads) + "\n" + """
  constexpr uint32_t stage_count = sizeof(stages) / sizeof(stages[0]);
  constexpr uint32_t check_chunks = sizeof(checks) / sizeof(checks[0]);
  for (uint32_t i = 0; i < stage_count; ++i) {
    #pragma omp parallel for schedule(static)
    for (uint32_t tid = 0; tid < 32; ++tid)
      stages[i](nullptr, tid, 32, 0);
    if ((i + 1) % 100 == 0 || i + 1 == stage_count) {
      fprintf(stderr, "completed %u/%u stages\\n", i + 1, stage_count);
      fflush(stderr);
    }
  }
  for (uint32_t i = 0; i < check_chunks; ++i) {
    if (!checks[i]()) {
      printf("stage_check_failed chunk=%u tohost=%u\\n", i, native_tohost);
      return 1;
    }
  }
  printf("stages=%u check_chunks=%u tohost=0\\n", stage_count, check_chunks);
  return 0;
}
"""
    write_if_changed(native / "main.cpp", source)
    return native


def build(target: Path, native: Path, jobs: int) -> Path:
    stage_sources = sorted(target.glob("stage_chunk_*.cpp"))
    verify_sources = sorted(target.glob("verify_chunk_*.cpp"))
    sources = [target / "model_data.cpp", *stage_sources, *verify_sources,
               native / "main.cpp"]
    compiler = ["g++", "-std=c++17", "-fopenmp", "-I", str(native),
                "-I", str(HERE)]
    dependencies = [native / "mu_intrinsics.h", native / "mu_schedule.h",
                    native / "kernel_verify.h", HERE / "pipeline_math.hpp",
                    HERE / "pipeline_int8.hpp", HERE / "pipeline_tied.hpp"]

    def compile_one(source: Path) -> Path:
        obj = native / (source.stem + ".o")
        if (not obj.exists() or obj.stat().st_mtime <
                max(path.stat().st_mtime for path in [source, *dependencies])):
            optimization = "-O0" if source.name == "model_data.cpp" else "-O2"
            process = subprocess.run(compiler + [optimization, "-c", str(source),
                                               "-o", str(obj)],
                                     capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError(f"{source}: {process.stdout}{process.stderr}")
        return obj

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        objects = list(executor.map(compile_one, sources))
    binary = native / "checkpoint_native"
    subprocess.run(["g++", "-fopenmp", *map(str, objects), "-o", str(binary)],
                   check=True)
    return binary


def run(model: str, generated_root: Path, threads: int, jobs: int,
        out: Path | None, allow_partial: bool = False) -> dict:
    target = (generated_root / model).resolve()
    manifest = json.loads((target / "manifest.json").read_text())
    specs = model_specs()
    full_layers = specs[model]["num_hidden_layers"]
    if (manifest["model"] != model or not manifest["device_elf_built"] or
            not manifest["checkpoint_weights"] or
            not manifest["device_check_all_stages"] or
            manifest["stage_limit"] is not None or
            not 1 <= manifest["layers"] <= full_layers or
            (manifest["layers"] != full_layers and not allow_partial)):
        raise ValueError("native check requires a full-depth checkpoint ELF "
                         "unless --allow-partial is set")
    elf = target / "kernel.radiance.elf"
    if sha256(elf) != manifest["radiance_elf_sha256"]:
        raise ValueError("ELF differs from its build manifest")
    image, image_paths = verify_image(Path(manifest["weight_image_manifest"]))
    if image["image_sha256"] != manifest["weight_image_sha256"]:
        raise ValueError("checkpoint image differs from its build manifest")
    verify_image_placement(elf, [image])
    native = generate(target, manifest, image, image_paths)
    binary = build(target, native, jobs)
    log = native / "native.log"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    with log.open("w") as output:
        process = subprocess.run([str(binary)], stdout=output,
                                 stderr=subprocess.STDOUT, env=env)
    log_text = log.read_text()
    success = re.search(r"stages=(\d+) check_chunks=(\d+) tohost=0", log_text)
    failure = re.search(r"stage_check_failed chunk=(\d+) tohost=(\d+)", log_text)
    passed = (process.returncode == 0 and success is not None and
              int(success.group(1)) == manifest["stages"] and
              int(success.group(2)) == len(list(target.glob("verify_chunk_*.cpp"))))
    record = {
        "model": model,
        "scope": ("full_checkpoint_generated_cpp_on_cpu"
                  if manifest["layers"] == full_layers else
                  "partial_checkpoint_generated_cpp_on_cpu"),
        "layers": manifest["layers"], "stage_count": manifest["stages"],
        "checked_float_stages": manifest["native_verified_float_stages"],
        "checked_integer_stages": manifest["native_verified_integer_stages"],
        "prefill_tokens": manifest["prefill_tokens"],
        "decode_steps": manifest["decode_steps"],
        "input_token_ids": manifest.get("input_token_ids"),
        "checkpoint_weight_format": manifest["checkpoint_weight_format"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "weight_image_sha256": image["image_sha256"],
        "reference_output_sha256": manifest["reference_output_sha256"],
        "comparison_tolerance": {"rtol": 5e-3, "atol": 5e-4},
        "status": "passed" if passed else "failed",
        "process_exit_code": process.returncode,
        "failed_check_chunk": int(failure.group(1)) if failure else None,
        "tohost": int(failure.group(2)) if failure else 0 if passed else None,
        "host_threads": threads,
        "native_binary_sha256": sha256(binary),
        "native_main_sha256": sha256(native / "main.cpp"),
        "device_elf_sha256": sha256(elf),
        "device_source_files_sha256": {
            name: sha256(target / name)
            for name in manifest["device_source_files"]},
        "model_specs_sha256": manifest["source_spec_sha256"],
        "log_sha256": sha256(log), "log_path": str(log),
        "log_tail": log_text[-1200:],
        "device_execution": False, "rtl_execution": False,
        "performance_measurement": False,
    }
    destination = out or native / "native-result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{model}: {record['status']} ({record['stage_count']} stages)")
    if not passed:
        raise SystemExit(process.returncode or 1)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-partial", action="store_true",
                        help="check all stages of a selected checkpoint layer subset")
    args = parser.parse_args()
    if args.threads <= 0 or args.threads > 32 or args.jobs <= 0:
        parser.error("threads must be 1..32 and jobs must be positive")
    run(args.model, args.generated_root, args.threads, args.jobs, args.out,
        args.allow_partial)


if __name__ == "__main__":
    main()
