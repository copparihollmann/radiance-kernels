#!/usr/bin/env python3
"""Check both generated decoder replicas against checkpoint-bound tensors on a CPU.

This runs the same generated stage sources used by the device ELF. It does not
simulate simultaneous clusters or measure device performance.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from run_checkpoint_native import build, model_specs, sha256
from split_decoder_weights import image_segments, verify_image, verify_image_placement


def checks_from_kernel(path: Path) -> list[tuple[int, str, int, str]]:
    pattern = re.compile(
        r"for \(uint32_t i = 0; i < (\d+); \+\+i\) if \(!mu_close\("
        r"(v_\w+)\[i \+ replica \* (\d+)u\], (v_gold_\w+)\[i\],")
    checks = [
        (int(count), output, int(stride), gold)
        for count, output, stride, gold in pattern.findall(path.read_text())
    ]
    if not checks or len({gold for _, _, _, gold in checks}) != len(checks):
        raise ValueError("expected distinct generated device comparison arrays")
    if any(count > stride or stride % 16 for count, _, stride, _ in checks):
        raise ValueError("invalid replica stride in generated device comparison")
    return checks


def generate(target: Path, manifest: dict, image: dict, image_paths: list[Path]) -> tuple[Path, int]:
    if manifest.get("batch_clusters") != 2 or manifest.get("device_check_all_stages"):
        raise ValueError("expected a two-cluster checkpoint build with selected checks")
    sources = set(manifest["device_source_files"])
    expected = {"kernel.cpp", "model_data.cpp"}
    expected.update(path.name for path in target.glob("stage_chunk_*.cpp"))
    if sources != expected:
        raise ValueError("generated source list differs from the ELF manifest")
    checks = checks_from_kernel(target / "kernel.cpp")
    if len(checks) != len(manifest["verified_tensors"]):
        raise ValueError("device comparisons differ from the build manifest")
    native = target / "native-batch-check"
    native.mkdir(exist_ok=True)
    (native / "mu_intrinsics.h").write_text(
        "#pragma once\n#define __global\n#define MU_NUM_THREADS 32\n#define MU_NUM_CORES 1\n")
    (native / "mu_schedule.h").write_text("#pragma once\n")
    (native / "kernel_verify.h").write_text("#pragma once\n")
    preloads = "\n".join(
        f"  if (!map_image({segment['gpu_base_address']}ull, "
        f"{segment['image_size_bytes']}ull, {json.dumps(str(path.resolve()))})) return 2;"
        for segment, path in zip(image_segments(image), image_paths)
    )
    stages = manifest["stages"]
    prototypes = "\n".join(
        f"void stage_{index}(void*, uint32_t, uint32_t, uint32_t);"
        for index in range(stages))
    symbols = sorted({name for _, output, _, gold in checks for name in (output, gold)})
    externs = "\n".join(f"extern float {name}[];" for name in symbols)
    token_ids = manifest.get("reference_generated_token_ids") or []
    token_externs = "\n".join(f"extern int32_t v_decode{step}_token_ids[];"
                             for step in range(len(token_ids)))
    compare = "\n".join(
        f"    for (uint32_t i = 0; i < {count}u; ++i) {{\n"
        f"      const float actual = {output}[replica * {stride}u + i];\n"
        f"      const float expected = {gold}[i];\n"
        "      const float delta = std::fabs(actual - expected);\n"
        "      if (!std::isfinite(actual) || delta > 5e-3f * std::fabs(expected) + 5e-4f) ++errors;\n"
        "      if (delta > maximum) maximum = delta;\n"
        "    }"
        for count, output, stride, gold in checks)
    token_report = "\n".join(
        f'    printf("replica=%u token_{step}=%d\\n", replica, '
        f"v_decode{step}_token_ids[replica * 16u]);"
        for step in range(len(token_ids)))
    source = """#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
""" + externs + "\n" + token_externs + "\n" + prototypes + "\n" + """
using Stage = void (*)(void*, uint32_t, uint32_t, uint32_t);
static const Stage stages[] = {""" + ", ".join(f"stage_{i}" for i in range(stages)) + "};\n" + """
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
""" + preloads + "\n" + """
  for (uint32_t replica = 0; replica < 2; ++replica) {
    for (uint32_t i = 0; i < sizeof(stages) / sizeof(stages[0]); ++i) {
      #pragma omp parallel for schedule(static)
      for (uint32_t tid = 0; tid < 32; ++tid)
        stages[i]((void*)(uintptr_t)replica, tid, 32, 0);
      if ((i + 1) % 100 == 0 || i + 1 == sizeof(stages) / sizeof(stages[0])) {
        fprintf(stderr, "replica %u completed %u stages\\n", replica, i + 1);
        fflush(stderr);
      }
    }
  }
  uint32_t errors = 0;
  float maximum = 0.0f;
  for (uint32_t replica = 0; replica < 2; ++replica) {
    const uint32_t errors_before = errors;
""" + compare + "\n" + token_report + "\n" + """
    printf("replica=%u errors=%u\\n", replica, errors - errors_before);
  }
  printf("stages=%zu replicas=2 checks=""" + str(len(checks)) + """ errors=%u max_abs_error=%.9g\\n",
         sizeof(stages) / sizeof(stages[0]), errors, maximum);
  return errors ? 1 : 0;
}
"""
    (native / "main.cpp").write_text(source)
    return native, len(checks)


def run(target: Path, *, threads: int = 2, jobs: int = 2) -> dict:
    target = target.resolve()
    manifest = json.loads((target / "manifest.json").read_text())
    if (not manifest.get("checkpoint_weights") or not manifest.get("device_elf_built")
            or manifest.get("stage_limit") is not None or
            manifest.get("scope") != "full_dimension_checkpoint_decoder" or
            manifest.get("layers") != model_specs()[manifest["model"]]["num_hidden_layers"] or
            manifest.get("hbm_launch_clusters") != 2):
        raise ValueError("expected a full-depth built checkpoint program")
    elf = target / "kernel.radiance.elf"
    if sha256(elf) != manifest["radiance_elf_sha256"]:
        raise ValueError("device ELF differs from its build manifest")
    image, image_paths = verify_image(Path(manifest["weight_image_manifest"]))
    if image["image_sha256"] != manifest["weight_image_sha256"]:
        raise ValueError("weight image differs from its build manifest")
    verify_image_placement(elf, [image])
    native, checks = generate(target, manifest, image, image_paths)
    binary = build(target, native, jobs)
    log = native / "native.log"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    with log.open("w") as output:
        process = subprocess.run([str(binary)], stdout=output,
                                 stderr=subprocess.STDOUT, env=env)
    log_text = log.read_text()
    summary = re.search(r"stages=(\d+) replicas=2 checks=(\d+) errors=(\d+) "
                        r"max_abs_error=([^\n]+)", log_text)
    tokens = {
        int(replica): [int(value) for _, value in sorted(
            ((int(step), value) for r, step, value in
             re.findall(r"replica=(\d+) token_(\d+)=(-?\d+)", log_text)
             if int(r) == int(replica)))]
        for replica in (0, 1)
    }
    expected = manifest.get("reference_generated_token_ids") or []
    per_replica_errors = {
        int(replica): int(errors) for replica, errors in
        re.findall(r"replica=(\d+) errors=(\d+)", log_text)
    }
    passed = (process.returncode == 0 and summary is not None and
              int(summary.group(1)) == manifest["stages"] and
              int(summary.group(2)) == checks and int(summary.group(3)) == 0 and
              tokens == {0: expected, 1: expected} and
              per_replica_errors == {0: 0, 1: 0})
    result = {
        "model": manifest["model"],
        "scope": "full_checkpoint_generated_cpp_two_independent_replicas_on_cpu",
        "status": "passed" if passed else "failed",
        "stages_per_replica": manifest["stages"], "replicas": 2,
        "hbm_launch_clusters": manifest["hbm_launch_clusters"],
        "checked_tensors_per_replica": checks,
        "generated_token_ids": tokens,
        "expected_token_ids": expected,
        "max_abs_error": float(summary.group(4)) if summary else None,
        "errors": int(summary.group(3)) if summary else None,
        "errors_per_replica": per_replica_errors,
        "comparison_tolerance": {"rtol": 0.005, "atol": 0.0005},
        "process_exit_code": process.returncode,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "weight_image_sha256": image["image_sha256"],
        "reference_output_sha256": manifest["reference_output_sha256"],
        "device_elf_sha256": sha256(elf),
        "native_binary_sha256": sha256(binary),
        "native_source_sha256": sha256(native / "main.cpp"),
        "native_log_sha256": sha256(log),
        "native_log_tail": log_text[-1200:],
        "host_threads": threads,
        "device_execution": False, "rtl_execution": False,
        "performance_measurement": False,
    }
    (native / "native-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    result = run(args.target, threads=args.threads, jobs=args.jobs)
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
