#!/usr/bin/env python3
"""Execute generated SmolVLA stage C++ on the CPU for a fast math check.

This uses the same generated stage functions and checkpoint image as the
Radiance ELF. It is a software arithmetic check, not device or RTL execution.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from compile_decoder import symbol, write_if_changed
from split_decoder_weights import image_segments, verify_image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "kernels/evaluation/llm"))
from stitch import build  # noqa: E402
from plan_buffers import plan  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(target: Path) -> Path:
    manifest = json.loads((target / "manifest.json").read_text())
    graph = build("smolvla_base")
    storage = plan(graph)
    if (manifest["stages"] != len(graph.stages) or
            manifest["execution_schedule_sha256"] !=
            storage["execution_schedule_sha256"] or
            manifest["output_validation"] != "all_elements_vs_upstream_policy"):
        raise ValueError("native run requires the full exact-golden ELF sources")
    images = []
    for key, digest in (("weight_image_manifest", "weight_image_sha256"),
                        ("input_image_manifest", "input_image_sha256")):
        image, paths = verify_image(Path(manifest[key]))
        if image["image_sha256"] != manifest[digest]:
            raise ValueError(f"{key} differs from ELF build")
        images.append((image, paths))
    native = target / "native"
    native.mkdir(exist_ok=True)
    stub = "#pragma once\n#define __global\n#define MU_NUM_THREADS 32\n#define MU_NUM_CORES 1\n"
    write_if_changed(native / "mu_intrinsics.h", stub)
    write_if_changed(native / "mu_schedule.h", "#pragma once\n")
    preload = []
    for image, paths in images:
        for segment, path in zip(image_segments(image), paths):
            preload.append(
                f"  if (!map_image({segment['gpu_base_address']}ull, "
                f"{segment['image_size_bytes']}ull, "
                f"{json.dumps(str(path.resolve()))})) return 2;")
    copies = []
    for item in images[1][0]["parameters"]:
        offset = storage["allocation"][item["logical_name"]]["offset_bytes"]
        copies.append(
            f"  memcpy(v_arena + {offset}u, "
            f"(const void*){item['gpu_address']}ull, {item['size_bytes']}u);")
    names = ["stage_" + symbol(stage["id"]) for stage in graph.stages]
    prototypes = "\n".join(
        f"void {name}(void*, uint32_t, uint32_t, uint32_t);" for name in names)

    def trace_stage(stage: dict) -> bool:
        name = stage["id"]
        return (name.endswith(("vision.final_norm", "connector", "state_proj",
                               "vlm.final_norm", "action_out")) or
                name in ("prefix.merge", "prefix.attention_mask",
                         "prefix.position_ids") or
                (name.startswith("vlm.layer") and name.endswith(
                    ("input_norm", "o_proj", "ffn_residual"))))

    entries = ",\n".join(
        f"  {{{json.dumps(stage['id'])}, {name}, "
        f"{storage['allocation'][stage['writes']]['offset_bytes']}u, "
        f"{4 * math.prod(stage['shape'])}u, "
        f"{'true' if trace_stage(stage) else 'false'}}}"
        for name, stage in zip(names, graph.stages))
    source = """#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
extern unsigned char v_arena[];
extern float v_golden_output[];
""" + prototypes + """
struct Stage { const char* name; void (*run)(void*, uint32_t, uint32_t, uint32_t);
               uint32_t offset; uint32_t bytes; bool trace; };
static const Stage stages[] = {
""" + entries + "\n};\n" + """
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
int main(int argc, char** argv) {
  const uint32_t total = sizeof(stages) / sizeof(stages[0]);
  const char* trace_dir = getenv("SMOLVLA_NATIVE_TRACE_DIR");
  uint32_t limit = argc > 1 ? (uint32_t)strtoul(argv[1], nullptr, 10) : total;
  if (limit == 0 || limit > total) return 2;
""" + "\n".join(preload) + "\n" + "\n".join(copies) + """
  for (uint32_t i = 0; i < limit; ++i) {
    #pragma omp parallel for schedule(static)
    for (uint32_t tid = 0; tid < 32; ++tid)
      stages[i].run(nullptr, tid, 32, 0);
    if (trace_dir && stages[i].trace) {
      char filename[4096];
      snprintf(filename, sizeof(filename), "%s/%04u.bin", trace_dir, i + 1);
      FILE* trace = fopen(filename, "wb");
      if (!trace || fwrite(v_arena + stages[i].offset, 1, stages[i].bytes,
                           trace) != stages[i].bytes) return 3;
      fclose(trace);
    }
    if ((i + 1) % 100 == 0 || i + 1 == limit) {
      fprintf(stderr, "completed %u/%u: %s\\n", i + 1, total, stages[i].name);
      fflush(stderr);
    }
  }
  const Stage& final = stages[limit - 1];
  if (argc > 2) {
    FILE* dump = fopen(argv[2], "wb");
    if (!dump || fwrite(v_arena + final.offset, 1, final.bytes, dump) != final.bytes)
      return 3;
    fclose(dump);
  }
  if (limit != total) return 0;
  const float* output = (const float*)(v_arena + final.offset);
  uint32_t failures = 0, first_failure = 0;
  float max_error = 0.0f;
  for (uint32_t i = 0; i < final.bytes / sizeof(float); ++i) {
    float error = fabsf(output[i] - v_golden_output[i]);
    if (!std::isfinite(output[i]) || error > 0.001f + 0.001f * fabsf(v_golden_output[i])) {
      if (!failures) first_failure = i;
      ++failures;
    }
    if (error > max_error) max_error = error;
  }
  printf("actions=%u failures=%u first_failure=%u max_abs_error=%.9g\\n",
         final.bytes / 4, failures, first_failure, max_error);
  return failures ? 1 : 0;
}
"""
    write_if_changed(native / "main.cpp", source)
    return native


def compile_native(target: Path, jobs: int = 2) -> Path:
    native = generate(target)
    compiler = ["g++", "-std=c++17", "-O2", "-fopenmp", "-I", str(native),
                "-I", str(HERE)]
    sources = [target / "model_data.cpp", *sorted(target.glob("stage_chunk_*.cpp")),
               native / "main.cpp"]

    def compile_one(source: Path) -> Path:
        obj = native / (source.stem + ".o")
        dependencies = [source, native / "mu_intrinsics.h",
                        native / "mu_schedule.h", HERE / "pipeline_smolvla.hpp",
                        HERE / "pipeline_math.hpp"]
        if (not obj.exists() or obj.stat().st_mtime <
                max(path.stat().st_mtime for path in dependencies)):
            result = subprocess.run(compiler + ["-c", str(source), "-o", str(obj)],
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"{source}: {result.stdout}{result.stderr}")
        return obj

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        objects = list(executor.map(compile_one, sources))
    binary = native / "smolvla_native"
    subprocess.run(["g++", "-fopenmp", *map(str, objects), "-o", str(binary)],
                   check=True)
    return binary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--stage-limit", type=int)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    if args.stage_limit is not None and not 1 <= args.stage_limit <= 3673:
        parser.error("--stage-limit must be between 1 and 3673")
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    target = (args.generated_root / "smolvla_base").resolve()
    binary = compile_native(target, args.jobs)
    command = [str(binary)]
    if args.stage_limit is not None:
        command.append(str(args.stage_limit))
    if args.dump:
        if args.stage_limit is None:
            command.append("3673")
        command.append(str(args.dump.resolve()))
    if args.threads <= 0 or args.threads > 32:
        parser.error("--threads must be between 1 and 32")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        env["SMOLVLA_NATIVE_TRACE_DIR"] = str(args.trace_dir.resolve())
    log_path = target / ("native-run-full.log" if args.stage_limit is None
                         else f"native-run-stage{args.stage_limit}.log")
    with log_path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=env)
    build_manifest = json.loads((target / "manifest.json").read_text())
    log_text = log_path.read_text()
    match = re.search(
        r"actions=(\d+) failures=(\d+) first_failure=(\d+) max_abs_error=([\deE+.-]+)",
        log_text)
    if args.stage_limit is None and not match:
        raise ValueError("native full-action run did not report its output comparison")
    record = {"model": "smolvla_base", "scope": "generated_stage_cpp_on_cpu",
              "stage_count": args.stage_limit or 3673,
              "host_threads": args.threads,
              "status": "passed" if result.returncode == 0 else "failed",
              "process_exit_code": result.returncode,
              "upstream_golden_compared": args.stage_limit is None,
              "golden_output_sha256": build_manifest["golden_output_sha256"],
              "golden_policy_precision": build_manifest["golden_policy_precision"],
              "comparison_tolerance": (build_manifest["output_check_tolerance"]
                                       if args.stage_limit is None else None),
              "output_elements": int(match.group(1)) if match else None,
              "failed_output_elements": int(match.group(2)) if match else None,
              "first_failed_element": int(match.group(3)) if match else None,
              "max_abs_error": float(match.group(4)) if match else None,
              "native_binary_sha256": sha256(binary),
              "generated_source_sha256": hashlib.sha256(
                  "".join(sha256(target / name) for name in build_manifest[
                      "device_source_files"]).encode()).hexdigest(),
              "device_elf_sha256": build_manifest["radiance_elf_sha256"],
              "log_sha256": sha256(log_path),
              "log_path": str(log_path),
              "log_tail": log_text[-1200:],
              "rtl_execution": False, "device_execution": False,
              "performance_measurement": False}
    if args.dump and args.dump.is_file():
        record["dump_sha256"] = sha256(args.dump)
    destination = args.out or target / "native-result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(f"SmolVLA native: {record['status']} ({record['stage_count']} stages)")
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
