#!/usr/bin/env python3
"""Check the exact generated SpectFormer stage code on the host CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from prepare import sha256


STUBS = {
    "mu_intrinsics.h": """#pragma once
#define __global
#define MU_NUM_THREADS 32
#define MU_NUM_CORES 1
inline void mu_barrier(unsigned, unsigned) {}
inline void mu_fence() {}
""",
    "mu_schedule.h": """#pragma once
#include <stdint.h>
typedef void (*mu_schedule_callback)(void*, uint32_t, uint32_t, uint32_t);
inline void mu_schedule(mu_schedule_callback callback, void* arg, uint32_t) {
  for (uint32_t tid = 0; tid < 32; ++tid) callback(arg, tid, 32, 0);
}
""",
    "kernel_verify.h": """#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
inline bool mu_close(float x, float golden, float rtol, float atol) {
  float delta = x - golden; if (delta < 0) delta = -delta;
  float magnitude = golden < 0 ? -golden : golden;
  extern float sf_max_abs_error;
  if (delta > sf_max_abs_error) sf_max_abs_error = delta;
  return delta <= rtol * magnitude + atol;
}
inline float sf_max_abs_error = 0.0f;
inline uint32_t mu_hart_id() { return 0; }
inline void mu_tohost(uint32_t code) {
  fprintf(stderr, "tohost=%u max_abs_error=%g\\n", code, sf_max_abs_error);
  if (code) exit(1);
}
""",
    "map_image.cpp": """#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
__attribute__((constructor)) static void map_image() {
  const char* path = getenv("SPECTFORMER_IMAGE");
  if (!path) { fputs("SPECTFORMER_IMAGE is missing\\n", stderr); exit(2); }
  int file = open(path, O_RDONLY); if (file < 0) { perror(path); exit(2); }
  struct stat info; if (fstat(file, &info)) { perror("fstat"); exit(2); }
  void* address = mmap((void*)0x30000000ull, info.st_size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_FIXED_NOREPLACE, file, 0);
  close(file);
  if (address != (void*)0x30000000ull) { perror("mmap image"); exit(2); }
  const char* runtime = getenv("SPECTFORMER_RUNTIME_INPUT");
  if (runtime) {
    const char* offset_string = getenv("SPECTFORMER_INPUT_OFFSET");
    const char* size_string = getenv("SPECTFORMER_INPUT_SIZE");
    if (!offset_string || !size_string) exit(2);
    const size_t offset = strtoull(offset_string, nullptr, 10);
    const size_t size = strtoull(size_string, nullptr, 10);
    if (offset + size > (size_t)info.st_size) exit(2);
    FILE* input = fopen(runtime, "rb");
    if (!input) { perror(runtime); exit(2); }
    if (fread((char*)address + offset, 1, size, input) != size ||
        fgetc(input) != EOF) { fputs("runtime input size mismatch\\n", stderr); exit(2); }
    fclose(input);
  }
}
""",
}


def run(target: Path, timeout: int, input_path: Path | None = None,
        trace_blocks: bool = False) -> dict:
    target = target.resolve()
    manifest = json.loads((target / "manifest.json").read_text())
    image_manifest = Path(manifest["source_image_manifest"])
    image = json.loads(image_manifest.read_text())
    output_kind = manifest.get("output_kind", "logits")
    output_count = manifest.get("output_count", image["num_classes"])
    output_offset = manifest.get("output_offset", manifest["logits_offset"])
    data = image_manifest.parent / "image.bin"
    if sha256(data) != manifest["image_sha256"]:
        raise ValueError("image differs from generated sources")
    native = target / "native"
    native.mkdir(exist_ok=True)
    for name, content in STUBS.items():
        (native / name).write_text(content)
    (native / "dump_arena.cpp").write_text(f'''#include <stdio.h>
#include <stdlib.h>
extern unsigned char v_arena[];
__attribute__((destructor)) static void dump_arena() {{
  const char* path = getenv("SPECTFORMER_DUMP_ARENA");
  if (!path) return;
  FILE* stream = fopen(path, "wb");
  if (!stream) {{ perror(path); return; }}
  fwrite(v_arena, 1, {manifest["arena_bytes"]}, stream);
  fclose(stream);
}}
''')
    if trace_blocks:
        (native / "trace.cpp").write_text('''#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void sf_trace(uint32_t index, const float* values, uint32_t count) {
  const char* directory = getenv("SPECTFORMER_TRACE_DIR");
  if (!directory) exit(2);
  char path[4096];
  snprintf(path, sizeof(path), "%s/trace-%03u.bin", directory, index);
  FILE* stream = fopen(path, "wb");
  if (!stream) { perror(path); exit(2); }
  if (fwrite(values, sizeof(float), count, stream) != count) exit(2);
  fclose(stream);
}
''')
    binary = native / "spectformer_native"
    command = ["g++", "-O2", "-std=c++20", "-I", str(native),
               "-I", str(Path(__file__).resolve().parent),
               "-I", str(Path(__file__).resolve().parents[1] / "model_chain"),
               str(target / "kernel.cpp"), str(target / "model_data.cpp"),
               str(native / "map_image.cpp"), str(native / "dump_arena.cpp"),
               "-o", str(binary)]
    if trace_blocks:
        command.insert(1, "-DSPECTFORMER_STAGE_TRACE")
        command.insert(-2, str(native / "trace.cpp"))
    subprocess.run(command, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT, text=True)
    environment = os.environ.copy()
    environment["SPECTFORMER_IMAGE"] = str(data.resolve())
    runtime_golden = None
    if input_path is not None:
        from reference import load_input
        from run_reusable import upstream_output
        pixels = load_input(input_path, image["image_size"])
        runtime_golden = upstream_output(image, pixels, output_kind)
        input_bytes = native / "input.bin"
        input_bytes.write_bytes(pixels.tobytes())
        entry = image["entries"]["input.image"]
        environment["SPECTFORMER_RUNTIME_INPUT"] = str(input_bytes)
        environment["SPECTFORMER_INPUT_OFFSET"] = str(entry["offset_bytes"])
        environment["SPECTFORMER_INPUT_SIZE"] = str(entry["size_bytes"])
    elif output_kind != "logits":
        from run_reusable import upstream_output
        pixels = np.fromfile(image_manifest.parent / "input.bin", dtype="<f4")
        pixels = pixels.reshape(1, 3, image["image_size"], image["image_size"])
        runtime_golden = upstream_output(image, pixels, output_kind)
    run_suffix = f"-{input_path.stem}" if input_path else ""
    arena_dump = native / f"arena{run_suffix}.bin"
    arena_dump.unlink(missing_ok=True)
    environment["SPECTFORMER_DUMP_ARENA"] = str(arena_dump)
    trace_dir = native / f"traces{run_suffix}"
    if trace_blocks:
        trace_dir.mkdir(exist_ok=True)
        environment["SPECTFORMER_TRACE_DIR"] = str(trace_dir)
    try:
        result = subprocess.run([str(binary)], env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=timeout)
        output, exit_code = result.stdout, result.returncode
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        exit_code = 124
    run_log = native / f"run{run_suffix}.log"
    run_log.write_text(output)
    comparison = None
    if arena_dump.is_file() and arena_dump.stat().st_size == manifest["arena_bytes"]:
        count = output_count
        with arena_dump.open("rb") as stream:
            stream.seek(output_offset)
            actual = np.frombuffer(stream.read(count * 4), dtype="<f4").copy()
        golden = runtime_golden
        if golden is None:
            golden = np.fromfile(image_manifest.parent / "golden.bin", dtype="<f4") if (
                image_manifest.parent / "golden.bin").exists() else None
        if golden is None:
            entry = image["entries"]["output.logits"]
            with data.open("rb") as stream:
                stream.seek(entry["offset_bytes"])
                golden = np.frombuffer(stream.read(entry["size_bytes"]), dtype="<f4").copy()
        finite = bool(np.isfinite(actual).all())
        difference = actual.astype(np.float64) - golden.astype(np.float64)
        comparison = {"all_finite": finite,
                      "all_close": finite and bool(np.allclose(
                          actual, golden, rtol=manifest["logit_rtol"],
                          atol=manifest["logit_atol"])),
                      "max_abs_error": float(np.max(np.abs(difference))) if finite else None,
                      "top1_agrees": (bool(np.argmax(actual) == np.argmax(golden))
                                      if finite and output_kind == "logits" else None),
                      "logits_sha256": hashlib.sha256(actual.tobytes()).hexdigest()}
    passed = (exit_code == 0 and "tohost=0" in output and
              comparison is not None and comparison["all_close"])
    block_comparison = None
    if trace_blocks and passed:
        from trace_blocks import compare
        if input_path is None:
            pixels = np.fromfile(image_manifest.parent / "input.bin", dtype="<f4")
            pixels = pixels.reshape(1, 3, image["image_size"], image["image_size"])
        block_comparison = compare(image, manifest, pixels, trace_dir)
        passed = block_comparison["status"] == "passed"
    match = re.search(r"max_abs_error=([0-9.eE+-]+)", output)
    record = {"model": manifest["model"], "status": "passed" if passed else "failed",
              "output_kind": output_kind, "output_count": output_count,
              "host_generated_source_execution": True,
              "device_execution": False, "performance_measurement": False,
              "weight_provenance": image["weight_provenance"],
              "input_provenance": image.get("input_provenance", "torch.linspace(-1,1)"),
              "stage_count": manifest["stages"], "image_sha256": sha256(data),
              "input_path": str(input_path.resolve()) if input_path else None,
              "radiance_elf_sha256": manifest.get("radiance_elf_sha256"),
              "output_comparison": comparison,
              "logit_comparison": comparison if output_kind == "logits" else None,
              "block_comparison": block_comparison,
              "max_abs_logit_error": (float(match.group(1))
                                      if match and output_kind == "logits" else None),
              "native_binary_sha256": sha256(binary),
              "log_sha256": sha256(run_log), "exit_code": exit_code,
              "output": output[-1000:]}
    result_name = (f"result-{input_path.stem}.json" if input_path else "result.json")
    (native / result_name).write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--input", type=Path,
                        help="new normalized NPY or RGB image without rebuilding the ELF")
    parser.add_argument("--trace-blocks", action="store_true",
                        help="compare every connected block output with upstream PyTorch")
    args = parser.parse_args()
    result = run(args.target, args.timeout, args.input, args.trace_blocks)
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
