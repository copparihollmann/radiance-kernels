#!/usr/bin/env python3
"""Check the exact generated SpectFormer stage code on the host CPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

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
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
__attribute__((constructor)) static void map_image() {
  const char* path = getenv("SPECTFORMER_IMAGE");
  if (!path) { fputs("SPECTFORMER_IMAGE is missing\\n", stderr); exit(2); }
  int file = open(path, O_RDONLY); if (file < 0) { perror(path); exit(2); }
  struct stat info; if (fstat(file, &info)) { perror("fstat"); exit(2); }
  void* address = mmap((void*)0x30000000ull, info.st_size, PROT_READ,
                       MAP_PRIVATE | MAP_FIXED_NOREPLACE, file, 0);
  close(file);
  if (address != (void*)0x30000000ull) { perror("mmap image"); exit(2); }
}
""",
}


def run(target: Path, timeout: int) -> dict:
    target = target.resolve()
    manifest = json.loads((target / "manifest.json").read_text())
    image_manifest = Path(manifest["source_image_manifest"])
    image = json.loads(image_manifest.read_text())
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
    binary = native / "spectformer_native"
    command = ["g++", "-O2", "-std=c++20", "-I", str(native),
               "-I", str(Path(__file__).resolve().parent),
               "-I", str(Path(__file__).resolve().parents[1] / "model_chain"),
               str(target / "kernel.cpp"), str(target / "model_data.cpp"),
               str(native / "map_image.cpp"), str(native / "dump_arena.cpp"),
               "-o", str(binary)]
    subprocess.run(command, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT, text=True)
    environment = os.environ.copy()
    environment["SPECTFORMER_IMAGE"] = str(data.resolve())
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
    (native / "run.log").write_text(output)
    passed = exit_code == 0 and "tohost=0" in output
    match = re.search(r"max_abs_error=([0-9.eE+-]+)", output)
    record = {"model": "spectformer_vanilla", "status": "passed" if passed else "failed",
              "host_generated_source_execution": True,
              "device_execution": False, "performance_measurement": False,
              "weight_provenance": image["weight_provenance"],
              "input_provenance": image.get("input_provenance", "torch.linspace(-1,1)"),
              "stage_count": manifest["stages"], "image_sha256": sha256(data),
              "max_abs_logit_error": float(match.group(1)) if match else None,
              "native_binary_sha256": sha256(binary),
              "log_sha256": sha256(native / "run.log"), "exit_code": exit_code,
              "output": output[-1000:]}
    (native / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    result = run(args.target, args.timeout)
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
