#!/usr/bin/env python3
"""Run the pinned Vortex Hotspot3D body on Radiance's measured fixture."""

from __future__ import annotations

from array import array
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import struct
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parent
HOTSPOT_ROOT = ROOT.parent
RUN_PATH = HOTSPOT_ROOT / "run.py"
PIN_PATH = ROOT / "vortex-hotspot3d-kernel.cl"
OPENCL_RUNNER = ROOT / "opencl_runner.c"

RADIANCE_BASE_REVISION = "8bf256a5406e749a6342522e398b468d56c1b5db"
RADIANCE_RUN_BLOB = "be3237cece302a5ab221502131f764a6914aff4f"
RADIANCE_RUN_SHA256 = (
    "f12b97edfa268e3f44e5d8ecd32749a8e77d9ddc40b09fa39d8905df4bd5aa5a"
)
VORTEX_REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"
VORTEX_OBJECTS = {
    "tests/opencl/hotspot3D/main.cc": {
        "blob": "af729c56b76263cb3869b80787173363f4ff69ae",
        "sha256": "3954ddc7b8a841605c09d21fa8ef1f4cc07b1216fc9c3748a425d7b88907feec",
    },
    "tests/opencl/hotspot3D/kernel.cl": {
        "blob": "ec62ac18e872fdb8cd514f146626b6e4ed590ea4",
        "sha256": "f3823a9850bab4460cbaa3763f55c8d2ad2f42d44e13bc11399a43a0e2ef0e47",
    },
    "tests/opencl/hotspot3D/Makefile": {
        "blob": "8cabdd94d49fa92918e25b8ae90d9d952096be80",
        "sha256": "38c9f1d989f66e6e23c55aaf5760f4a8d9270bdf310874de6da9f1690c2acf64",
    },
}
VORTEX_KERNEL_PATH = "tests/opencl/hotspot3D/kernel.cl"
VORTEX_KERNEL_BYTES = 1686

NX = 64
NY = 64
NZ = 8
ITERATIONS = 4
ELEMENTS = NX * NY * NZ
UPDATES = ELEMENTS * ITERATIONS
MEMORY_LIMIT_BYTES = 4 * 1024**3
OPENCL_UNAVAILABLE = 77
COEFFICIENT_WORDS = {
    "step_div_cap": "3baec33d",
    "ce": "3a0bcf64",
    "cw": "3a0bcf64",
    "cn": "3a0bcf64",
    "cs": "3a0bcf64",
    "ct": "3a0bcf64",
    "cb": "3a0bcf64",
    "cc": "3f7f0b55",
}
POWER_SHA256 = "8310cd3288477888e846b3a01354a006aa1e46c60aefb338ecb2f752319105c9"
INITIAL_SHA256 = "9c208185aafd083c0363c8d4317ad3b711556a8a90b3ab917903d682fee9adf2"
STATE_SHA256 = [
    INITIAL_SHA256,
    "3feba706e311cb26c60d412cc14fb5469cfdbda4ce8af84c78c2be318926638a",
    "ba07a87dbe8cb4de2d93f4150f30ae0a1f6217ba29785c541a0c6deb6add6037",
    "e657e48342dfb471577d5d252fe7b4414dc7368012a3d9ce41040e922f72342b",
    "fcf9241fc25c286e8338eb224886fe865bd2c9cfae0e277c8e55120eaaabf7da",
]
STATE_FNV64 = [
    "9e8d96a7f788705d",
    "28810e5b01482a24",
    "b131ee223001dbd4",
    "a38c0ad2140951da",
    "28033df6e64f5859",
]


def enforce_memory_limit() -> dict:
    """Put this process and all descendants under a hard <=4 GiB AS limit."""
    _, old_hard = resource.getrlimit(resource.RLIMIT_AS)
    target = MEMORY_LIMIT_BYTES
    if old_hard != resource.RLIM_INFINITY:
        target = min(target, old_hard)
    resource.setrlimit(resource.RLIMIT_AS, (target, target))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if soft > MEMORY_LIMIT_BYTES or hard > MEMORY_LIMIT_BYTES:
        raise RuntimeError("failed to enforce the 4 GiB address-space ceiling")
    return {
        "resource": "RLIMIT_AS",
        "soft_bytes": soft,
        "hard_bytes": hard,
        "requested_ceiling_bytes": MEMORY_LIMIT_BYTES,
        "inherited_by_all_child_processes": True,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def pinned_kernel() -> bytes:
    data = PIN_PATH.read_bytes()
    expected = VORTEX_OBJECTS[VORTEX_KERNEL_PATH]
    if len(data) != VORTEX_KERNEL_BYTES:
        raise RuntimeError(
            f"pinned kernel has {len(data)} bytes, expected {VORTEX_KERNEL_BYTES}"
        )
    if sha256_bytes(data) != expected["sha256"]:
        raise RuntimeError("pinned Vortex kernel SHA-256 mismatch")
    if git_blob_sha1(data) != expected["blob"]:
        raise RuntimeError("pinned Vortex kernel Git blob mismatch")
    return data


def run_command(
    command: list[str], *, cwd: Path | None = None, allow_failure: bool = False
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0 and not allow_failure:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def command_output(command: list[str]) -> str:
    return run_command(command).stdout.strip()


def verify_vortex_checkout(path: Path | None, kernel: bytes) -> dict:
    if path is None:
        return {
            "status": "not-available",
            "note": "tracked kernel still verifies by both Git-blob and SHA-256 identity",
        }
    path = path.resolve()
    if command_output(
        ["git", "-C", str(path), "cat-file", "-t", VORTEX_REVISION]
    ) != "commit":
        raise RuntimeError(f"Vortex pin is not a commit in {path}")
    verified = {}
    for source_path, expected in VORTEX_OBJECTS.items():
        blob = command_output(
            ["git", "-C", str(path), "rev-parse", f"{VORTEX_REVISION}:{source_path}"]
        )
        source = subprocess.run(
            ["git", "-C", str(path), "show", f"{VORTEX_REVISION}:{source_path}"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
        digest = sha256_bytes(source)
        if blob != expected["blob"] or digest != expected["sha256"]:
            raise RuntimeError(f"Vortex object identity mismatch: {source_path}")
        verified[source_path] = {
            "git_blob": blob, "sha256": digest, "bytes": len(source)
        }
        if source_path == VORTEX_KERNEL_PATH and source != kernel:
            raise RuntimeError("tracked kernel is not byte-exact to Vortex revision")
    return {
        "status": "passed",
        "checkout": str(path),
        "revision": VORTEX_REVISION,
        "objects": verified,
        "kernel_byte_exact": True,
    }


def load_radiance_run():
    source = RUN_PATH.read_bytes()
    if sha256_bytes(source) != RADIANCE_RUN_SHA256:
        raise RuntimeError("Radiance measured-fixture run.py SHA-256 mismatch")
    if git_blob_sha1(source) != RADIANCE_RUN_BLOB:
        raise RuntimeError("Radiance measured-fixture run.py Git blob mismatch")
    spec = importlib.util.spec_from_file_location("hotspot3d_fixture_run", RUN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compiler_identity(executable: str) -> dict:
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(f"required compiler not found: {executable}")
    version = command_output([resolved, "--version"]).splitlines()[0]
    return {"path": resolved, "version": version}


def float_to_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def word_to_float(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def ordered_word(word: int) -> int:
    return (~word & 0xFFFFFFFF) if word & 0x80000000 else word | 0x80000000


def write_f32(path: Path, values: Iterable[float]) -> None:
    data = array("f", values)
    if sys.byteorder != "little":
        data.byteswap()
    path.write_bytes(data.tobytes())


def read_words(path: Path, records: int = ELEMENTS) -> array:
    data = array("I")
    with path.open("rb") as stream:
        data.fromfile(stream, records)
        if stream.read(1):
            raise RuntimeError(f"{path} contains more than {records} words")
    if len(data) != records:
        raise RuntimeError(f"{path} contains {len(data)} words, expected {records}")
    if sys.byteorder != "little":
        data.byteswap()
    return data


def fnv_digest(words: Iterable[int]) -> str:
    value = 0xCBF29CE484222325
    for word in words:
        value = ((value ^ int(word)) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def stage_ulp_limit(step: int) -> int:
    if step < 1 or step > ITERATIONS:
        raise ValueError("step is outside measured fixture")
    return max(32, 16 * step)


def compare_words(actual: array, expected: array, step: int, *, exact: bool) -> dict:
    if len(actual) != len(expected):
        raise RuntimeError("output lengths differ")
    limit = 0 if exact else stage_ulp_limit(step)
    exact_count = 0
    nonfinite = 0
    over_limit = 0
    maximum = 0
    first_bad: list[dict] = []
    for index, (actual_word, expected_word) in enumerate(zip(actual, expected)):
        if not math.isfinite(word_to_float(actual_word)):
            nonfinite += 1
            distance = 0xFFFFFFFF
        else:
            distance = abs(ordered_word(actual_word) - ordered_word(expected_word))
        exact_count += actual_word == expected_word
        maximum = max(maximum, distance)
        if distance > limit:
            over_limit += 1
            if len(first_bad) < 8:
                first_bad.append({
                    "index": index,
                    "actual_word": f"{actual_word:08x}",
                    "expected_word": f"{expected_word:08x}",
                    "ulp_distance": distance,
                })
    return {
        "step": step,
        "records_compared": len(actual),
        "expected_sha256": STATE_SHA256[step],
        "exact_word_matches": exact_count,
        "nonfinite": nonfinite,
        "maximum_ulp_distance": maximum,
        "maximum_ulp_contract": limit,
        "over_limit": over_limit,
        "first_over_limit": first_bad,
        "passed": nonfinite == 0 and over_limit == 0,
    }


def native_source(kernel: bytes) -> bytes:
    prefix = b"""#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

static size_t hotspot_cpu_global_id[2];
#define __kernel __attribute__((noinline))
#define __global
#define get_global_id(dimension) (hotspot_cpu_global_id[(dimension)])
#line 1 "vortex/tests/opencl/hotspot3D/kernel.cl"
"""
    suffix = b"""
#undef get_global_id
#undef __global
#undef __kernel

static const rlim_t memory_limit_bytes = (rlim_t)4 * 1024 * 1024 * 1024;

static void enforce_memory_limit(void) {
  const struct rlimit limit = {memory_limit_bytes, memory_limit_bytes};
  if (setrlimit(RLIMIT_AS, &limit) != 0) {
    fprintf(stderr, "setrlimit(RLIMIT_AS): %s\\n", strerror(errno));
    exit(2);
  }
}

static unsigned long parse_positive(const char *text) {
  char *end = NULL;
  unsigned long value = strtoul(text, &end, 10);
  if (end == text || *end != '\\0' || value == 0 || value > INT32_MAX)
    exit(2);
  return value;
}

static float float_from_word(const char *text) {
  char *end = NULL;
  unsigned long raw = strtoul(text, &end, 16);
  if (end == text || *end != '\\0' || raw > UINT32_MAX) exit(2);
  uint32_t word = (uint32_t)raw;
  float value;
  memcpy(&value, &word, sizeof(value));
  return value;
}

static float *read_exact(const char *path, size_t bytes) {
  FILE *stream = fopen(path, "rb");
  float *data = malloc(bytes);
  if (stream == NULL || data == NULL ||
      fread(data, 1, bytes, stream) != bytes || fgetc(stream) != EOF) {
    fprintf(stderr, "cannot read exactly %zu bytes from %s\\n", bytes, path);
    exit(2);
  }
  fclose(stream);
  return data;
}

static void write_step(const char *prefix, unsigned step, const float *data,
                       size_t bytes) {
  char path[4096];
  int length = snprintf(path, sizeof(path), "%s-step-%u.bin", prefix, step);
  if (length < 0 || (size_t)length >= sizeof(path)) exit(2);
  FILE *stream = fopen(path, "wb");
  if (stream == NULL || fwrite(data, 1, bytes, stream) != bytes ||
      fclose(stream) != 0) {
    fprintf(stderr, "cannot write %s\\n", path);
    exit(2);
  }
}

int main(int argc, char **argv) {
  enforce_memory_limit();
  if (argc != 16) {
    fprintf(stderr, "usage: %s POWER INITIAL OUTPUT_PREFIX NX NY NZ ITERATIONS "
                    "SDC CE CW CN CS CT CB CC\\n", argv[0]);
    return 2;
  }
  const int nx = (int)parse_positive(argv[4]);
  const int ny = (int)parse_positive(argv[5]);
  const int nz = (int)parse_positive(argv[6]);
  const int iterations = (int)parse_positive(argv[7]);
  if (nz < 2) return 2;
  const size_t elements = (size_t)nx * (size_t)ny * (size_t)nz;
  if (elements > SIZE_MAX / sizeof(float)) return 2;
  const size_t bytes = elements * sizeof(float);
  float *power = read_exact(argv[1], bytes);
  float *buffers[2] = {read_exact(argv[2], bytes), malloc(bytes)};
  if (buffers[1] == NULL) return 2;
  const float sdc = float_from_word(argv[8]);
  const float ce = float_from_word(argv[9]);
  const float cw = float_from_word(argv[10]);
  const float cn = float_from_word(argv[11]);
  const float cs = float_from_word(argv[12]);
  const float ct = float_from_word(argv[13]);
  const float cb = float_from_word(argv[14]);
  const float cc = float_from_word(argv[15]);
  int input_index = 0;
  for (int step = 1; step <= iterations; ++step) {
    int output_index = 1 - input_index;
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        hotspot_cpu_global_id[0] = (size_t)i;
        hotspot_cpu_global_id[1] = (size_t)j;
        hotspotOpt1(power, buffers[input_index], buffers[output_index], sdc,
                    nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
      }
    }
    write_step(argv[3], (unsigned)step, buffers[output_index], bytes);
    input_index = output_index;
  }
  free(buffers[1]);
  free(buffers[0]);
  free(power);
  return 0;
}
"""
    return prefix + kernel + suffix


def compile_spir(kernel_path: Path, build: Path, clang: str) -> dict:
    bitcode = build / "vortex-hotspot3d.spir64.bc"
    command = [
        clang, "-x", "cl", "-cl-std=CL1.2", "-target",
        "spir64-unknown-unknown", "-emit-llvm", "-c", str(kernel_path),
        "-o", str(bitcode),
    ]
    run_command(command)
    return {
        "status": "passed",
        "target": "spir64-unknown-unknown",
        "language": "OpenCL C 1.2",
        "bitcode_sha256": sha256_file(bitcode),
        "bitcode_bytes": bitcode.stat().st_size,
        "command": [
            "clang", "-x", "cl", "-cl-std=CL1.2", "-target",
            "spir64-unknown-unknown", "-emit-llvm", "-c", "PINNED_KERNEL",
            "-o", "BITCODE",
        ],
    }


def compile_opencl_runner(build: Path, cc: str) -> Path:
    executable = build / "opencl_runner"
    run_command([
        cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
        str(OPENCL_RUNNER), "-lOpenCL", "-o", str(executable),
    ])
    return executable


def runner_arguments(
    power: Path, initial: Path, prefix: Path, coefficient_words: dict[str, str]
) -> list[str]:
    return [
        str(power), str(initial), str(prefix), str(NX), str(NY), str(NZ),
        str(ITERATIONS), coefficient_words["step_div_cap"],
        coefficient_words["ce"], coefficient_words["cw"],
        coefficient_words["cn"], coefficient_words["cs"],
        coefficient_words["ct"], coefficient_words["cb"],
        coefficient_words["cc"],
    ]


def try_opencl(
    executable: Path, kernel: Path, arguments: list[str]
) -> dict:
    completed = run_command(
        [str(executable), str(kernel), *arguments], allow_failure=True
    )
    result = {
        "attempted": True,
        "cpu_device_required": True,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode == 0:
        result["status"] = "executed"
    elif completed.returncode == OPENCL_UNAVAILABLE:
        result["status"] = "unavailable"
    else:
        raise RuntimeError(
            "an OpenCL CPU platform was present but execution failed:\n"
            + completed.stderr
        )
    return result


def compile_native_runner(
    build: Path, source: Path, cc: str, name: str, contract: str
) -> tuple[Path, list[str]]:
    executable = build / f"native-{name}"
    flags = [
        "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
        "-fno-fast-math", f"-ffp-contract={contract}",
        "-fexcess-precision=standard",
    ]
    if contract == "off":
        flags.append("-frounding-math")
    else:
        flags.append("-mfma")
    run_command([cc, *flags, str(source), "-o", str(executable)])
    return executable, flags


def compare_execution(
    name: str, method: str, prefix: Path, expected_paths: list[Path],
    *, exact: bool, flags: list[str] | None = None,
) -> dict:
    stages = []
    for step, expected_path in enumerate(expected_paths, 1):
        output = Path(f"{prefix}-step-{step}.bin")
        actual_words = read_words(output)
        expected_words = read_words(expected_path)
        comparison = compare_words(actual_words, expected_words, step, exact=exact)
        comparison["actual_sha256"] = sha256_file(output)
        comparison["actual_fnv64"] = fnv_digest(actual_words)
        comparison["expected_fnv64"] = STATE_FNV64[step]
        stages.append(comparison)
    result = {
        "name": name,
        "method": method,
        "requires_bit_exact": exact,
        "steps": stages,
        "records_compared_total": ELEMENTS * ITERATIONS,
        "passed": all(stage["passed"] for stage in stages),
    }
    if flags is not None:
        result["compiler_flags"] = flags
    return result


def generate_evidence(
    build: Path, vortex_root: Path | None, cc_name: str, clang_name: str
) -> dict:
    memory_limit = enforce_memory_limit()
    build.mkdir(parents=True, exist_ok=True)
    kernel = pinned_kernel()
    kernel_path = build / "kernel.cl"
    kernel_path.write_bytes(kernel)
    wrapper_path = build / "native-wrapper.c"
    wrapper = native_source(kernel)
    if wrapper.count(kernel) != 1:
        raise RuntimeError("mechanical wrapper does not embed the pin exactly once")
    wrapper_path.write_bytes(wrapper)

    radiance = load_radiance_run()
    dimensions = radiance.PROFILES.get("meaningful")
    if dimensions != {
        "nx": NX, "ny": NY, "nz": NZ, "iterations": ITERATIONS
    }:
        raise RuntimeError("Radiance meaningful fixture dimensions changed")
    coefficients = radiance.coefficients(NX, NY, NZ)
    coefficient_words = {
        name: f"{float_to_word(value):08x}" for name, value in coefficients.items()
    }
    if coefficient_words != COEFFICIENT_WORDS:
        raise RuntimeError("Radiance coefficient identity mismatch")
    power, initial = radiance.reference_inputs(ELEMENTS)
    states = radiance.reference_states(
        power, initial, NX, NY, NZ, ITERATIONS, coefficients
    )
    if len(states) != ITERATIONS + 1:
        raise RuntimeError("Radiance oracle did not produce every state")
    power_path = build / "power.bin"
    initial_path = build / "initial.bin"
    write_f32(power_path, power)
    write_f32(initial_path, initial)
    expected_paths = []
    for step, state in enumerate(states[1:], 1):
        path = build / f"radiance-expected-step-{step}.bin"
        write_f32(path, state)
        expected_paths.append(path)
    actual_state_hashes = [sha256_file(initial_path)] + [
        sha256_file(path) for path in expected_paths
    ]
    actual_state_fnv = [fnv_digest(read_words(initial_path))] + [
        fnv_digest(read_words(path)) for path in expected_paths
    ]
    if sha256_file(power_path) != POWER_SHA256:
        raise RuntimeError("exact measured power fixture identity mismatch")
    if actual_state_hashes != STATE_SHA256 or actual_state_fnv != STATE_FNV64:
        raise RuntimeError("exact measured temperature/oracle identity mismatch")

    cc = compiler_identity(cc_name)
    clang = compiler_identity(clang_name)
    spir = compile_spir(kernel_path, build, clang["path"])
    arguments = runner_arguments(
        power_path, initial_path, build / "opencl-output", coefficient_words
    )
    opencl_executable = compile_opencl_runner(build, cc["path"])
    opencl_attempt = try_opencl(opencl_executable, kernel_path, arguments)

    executions = []
    if opencl_attempt["status"] == "executed":
        executions.append(compare_execution(
            "opencl-cpu", "native-opencl-cpu-byte-exact-pinned-source",
            build / "opencl-output", expected_paths, exact=False,
        ))
    for name, contract, exact in (
        ("strict", "off", True), ("contracted", "fast", False)
    ):
        executable, flags = compile_native_runner(
            build, wrapper_path, cc["path"], name, contract
        )
        prefix = build / f"native-{name}-output"
        native_args = runner_arguments(
            power_path, initial_path, prefix, coefficient_words
        )
        run_command([str(executable), *native_args])
        executions.append(compare_execution(
            f"cpu-{name}",
            "mechanically-wrapped-byte-exact-pinned-kernel-body",
            prefix, expected_paths, exact=exact, flags=flags,
        ))

    checkout = verify_vortex_checkout(vortex_root, kernel)
    passed = all(execution["passed"] for execution in executions)
    limitations = []
    if opencl_attempt["status"] != "executed":
        limitations.extend([
            "No OpenCL CPU ICD/platform was available, so no OpenCL runtime "
            "compiled or executed the pinned kernel on this host.",
            "The CPU fallback uses preprocessor definitions only for OpenCL "
            "qualifiers and get_global_id, then executes the byte-exact source "
            "body; it does not prove behavior of a particular OpenCL device compiler.",
        ])
    limitations.append(
        "This is source-level CPU differential evidence only; it does not invoke "
        "Cyclotron, FireSim, an FPGA, or a bitstream."
    )
    evidence = {
        "schema": "radiance-hotspot3d-upstream-differential-v1",
        "status": "passed" if passed else "failed",
        "scope": "exact measured Hotspot3D 64x64x8 four-step fixture",
        "reference": {
            "repository": "vortexgpgpu/vortex",
            "revision": VORTEX_REVISION,
            "kernel_path": VORTEX_KERNEL_PATH,
            "kernel_bytes": len(kernel),
            "kernel_git_blob": git_blob_sha1(kernel),
            "kernel_sha256": sha256_bytes(kernel),
            "tracked_copy": str(PIN_PATH.relative_to(HOTSPOT_ROOT)),
            "checkout_verification": checkout,
        },
        "radiance_fixture": {
            "base_revision": RADIANCE_BASE_REVISION,
            "run_path": "kernels/hotspot3d/run.py",
            "run_git_blob": RADIANCE_RUN_BLOB,
            "run_sha256": RADIANCE_RUN_SHA256,
            "profile": "meaningful",
            "nx": NX, "ny": NY, "nz": NZ,
            "iterations": ITERATIONS,
            "elements_per_state": ELEMENTS,
            "full_stencil_updates": UPDATES,
            "seed": radiance.REFERENCE_SEED,
            "coefficient_words": coefficient_words,
            "power_sha256": sha256_file(power_path),
            "initial_sha256": sha256_file(initial_path),
            "state_sha256_initial_through_step4": actual_state_hashes,
            "state_fnv64_initial_through_step4": actual_state_fnv,
            "oracle": (
                "Radiance independent full-field binary32 oracle; every input "
                "conversion, multiply, and left-associated add is explicitly rounded"
            ),
        },
        "contract": {
            "coverage": (
                "all 32,768 outputs at each of four steps (131,072 comparisons)"
            ),
            "nonfinite": "always rejected",
            "strict_mechanical_wrapper": "zero ULP; every float32 word exact",
            "opencl_or_contracted": {
                "step_1_maximum_ulp": stage_ulp_limit(1),
                "step_2_maximum_ulp": stage_ulp_limit(2),
                "step_3_maximum_ulp": stage_ulp_limit(3),
                "step_4_maximum_ulp": stage_ulp_limit(4),
                "basis": (
                    "Radiance's documented conservative 16 ULP per timestep, "
                    "with the existing 32 ULP minimum"
                ),
            },
        },
        "source_execution": {
            "kernel_embedded_byte_exactly_once": wrapper.count(kernel) == 1,
            "native_wrapper_sha256": sha256_bytes(wrapper),
            "wrapper_transform": (
                "prefix defines __kernel, __global, and two-dimensional "
                "get_global_id; suffix supplies file I/O and the unchanged launch loop"
            ),
            "opencl_cpu_attempt": opencl_attempt,
            "spir_compilation": spir,
            "selected_runtime_path": (
                "native-opencl-cpu-plus-mechanical-controls"
                if opencl_attempt["status"] == "executed"
                else "cpu-mechanical-wrapper-strict-and-contracted"
            ),
        },
        "executions": executions,
        "harness": {
            "upstream_diff_sha256": sha256_file(Path(__file__).resolve()),
            "opencl_runner_sha256": sha256_file(OPENCL_RUNNER),
        },
        "resource_policy": memory_limit,
        "environment": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "byteorder": sys.byteorder,
            "cc": cc,
            "clang": clang,
        },
        "limitations": limitations,
    }
    if not passed:
        raise RuntimeError("one or more Hotspot3D differential executions failed")
    return evidence


def validate_evidence(evidence: dict) -> None:
    if evidence.get("schema") != "radiance-hotspot3d-upstream-differential-v1":
        raise ValueError("unexpected evidence schema")
    if evidence.get("status") != "passed":
        raise ValueError("evidence did not pass")
    reference = evidence["reference"]
    expected_kernel = VORTEX_OBJECTS[VORTEX_KERNEL_PATH]
    if reference["revision"] != VORTEX_REVISION:
        raise ValueError("wrong Vortex revision")
    if reference["kernel_git_blob"] != expected_kernel["blob"]:
        raise ValueError("wrong Vortex kernel Git blob")
    if reference["kernel_sha256"] != expected_kernel["sha256"]:
        raise ValueError("wrong Vortex kernel SHA-256")
    fixture = evidence["radiance_fixture"]
    fixture_identity = (
        fixture["base_revision"], fixture["nx"], fixture["ny"], fixture["nz"],
        fixture["iterations"], fixture["elements_per_state"],
        fixture["power_sha256"], fixture["initial_sha256"],
        fixture["state_sha256_initial_through_step4"],
        fixture["state_fnv64_initial_through_step4"],
        fixture["coefficient_words"],
    )
    expected_identity = (
        RADIANCE_BASE_REVISION, NX, NY, NZ, ITERATIONS, ELEMENTS,
        POWER_SHA256, INITIAL_SHA256, STATE_SHA256, STATE_FNV64,
        COEFFICIENT_WORDS,
    )
    if fixture_identity != expected_identity:
        raise ValueError("evidence does not identify the exact measured fixture")
    names = [execution["name"] for execution in evidence["executions"]]
    if "cpu-strict" not in names or "cpu-contracted" not in names:
        raise ValueError("both mechanical CPU evaluation modes are required")
    for execution in evidence["executions"]:
        if not execution["passed"] or len(execution["steps"]) != ITERATIONS:
            raise ValueError(f"execution did not pass all steps: {execution['name']}")
        if execution["records_compared_total"] != UPDATES:
            raise ValueError("execution did not compare every output at every step")
        for step_index, step in enumerate(execution["steps"], 1):
            if step["step"] != step_index or step["records_compared"] != ELEMENTS:
                raise ValueError("step coverage mismatch")
            if step["nonfinite"] or step["over_limit"] or not step["passed"]:
                raise ValueError("numerical contract failure")
            if execution["requires_bit_exact"]:
                if step["exact_word_matches"] != ELEMENTS:
                    raise ValueError("strict wrapper is not bit exact")
                if step["maximum_ulp_distance"] != 0:
                    raise ValueError("strict wrapper has nonzero ULP error")
    policy = evidence["resource_policy"]
    if policy["soft_bytes"] > MEMORY_LIMIT_BYTES or policy["hard_bytes"] > MEMORY_LIMIT_BYTES:
        raise ValueError("evidence was not produced under <=4 GiB RLIMIT_AS")
    harness = evidence["harness"]
    if harness["upstream_diff_sha256"] != sha256_file(Path(__file__).resolve()):
        raise ValueError("evidence was generated by a different differential harness")
    if harness["opencl_runner_sha256"] != sha256_file(OPENCL_RUNNER):
        raise ValueError("evidence was generated by a different OpenCL runner")


def default_vortex_root() -> Path | None:
    configured = os.environ.get("VORTEX_ROOT")
    if configured:
        return Path(configured)
    candidate = Path("/scratch/agustin/radiance-fpga-1a/reference/vortex")
    return candidate if candidate.is_dir() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".build")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vortex-root", type=Path, default=default_vortex_root())
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--clang", default="clang")
    args = parser.parse_args()
    evidence = generate_evidence(
        args.build_dir.resolve(), args.vortex_root, args.cc, args.clang
    )
    validate_evidence(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": evidence["status"],
        "fixture": evidence["scope"],
        "selected_runtime_path": evidence["source_execution"]["selected_runtime_path"],
        "executions": [{
            "name": execution["name"],
            "exact_by_step": [stage["exact_word_matches"] for stage in execution["steps"]],
            "max_ulp_by_step": [stage["maximum_ulp_distance"] for stage in execution["steps"]],
        } for execution in evidence["executions"]],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
