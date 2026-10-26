#!/usr/bin/env python3
"""Differentially execute pinned Vortex Nearn on Radiance's exact 1M fixture."""

from __future__ import annotations

from array import array
import argparse
import base64
import datetime as dt
import hashlib
import heapq
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parent
NEARN_ROOT = ROOT.parent
RUN_PATH = NEARN_ROOT / "run.py"
PIN_B64 = ROOT / "vortex-nearn-kernel.cl.b64"
OPENCL_RUNNER = ROOT / "opencl_runner.c"
VORTEX_REVISION = "b447578d9f3bb116d658c614cf3407153f0536da"
VORTEX_KERNEL_PATH = "tests/opencl/nearn/kernel.cl"
VORTEX_KERNEL_SHA256 = (
    "43d6322e0be3adddb4c060b05dcb77a2ffd489ccdd07a5c4294e2f5f7cc17b34"
)
VORTEX_KERNEL_BYTES = 662
FULL_RECORDS = 1_048_576
FULL_INPUT_SHA256 = (
    "69edd36c8dbd9a94327bf6765867ece0823428be8e96f8da2dd0d1aaec4403b1"
)
FULL_EXPECTED_SHA256 = (
    "070b09a4b8de6a7cd60fa32f5548dd6c84c62587955503c2787b133acc29822b"
)
FULL_EXPECTED_DIGEST = "ce40a65c08c744c7"
FULL_TOP_FIVE = [1045179, 524842, 895400, 816969, 497552]
MAX_ULP = 1
OPENCL_UNAVAILABLE = 77


def load_nearn_run():
    spec = importlib.util.spec_from_file_location("nearn_fixture_run", RUN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def pinned_kernel() -> bytes:
    encoded = b"".join(PIN_B64.read_bytes().split())
    data = base64.b64decode(encoded, validate=True)
    if len(data) != VORTEX_KERNEL_BYTES:
        raise RuntimeError(
            f"pinned kernel has {len(data)} bytes, expected {VORTEX_KERNEL_BYTES}"
        )
    if sha256_bytes(data) != VORTEX_KERNEL_SHA256:
        raise RuntimeError("pinned Vortex kernel SHA-256 mismatch")
    return data


def command_output(command: list[str], **kwargs) -> str:
    return subprocess.run(
        command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, **kwargs
    ).stdout.strip()


def verify_vortex_checkout(path: Path | None, kernel: bytes) -> dict:
    if path is None:
        return {
            "status": "not-requested",
            "note": "vendored byte-exact source remains independently hash-pinned",
        }
    path = path.resolve()
    revision_type = command_output(
        ["git", "-C", str(path), "cat-file", "-t", VORTEX_REVISION]
    )
    if revision_type != "commit":
        raise RuntimeError(f"{VORTEX_REVISION} is not a commit in {path}")
    source = subprocess.run(
        ["git", "-C", str(path), "show",
         f"{VORTEX_REVISION}:{VORTEX_KERNEL_PATH}"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    if source != kernel:
        raise RuntimeError("vendored kernel differs byte-for-byte from Vortex pin")
    return {
        "status": "passed",
        "checkout": str(path),
        "object_type": revision_type,
        "byte_exact": True,
        "source_sha256": sha256_bytes(source),
    }


def compiler_identity(executable: str) -> dict:
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(f"required compiler not found: {executable}")
    version = command_output([resolved, "--version"]).splitlines()
    return {"path": resolved, "version": version[0]}


def write_f32(path: Path, values: Iterable[float]) -> None:
    data = array("f", values)
    if sys.byteorder != "little":
        data.byteswap()
    path.write_bytes(data.tobytes())


def read_words(path: Path, records: int) -> array:
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


def word_to_float(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def float_to_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def ordered_word(word: int) -> int:
    return (~word & 0xFFFFFFFF) if word & 0x80000000 else word | 0x80000000


def fnv_digest(words: Iterable[int]) -> str:
    value = 0xCBF29CE484222325
    for word in words:
        value = ((value ^ word) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def stable_top_five(words: array) -> list[int]:
    return heapq.nsmallest(
        5, range(len(words)), key=lambda index: (word_to_float(words[index]), index)
    )


def compare_words(actual: array, expected: array) -> dict:
    if len(actual) != len(expected):
        raise RuntimeError("output lengths differ")
    exact = 0
    one_ulp = 0
    over_limit = 0
    max_ulp = 0
    first_over_limit: list[dict] = []
    for index, (actual_word, expected_word) in enumerate(zip(actual, expected)):
        distance = abs(ordered_word(actual_word) - ordered_word(expected_word))
        max_ulp = max(max_ulp, distance)
        if distance == 0:
            exact += 1
        elif distance == 1:
            one_ulp += 1
        else:
            over_limit += 1
            if len(first_over_limit) < 8:
                first_over_limit.append({
                    "index": index,
                    "actual_word": f"{actual_word:08x}",
                    "expected_word": f"{expected_word:08x}",
                    "ulp_distance": distance,
                })
    actual_top = stable_top_five(actual)
    expected_top = stable_top_five(expected)
    return {
        "records_compared": len(actual),
        "max_ulp_contract": MAX_ULP,
        "exact_word_matches": exact,
        "one_ulp_matches": one_ulp,
        "over_one_ulp": over_limit,
        "maximum_ulp_distance": max_ulp,
        "first_over_limit": first_over_limit,
        "actual_top_five_indices": actual_top,
        "expected_top_five_indices": expected_top,
        "stable_top_five_exact": actual_top == expected_top,
        "passed": over_limit == 0 and actual_top == expected_top,
    }


def native_source(kernel: bytes) -> bytes:
    prefix = b"""#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t nearn_cpu_global_id;
#define __kernel __attribute__((noinline))
#define __global
#define get_global_id(dimension) (nearn_cpu_global_id)
#define sqrt sqrtf
#line 1 \"vortex/tests/opencl/nearn/kernel.cl\"
"""
    suffix = b"""
#undef sqrt
#undef get_global_id
#undef __global
#undef __kernel

static void die(const char *message, const char *path) {
  fprintf(stderr, \"%s: %s (%s)\\n\", message, path, strerror(errno));
  exit(2);
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

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, \"usage: %s INPUT OUTPUT RECORDS LAT_WORD LNG_WORD\\n\", argv[0]);
    return 2;
  }
  char *end = NULL;
  unsigned long parsed = strtoul(argv[3], &end, 10);
  if (end == argv[3] || *end != '\\0' || parsed == 0 || parsed > INT32_MAX)
    return 2;
  int records = (int)parsed;
  size_t input_bytes = (size_t)records * sizeof(LatLong);
  size_t output_bytes = (size_t)records * sizeof(float);
  FILE *input = fopen(argv[1], \"rb\");
  if (input == NULL) die(\"cannot open\", argv[1]);
  LatLong *locations = malloc(input_bytes);
  float *distances = malloc(output_bytes);
  if (locations == NULL || distances == NULL) return 2;
  if (fread(locations, 1, input_bytes, input) != input_bytes ||
      fgetc(input) != EOF) return 2;
  fclose(input);
  float lat = float_from_word(argv[4]);
  float lng = float_from_word(argv[5]);
  for (nearn_cpu_global_id = 0; nearn_cpu_global_id < (size_t)records;
       ++nearn_cpu_global_id)
    NearestNeighbor(locations, distances, records, lat, lng);
  FILE *output = fopen(argv[2], \"wb\");
  if (output == NULL) die(\"cannot open\", argv[2]);
  if (fwrite(distances, 1, output_bytes, output) != output_bytes ||
      fclose(output) != 0) die(\"cannot write\", argv[2]);
  free(distances);
  free(locations);
  return 0;
}
"""
    return prefix + kernel + suffix


def run_checked(command: list[str], *, cwd: Path | None = None) -> dict:
    completed = subprocess.run(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return {
        "command": [str(item) for item in command],
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def compile_spir(kernel_path: Path, build: Path, clang: str) -> dict:
    bitcode = build / "vortex-nearn.spir64.bc"
    command = [
        clang, "-x", "cl", "-cl-std=CL1.2",
        "-target", "spir64-unknown-unknown", "-emit-llvm", "-c",
        str(kernel_path), "-o", str(bitcode),
    ]
    run_checked(command)
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
    run_checked([
        cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
        str(OPENCL_RUNNER), "-lOpenCL", "-o", str(executable),
    ])
    return executable


def try_opencl(
    executable: Path, kernel_path: Path, input_path: Path, output_path: Path,
    records: int, lat_word: int, lng_word: int,
) -> dict:
    command = [
        str(executable), str(kernel_path), str(input_path), str(output_path),
        str(records), f"{lat_word:08x}", f"{lng_word:08x}",
    ]
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    result = {
        "attempted": True,
        "exit_code": completed.returncode,
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode == 0:
        result["status"] = "executed"
        return result
    if completed.returncode == OPENCL_UNAVAILABLE:
        result["status"] = "unavailable"
        return result
    raise RuntimeError(
        "an OpenCL platform was present but pinned-kernel execution failed:\n"
        + completed.stderr
    )


def compile_native_runner(
    build: Path, source: Path, cc: str, name: str, contract: str
) -> tuple[Path, list[str]]:
    executable = build / f"native-{name}"
    flags = [
        "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
        "-fno-fast-math", f"-ffp-contract={contract}",
        "-fexcess-precision=standard",
    ]
    if contract == "fast":
        # The audit host is x86_64. Generic x86-64 does not expose FMA to GCC,
        # which would make the nominally contracted control identical to the
        # strict build. Enabling the instruction makes this a real contraction
        # sensitivity check; the evidence records both the flag and output.
        flags.append("-mfma")
    run_checked([cc, *flags, str(source), "-lm", "-o", str(executable)])
    return executable, flags


def execute_native(
    executable: Path, input_path: Path, output_path: Path, records: int,
    lat_word: int, lng_word: int,
) -> None:
    run_checked([
        str(executable), str(input_path), str(output_path), str(records),
        f"{lat_word:08x}", f"{lng_word:08x}",
    ])


def generate_evidence(
    records: int, build: Path, vortex_root: Path | None,
    cc_name: str, clang_name: str,
) -> dict:
    build.mkdir(parents=True, exist_ok=True)
    kernel = pinned_kernel()
    kernel_path = build / "kernel.cl"
    kernel_path.write_bytes(kernel)
    source_path = build / "native-wrapper.c"
    source = native_source(kernel)
    source_path.write_bytes(source)

    nearn = load_nearn_run()
    query_lat, query_lng, locations = nearn.reference_dataset(records)
    expected_values = nearn.reference_output(query_lat, query_lng, locations)
    input_path = build / "input.bin"
    expected_path = build / "expected.bin"
    write_f32(input_path, (item for pair in locations for item in pair))
    write_f32(expected_path, expected_values)
    expected_words = read_words(expected_path, records)
    input_sha = sha256_file(input_path)
    expected_sha = sha256_file(expected_path)
    expected_digest = fnv_digest(expected_words)
    expected_top = stable_top_five(expected_words)
    if records == FULL_RECORDS:
        fixed = (input_sha, expected_sha, expected_digest, expected_top)
        wanted = (
            FULL_INPUT_SHA256, FULL_EXPECTED_SHA256, FULL_EXPECTED_DIGEST,
            FULL_TOP_FIVE,
        )
        if fixed != wanted:
            raise RuntimeError(f"1M fixture identity mismatch: {fixed!r}")

    cc = compiler_identity(cc_name)
    clang = compiler_identity(clang_name)
    spir = compile_spir(kernel_path, build, clang["path"])
    opencl_executable = compile_opencl_runner(build, cc["path"])
    opencl_output = build / "opencl-output.bin"
    lat_word = float_to_word(query_lat)
    lng_word = float_to_word(query_lng)
    opencl = try_opencl(
        opencl_executable, kernel_path, input_path, opencl_output, records,
        lat_word, lng_word,
    )

    executions: list[dict] = []
    if opencl["status"] == "executed":
        words = read_words(opencl_output, records)
        comparison = compare_words(words, expected_words)
        executions.append({
            "name": "opencl",
            "method": "native-opencl-runtime-byte-exact-pinned-kernel",
            "output_sha256": sha256_file(opencl_output),
            "output_digest": fnv_digest(words),
            "comparison": comparison,
        })
    else:
        for name, contract in (("strict", "off"), ("contracted", "fast")):
            executable, flags = compile_native_runner(
                build, source_path, cc["path"], name, contract
            )
            output = build / f"native-{name}-output.bin"
            execute_native(
                executable, input_path, output, records, lat_word, lng_word
            )
            words = read_words(output, records)
            comparison = compare_words(words, expected_words)
            executions.append({
                "name": f"cpu-{name}",
                "method": "mechanically-wrapped-byte-exact-pinned-kernel-body",
                "float_contract": contract,
                "compiler_flags": flags,
                "output_sha256": sha256_file(output),
                "output_digest": fnv_digest(words),
                "comparison": comparison,
            })

    passed = all(item["comparison"]["passed"] for item in executions)
    source_occurrences = source.count(kernel)
    if source_occurrences != 1:
        raise RuntimeError("generated native wrapper does not embed the pin once")
    checkout = verify_vortex_checkout(vortex_root, kernel)
    evidence = {
        "schema": "radiance-nearn-upstream-1m-differential-v1",
        "status": "passed" if passed else "failed",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "exact measured Radiance Nearn fixture",
        "reference": {
            "repository": "vortexgpgpu/vortex",
            "revision": VORTEX_REVISION,
            "path": VORTEX_KERNEL_PATH,
            "bytes": len(kernel),
            "sha256": sha256_bytes(kernel),
            "vendored_encoding": str(PIN_B64.relative_to(NEARN_ROOT.parents[1])),
            "checkout_verification": checkout,
        },
        "source_execution": {
            "kernel_embedded_byte_exactly_once": source_occurrences == 1,
            "native_wrapper_sha256": sha256_bytes(source),
            "opencl_attempt": opencl,
            "spir_compilation": spir,
            "selected_path": (
                "native-opencl" if opencl["status"] == "executed"
                else "cpu-mechanical-wrapper"
            ),
        },
        "fixture": {
            "records": records,
            "seed": nearn.REFERENCE_SEED,
            "input_layout": "packed little-endian pairs of float32 latitude/longitude",
            "input_bytes": input_path.stat().st_size,
            "input_sha256": input_sha,
            "query_lat_word": f"{lat_word:08x}",
            "query_lng_word": f"{lng_word:08x}",
            "expected_bytes": expected_path.stat().st_size,
            "expected_sha256": expected_sha,
            "expected_digest": expected_digest,
            "expected_top_five_indices": expected_top,
            "oracle": "Radiance measured-fixture fmaf/sqrtf binary32 oracle",
        },
        "contract": {
            "distance_words": "all records compared",
            "maximum_ulp": MAX_ULP,
            "top_five": "exact indices ordered by (float32 distance, original index)",
        },
        "executions": executions,
        "environment": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cc": cc,
            "clang": clang,
            "byteorder": sys.byteorder,
        },
        "limitations": (
            [] if opencl["status"] == "executed" else [
                "No OpenCL ICD/platform was available, so the pinned kernel was not "
                "executed by an OpenCL runtime on this host.",
                "The CPU fallback executes the unchanged kernel body through macro "
                "definitions for OpenCL qualifiers/get_global_id and maps OpenCL sqrt "
                "to binary32 sqrtf; it is not evidence about a particular OpenCL "
                "device compiler or FPGA execution.",
                "Both contraction-disabled and contraction-enabled builds are required "
                "to pass, bounding the material source-level float32 evaluation choice."
            ]
        ),
    }
    if not passed:
        raise RuntimeError("one or more direct-upstream differential executions failed")
    return evidence


def validate_evidence(evidence: dict, require_full: bool = True) -> None:
    if evidence.get("schema") != "radiance-nearn-upstream-1m-differential-v1":
        raise ValueError("unexpected evidence schema")
    if evidence.get("status") != "passed":
        raise ValueError("evidence did not pass")
    reference = evidence["reference"]
    if reference["revision"] != VORTEX_REVISION:
        raise ValueError("wrong Vortex revision")
    if reference["sha256"] != VORTEX_KERNEL_SHA256:
        raise ValueError("wrong Vortex kernel hash")
    fixture = evidence["fixture"]
    if require_full:
        expected = (
            FULL_RECORDS, FULL_INPUT_SHA256, FULL_EXPECTED_SHA256,
            FULL_EXPECTED_DIGEST, FULL_TOP_FIVE,
        )
        actual = (
            fixture["records"], fixture["input_sha256"],
            fixture["expected_sha256"], fixture["expected_digest"],
            fixture["expected_top_five_indices"],
        )
        if actual != expected:
            raise ValueError("evidence does not identify the exact measured 1M fixture")
    if not evidence["executions"]:
        raise ValueError("no differential execution")
    for execution in evidence["executions"]:
        comparison = execution["comparison"]
        if not comparison["passed"]:
            raise ValueError(f"execution failed: {execution['name']}")
        if comparison["records_compared"] != fixture["records"]:
            raise ValueError("execution did not compare every record")
        if comparison["over_one_ulp"] != 0:
            raise ValueError("execution exceeded the ULP contract")
        if not comparison["stable_top_five_exact"]:
            raise ValueError("execution top five differs")
    if evidence["source_execution"]["selected_path"] == "cpu-mechanical-wrapper":
        names = [item["name"] for item in evidence["executions"]]
        if names != ["cpu-strict", "cpu-contracted"]:
            raise ValueError("CPU fallback lacks both required evaluation modes")


def default_vortex_root() -> Path | None:
    value = os.environ.get("VORTEX_ROOT")
    if value:
        return Path(value)
    candidate = Path("/scratch/agustin/radiance-fpga-1a/reference/vortex")
    return candidate if candidate.is_dir() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=FULL_RECORDS)
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".build")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vortex-root", type=Path, default=default_vortex_root())
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--clang", default="clang")
    args = parser.parse_args()
    if args.records < 5 or args.records > FULL_RECORDS:
        parser.error(f"--records must be between 5 and {FULL_RECORDS}")
    evidence = generate_evidence(
        args.records, args.build_dir.resolve(), args.vortex_root,
        args.cc, args.clang,
    )
    validate_evidence(evidence, require_full=args.records == FULL_RECORDS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({
        "status": evidence["status"],
        "records": evidence["fixture"]["records"],
        "selected_path": evidence["source_execution"]["selected_path"],
        "executions": [
            {
                "name": item["name"],
                "maximum_ulp_distance": item["comparison"]["maximum_ulp_distance"],
                "over_one_ulp": item["comparison"]["over_one_ulp"],
                "stable_top_five_exact": item["comparison"]["stable_top_five_exact"],
            }
            for item in evidence["executions"]
        ],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
