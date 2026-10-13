#!/usr/bin/env python3
"""Run a source-derived SpectFormer MX-FP8 tile on real Spike libgemmini."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np


M, N, K = 16, 16, 32

PROGRAM = r'''#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "include/gemmini.h"
#include "tile_data.h"

int main(void) {
  static uint16_t actual[16][16] __attribute__((aligned(64)));
  static uint32_t output_scales[512];
  memset(actual, 0, sizeof(actual));
  const int ti = 1, tj = 1, tk = 2;
  const uint32_t a_base = 0;
  const uint32_t b_base = BANK_NUM * BANK_ROWS - tk * tj * DIM;
  const uint32_t dest = 256;
  gemmini_flush(0);
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
                              1, 1, 0, 0, false, 0, 0, 3, 0);
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
  gemmini_config_ld(32);
  for (int k = 0; k < tk; ++k)
    gemmini_extended_mvin((void*)(&A_in[0][k * DIM]), a_base + k * DIM, DIM, DIM);
  gemmini_config_ld(16);
  for (int k = 0; k < tk; ++k)
    gemmini_extended_mvin((void*)(&B_in[k * DIM][0]), b_base + k * DIM, DIM, DIM);
  gemmini_mxquant_config_mvout((uint64_t)output_scales, ti, tj, tk, 0, 0, 1);
  gemmini_loop_ws_spad(ti, tj, tk, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS,
                       0, dest, false, false, false, false, false,
                       NO_ACTIVATION, 0, 0, false, 0x38);
  gemmini_fence();
  gemmini_mx_read_smem(&actual[0][0], dest * DIM, 256);
  int errors = 0;
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n)
      if (actual[m][n] != expected[m][n]) ++errors;
  printf("spectformer_mx_spike errors=%d elements=256 first=%04x expected=%04x\n",
         errors, actual[0][0], expected[0][0]);
  return errors ? 1 : 0;
}
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def declaration(name: str, values: np.ndarray, ctype: str) -> str:
    rows = ",\n".join("  {" + ", ".join(str(int(x)) for x in row) + "}"
                        for row in values)
    return (f"static const {ctype} {name}[{values.shape[0]}][{values.shape[1]}] "
            f"__attribute__((aligned(64))) = {{\n{rows}\n}};\n")


def run(tile: Path, out: Path, gemmini_source: Path, spike: Path,
        gemmini_library_dir: Path, gcc: Path, pk: Path,
        timeout: int) -> dict:
    tile, out = tile.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_result = json.loads((tile / "result.json").read_text())
    arrays = {
        "A_in": np.fromfile(tile / "A.bin", dtype=np.uint8).reshape(M, K),
        "B_in": np.fromfile(tile / "B.bin", dtype=np.uint8).reshape(K, N),
        "A_scales_row": np.fromfile(tile / "SA.bin", dtype=np.uint8).reshape(K // 32, M),
        "B_scales_col": np.fromfile(tile / "SB.bin", dtype=np.uint8).reshape(K // 32, N),
        "expected": np.fromfile(tile / "C.bin", dtype="<u2").reshape(M, N),
    }
    for name, file in (("A_in", "A.bin"), ("B_in", "B.bin"),
                       ("A_scales_row", "SA.bin"), ("B_scales_col", "SB.bin")):
        if source_result["operand_sha256"][file] != sha256(tile / file):
            raise ValueError(f"{name} differs from the source-derived tile")
    if source_result["model_output_sha256"] != sha256(tile / "C.bin"):
        raise ValueError("MX golden output differs from the tile receipt")
    header = out / "tile_data.h"
    header.write_text("#pragma once\n#include <stdint.h>\n" + "".join(
        declaration(name, values, "uint16_t" if name == "expected" else "uint8_t")
        for name, values in arrays.items()))
    source = out / "tile.c"
    source.write_text(PROGRAM)
    binary = out / "tile.riscv"
    subprocess.run([str(gcc), "-static", "-O2", "-march=rv64gc", "-mabi=lp64d",
                    "-I", str(gemmini_source), "-I", str(out),
                    str(source), "-o", str(binary)], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = (str(gemmini_library_dir.resolve()) + ":" +
                                      environment.get("LD_LIBRARY_PATH", ""))
    try:
        process = subprocess.run([str(spike), "--extension=gemmini", str(pk),
                                  str(binary)], env=environment, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, timeout=timeout)
        output, exit_code = process.stdout, process.returncode
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        exit_code = 124
    log = out / "spike.log"
    log.write_text(output)
    matches = re.findall(r"spectformer_mx_spike errors=(\d+) elements=(\d+)", output)
    passed = exit_code == 0 and matches == [("0", "256")]
    record = {
        "scope": "source_derived_spectformer_projection_16x16x32_partial_tile",
        "status": "passed" if passed else "failed",
        "spike_execution": True, "radiance_device_execution": False,
        "full_model_execution": False, "performance_measurement": False,
        "elements_compared": 256, "bit_exact_vs_mx_golden": passed,
        "upstream_source_sha256": source_result["upstream_source_sha256"],
        "checkpoint_sha256": source_result["checkpoint_sha256"],
        "tile_result_sha256": sha256(tile / "result.json"),
        "tile_data_sha256": {name: sha256(tile / name)
                             for name in ("A.bin", "B.bin", "SA.bin", "SB.bin", "C.bin")},
        "spike_sha256": sha256(spike), "gemmini_library_sha256": sha256(
            gemmini_library_dir / "libgemmini.so"),
        "binary_sha256": sha256(binary), "log_sha256": sha256(log),
        "exit_code": exit_code, "output": output[-500:],
    }
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gemmini-source", type=Path, required=True,
                        help="MX-enabled gemmini-rocc-tests checkout")
    parser.add_argument("--spike", type=Path, required=True)
    parser.add_argument("--gemmini-library-dir", type=Path, required=True)
    parser.add_argument("--gcc", type=Path, required=True)
    parser.add_argument("--pk", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    result = run(args.tile, args.out, args.gemmini_source, args.spike,
                 args.gemmini_library_dir, args.gcc, args.pk, args.timeout)
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
