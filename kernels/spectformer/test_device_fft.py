"""Compare the Radiance row FFT with NumPy at a SpectFormer grid size."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "kernels/model_chain"))
from compile_decoder import build_device  # noqa: E402


def aligned(value):
    return (value + 63) & ~63


class DeviceFftTest(unittest.TestCase):
    def test_row_fft_matches_reference(self):
        self.check_grid(14, 14, 3)

    def test_larger_row_ffts_match_reference(self):
        for grid in ((28, 28, 2), (56, 56, 1)):
            with self.subTest(grid=grid):
                self.check_grid(*grid)

    def check_grid(self, height, width, channels):
        cyclotron_root = Path(os.environ.get(
            "CYCLOTRON_ROOT", "/scratch/agustin/projects/chipyard/generators/radiance/cyclotron"))
        simulator = cyclotron_root / "target/release/cyclotron"
        toolchain = Path(os.environ.get(
            "RISCV_TOOLCHAIN_PATH", "/scratch/agustin/projects/chipyard/.conda-env/riscv-tools"))
        llvm_muon = Path(os.environ.get("LLVM_MUON", str(REPO / "llvm/llvm-muon")))
        for needed in (simulator, cyclotron_root / "config.toml",
                       toolchain / "bin/riscv64-unknown-elf-nm", llvm_muon):
            if not needed.exists():
                self.skipTest(f"device test dependency missing: {needed}")

        rng = np.random.default_rng(8421)
        values = rng.normal(size=(height, width, channels)).astype("<f4")
        expected = np.fft.rfft(values, axis=1, norm="ortho").astype("<c8")
        bins = width // 2 + 1
        phases = 2 * np.pi * np.outer(np.arange(width), np.arange(width)) / width
        cosine = np.cos(phases).astype("<f4")
        sine = np.sin(phases).astype("<f4")
        output_size = height * bins * channels * 2 * 4
        output_offset = aligned(values.nbytes)
        cosine_offset = aligned(output_offset + output_size)
        sine_offset = aligned(cosine_offset + cosine.nbytes)
        arena_size = aligned(sine_offset + sine.nbytes)
        initial = bytearray(arena_size)
        for offset, data in ((0, values), (cosine_offset, cosine), (sine_offset, sine)):
            initial[offset:offset + data.nbytes] = data.tobytes()

        temporary_root = HERE.parents[2] / ".tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="spectformer-fft-", dir=temporary_root) as name:
            target = Path(name)
            shutil.copy2(HERE / "spectral.hpp", target / "spectral.hpp")
            (target / "input.bin").write_bytes(initial)
            (target / "model_data.cpp").write_text(
                "#include <mu_intrinsics.h>\n"
                f"alignas(64) __global unsigned char v_arena[{arena_size}] = {{0}};\n")
            (target / "kernel.cpp").write_text(f'''
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
#include "spectral.hpp"
extern "C" uint32_t __mu_num_warps = 1;
extern __global unsigned char v_arena[];
static void stage(void*, uint32_t tid, uint32_t tpb, uint32_t) {{
  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return;
  tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS
      + tid % MU_NUM_THREADS;
  tpb /= MU_NUM_CORES;
  spectformer::real_dft_rows((const __global float*)v_arena,
      (__global float*)(v_arena + {output_offset}),
      (const __global float*)(v_arena + {cosine_offset}),
      (const __global float*)(v_arena + {sine_offset}),
      {height}u, {width}u, {channels}u, tid, tpb);
}}
int main() {{
  mu_schedule(stage, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES);
  mu_fence();
  if (mu_hart_id() != 0) {{ for (;;) {{}} }}
  mu_tohost(0u); return 0;
}}
''')
            (target / "host.cpp").write_text('''
#include <inttypes.h>
#include <radiance.h>
int main() {
  tohost = 0; *tocpu = tohost;
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);
  uint32_t finished = 0;
  while (!finished) { SYNC_GPU(); finished = READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED); }
  return 0;
}
''')
            (target / "Makefile").write_text(f'''
PROJECT = spectformer
MU_SRCS = kernel.cpp
HOST_SRCS = host.cpp
MU_SRC_DEPS = model_data.cpp
RADIANCE_LIB_PATH := {REPO / 'lib'}
RADIANCE_INCLUDE_PATH := $(RADIANCE_LIB_PATH)/include
GEMMINI_SW_PATH := $(RADIANCE_LIB_PATH)/mxgemmini
SOC_DIR := {REPO / 'soc'}
LLVM_MUON ?= {llvm_muon}
EXTRA_MU_CFLAGS += -I{REPO / 'kernels/model_chain'}
include {REPO / 'kernels/common.mk'}
kernel.mu.o: spectral.hpp
''')
            (target / "manifest.json").write_text("{}\n")
            try:
                build_device(target)
            except subprocess.CalledProcessError as error:
                self.fail(f"{error}\n{(target / 'build.log').read_text()[-4000:]}")
            elf = target / "kernel.radiance.elf"
            symbols = subprocess.check_output([
                str(toolchain / "bin/riscv64-unknown-elf-nm"), "-n", str(elf)], text=True)
            addresses = re.findall(r"(?m)^([0-9a-fA-F]+) [A-Za-z] v_arena$", symbols)
            self.assertEqual(len(addresses), 1)
            address = int(addresses[0], 16)
            config = (cyclotron_root / "config.toml").read_text()
            config, count = re.subn(r"(?m)^timeout\s*=\s*\d+", "timeout = 10000000", config, count=1)
            self.assertEqual(count, 1)
            config = config.replace('"config/timing/', f'"{cyclotron_root}/config/timing/')
            (target / "cyclotron-config.toml").write_text(config)
            env = os.environ.copy()
            env["CYCLOTRON_WEIGHTS"] = f"0x{address:x}:{target / 'input.bin'}"
            env["CYCLOTRON_DUMP_GMEM"] = (
                f"0x{address + output_offset:x}:{output_size}:{target / 'output.bin'}")
            env["RADIANCE_DISABLE_CYCLOTRON_TRACE"] = "1"
            result = subprocess.run([
                str(simulator), str(target / "cyclotron-config.toml"),
                "--binary-path", str(elf), "--gen-trace", "false"],
                cwd=cyclotron_root, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("isa-test passed with tohost=0", result.stdout + result.stderr)
            actual = np.fromfile(target / "output.bin", dtype="<f4").view("<c8")
            actual = actual.reshape(height, bins, channels)
            np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
