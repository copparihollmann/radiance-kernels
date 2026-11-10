#include <stdint.h>
#include <radiance.h>

#include "generated/config.h"
#include "generated/symbols.h"
#include <nightly/host.h>

static uint64_t digest_output() {
  volatile uint32_t* output = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(GEMV_OUTPUT_ADDR));
  uint64_t digest = 0xCBF29CE484222325ULL;
  for (uint32_t i = 0; i < GEMV_OUTPUT_WORDS; ++i) {
    digest ^= output[i];
    digest *= 0x100000001B3ULL;
  }
  return digest;
}

static bool guards_intact() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(GEMV_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(GEMV_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16; ++i) {
    if (before[i] != (0xA5A50000u ^ i) ||
        after[i] != (0x5A5A0000u ^ i)) return false;
  }
  return true;
}

int main() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(GEMV_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(GEMV_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16; ++i) {
    before[i] = 0xA5A50000u ^ i;
    after[i] = 0x5A5A0000u ^ i;
  }
  asm volatile("fence rw, rw" ::: "memory");
  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int completed = nightly_host_wait(NIGHTLY_CLUSTERS);
  const uint32_t kernel_cycles = nightly_host_report(NIGHTLY_CLUSTERS);
  // Compute axis: GEMV_ROWS * GEMV_COLS BF16 fused multiply-adds (2 FLOP each) plus one
  // (lanes - 1) tree reduction per row.  GEMV_COLS is a whole multiple of the lane x unroll
  // step here, so no lane is masked off on a tail and executed == useful; the assert below
  // makes that a checked property of the build rather than an assumption.
  {
    const uint64_t rows = (uint64_t)GEMV_ROWS;
    const uint64_t cols = (uint64_t)GEMV_COLS;
    const uint64_t lanes = NIGHTLY_SIMT_LANES_PER_CORE;
    const uint64_t step = lanes * (uint64_t)GEMV_UNROLL * (uint64_t)GEMV_ILP;
    const int exact_tiling = (cols % step) == 0u;
    const uint64_t useful_ops = 2ull * rows * cols + rows * (lanes - 1ull);
    nh_puts("gemv_fma_count="); nh_putu(rows * cols); nh_puts("\n");
    nh_puts("gemv_lane_tree_adds="); nh_putu(rows * (lanes - 1ull)); nh_puts("\n");
    nh_puts("gemv_exact_lane_tiling="); nh_putu(exact_tiling ? 1u : 0u); nh_puts("\n");
    nightly_report_compute_scalar("flop", useful_ops, useful_ops, 0,
                                  NIGHTLY_CLUSTERS, kernel_cycles);
  }
  asm volatile("fence rw, rw" ::: "memory");
  const bool passed = completed && guards_intact() &&
      digest_output() == GEMV_EXPECTED_DIGEST;
  nh_puts(passed ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(passed ? 0 : 1);
}
