#include <stdint.h>
#include <radiance.h>
#include <nightly/host.h>

#include "generated/config.h"
#include "generated/symbols.h"
#include "generated/ops.h"

static bool finite_word(uint32_t bits) {
  return ((bits >> 23u) & 0xffu) != 0xffu;
}

static bool guards_intact() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPMV_JDS_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPMV_JDS_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16u; ++i) {
    if (before[i] != (0xA5D50000u ^ i) ||
        after[i] != (0x5A2A0000u ^ i))
      return false;
  }
  return true;
}

int main() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPMV_JDS_GUARD_BEFORE_ADDR));
  volatile uint32_t* output = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPMV_JDS_OUTPUT_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPMV_JDS_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16u; ++i) {
    before[i] = 0xA5D50000u ^ i;
    after[i] = 0x5A2A0000u ^ i;
  }
  // A NaN prefill makes omitted rows fail even when their correct value is 0.
  for (uint32_t i = 0; i < SPMV_DIMENSION; ++i) output[i] = 0x7fc00000u;
  asm volatile("fence rw, rw" ::: "memory");

  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int completed = nightly_host_wait(NIGHTLY_CLUSTERS);
  const uint32_t kernel_cycles = nightly_host_report(NIGHTLY_CLUSTERS);
  // Compute axis: 1 multiply + 1 ordered add per expanded nonzero.
  //
  // executed is counted from the device's own JDS group bounds (SPMV_JDS_GROUP_BOUNDS_ADDR,
  // SPMV_DIMENSION/32 + 1 reads).  process_rows() iterates over real output rows only, and
  // row_to_jds maps those bijectively onto JDS positions 0..SPMV_DIMENSION-1, so the executed
  // multiply-accumulates are sum over j < SPMV_DIMENSION of group_bounds[j / warp].  Every
  // row is visited exactly once under the block-cyclic schedule.
  //
  // Note that this is NOT the padded array length.  The JDS data array holds
  // warp * sum(group_bounds) entries, and the entries belonging to JDS rows at and beyond
  // SPMV_DIMENSION are never read.  Both numbers are published so the difference is visible.
  {
    const volatile uint32_t* bounds =
        reinterpret_cast<const volatile uint32_t*>(
            rad_device_to_host_address(SPMV_JDS_GROUP_BOUNDS_ADDR));
    const uint32_t warp = SPMV_JDS_WARP_SIZE;
    const uint32_t groups = (SPMV_DIMENSION + warp - 1u) / warp;
    const uint32_t full = SPMV_DIMENSION / warp;
    const uint32_t tail = SPMV_DIMENSION - full * warp;
    uint64_t sum_all = 0u, sum_full = 0u;
    for (uint32_t g = 0u; g < groups; ++g) {
      const uint64_t b = bounds[g];
      sum_all += b;
      if (g < full) sum_full += b;
    }
    const uint64_t array_entries = (uint64_t)warp * sum_all;
    const uint64_t executed_macs =
        (uint64_t)warp * sum_full + (tail ? (uint64_t)tail * (uint64_t)bounds[full] : 0ull);
    const uint64_t useful_ops = 2ull * (uint64_t)SPMV_EXPANDED_NNZ;
    const uint64_t executed_ops = 2ull * executed_macs;
    nh_puts("spmv_expanded_nnz="); nh_putu((uint64_t)SPMV_EXPANDED_NNZ); nh_puts("\n");
    nh_puts("spmv_executed_macs="); nh_putu(executed_macs); nh_puts("\n");
    nh_puts("spmv_jds_array_entries="); nh_putu(array_entries); nh_puts("\n");
    nh_puts("spmv_executed_zero_pad_macs=");
    nh_putu(executed_macs - (uint64_t)SPMV_EXPANDED_NNZ); nh_puts("\n");
    nh_puts("spmv_unvisited_pad_entries="); nh_putu(array_entries - executed_macs);
    nh_puts("\n");
    nightly_report_compute_scalar("flop", useful_ops, executed_ops, 1,
                                  NIGHTLY_CLUSTERS, kernel_cycles);
  }
  asm volatile("fence rw, rw" ::: "memory");

  volatile uint32_t* expected = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPMV_JDS_EXPECTED_ADDR));
  uint64_t digest = 0xCBF29CE484222325ULL;
  uint32_t exact = 0;
  uint32_t bad = 0;
  uint32_t nonfinite = 0;
  for (uint32_t i = 0; i < SPMV_DIMENSION; ++i) {
    const uint32_t actual_word = output[i];
    const uint32_t expected_word = expected[i];
    digest ^= actual_word;
    digest *= 0x100000001B3ULL;
    exact += actual_word == expected_word;
    nonfinite += !finite_word(actual_word);
    bad += actual_word != expected_word;
  }

  const bool guards_ok = guards_intact();
  const bool digest_ok = digest == SPMV_EXPECTED_DIGEST;
  nh_puts("SPMV_JDS_CHECK n="); nh_putu(SPMV_DIMENSION);
  nh_puts(" exact="); nh_putu(exact);
  nh_puts(" bad="); nh_putu(bad);
  nh_puts(" nonfinite="); nh_putu(nonfinite);
  nh_puts(" guards="); nh_putu(guards_ok);
  nh_puts(" digest="); nh_putx(digest, 16);
  nh_puts(" digest_ok="); nh_putu(digest_ok);
  nh_puts("\n");
  const bool passed = completed && guards_ok && bad == 0u &&
                      nonfinite == 0u && digest_ok;
  nh_puts(passed ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(passed ? 0 : 1);
}
