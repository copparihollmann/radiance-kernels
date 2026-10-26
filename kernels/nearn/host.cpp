#include <stdint.h>
#include <radiance.h>
#include <nightly/host.h>

#include "generated/config.h"
#include "generated/symbols.h"

static uint32_t ordered_float(uint32_t bits) {
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

static uint32_t ulp_distance(uint32_t lhs, uint32_t rhs) {
  const uint32_t a = ordered_float(lhs);
  const uint32_t b = ordered_float(rhs);
  return a > b ? a - b : b - a;
}

static bool float_less(uint32_t lhs, uint32_t rhs) {
  const uint32_t lhs_magnitude = lhs & 0x7fffffffu;
  const uint32_t rhs_magnitude = rhs & 0x7fffffffu;
  /* Match IEEE-754 `lhs < rhs` without executing an F instruction on the
   * no-FPU Rocket host.  Comparisons involving NaNs are false, and signed
   * zeroes compare equal. */
  if (lhs_magnitude > 0x7f800000u || rhs_magnitude > 0x7f800000u)
    return false;
  if (lhs_magnitude == 0 && rhs_magnitude == 0)
    return false;
  return ordered_float(lhs) < ordered_float(rhs);
}

static uint64_t host_hart_id() {
  uint64_t id;
  asm volatile("csrr %0, mhartid" : "=r"(id) :: "memory");
  return id;
}

static bool guards_intact() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(NEARN_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(NEARN_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16; ++i) {
    if (before[i] != (0xA6EA0000u ^ i) ||
        after[i] != (0x59150000u ^ i)) return false;
  }
  return true;
}

int main() {
  /* The fused Rocket program is entered by every host hart.  Only hart 0 may
   * own the launch/reset/check protocol; secondary harts otherwise clear the
   * shared END slots and relaunch the GPU while hart 0 validates output. */
  if (host_hart_id() != 0)
    for (;;) asm volatile("wfi" ::: "memory");

  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(NEARN_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(NEARN_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16; ++i) {
    before[i] = 0xA6EA0000u ^ i;
    after[i] = 0x59150000u ^ i;
  }
  asm volatile("fence rw, rw" ::: "memory");
  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int completed = nightly_host_wait(NIGHTLY_CLUSTERS);
  nightly_host_report(NIGHTLY_CLUSTERS);
  asm volatile("fence rw, rw" ::: "memory");

  volatile uint32_t* got = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(NEARN_OUTPUT_ADDR));
  volatile uint32_t* expected = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(NEARN_EXPECTED_ADDR));
  uint32_t exact = 0;
  uint32_t bad = 0;
  uint32_t nonfinite = 0;
  uint32_t max_ulp = 0;
  uint32_t selected[NEARN_TOP_K];
  uint32_t selected_distance_bits[NEARN_TOP_K];
  for (uint32_t rank = 0; rank < NEARN_TOP_K; ++rank) {
    selected[rank] = 0xffffffffu;
    selected_distance_bits[rank] = 0x7f800000u;
  }
  for (uint32_t i = 0; i < NEARN_RECORDS; ++i) {
    const uint32_t actual = got[i];
    const uint32_t reference = expected[i];
    exact += actual == reference;
    nonfinite += ((actual >> 23) & 0xffu) == 0xffu;
    const uint32_t ulp = ulp_distance(actual, reference);
    if (ulp > max_ulp) max_ulp = ulp;
    bad += ulp > NEARN_MAX_ULP;

    /* Maintain the same stable (distance, input-index) top-k order as the
     * former rank-by-rank rescans.  Since i increases monotonically and the
     * comparison is strict, equal distances retain ascending input order. */
    uint32_t insert = NEARN_TOP_K;
    for (uint32_t rank = 0; rank < NEARN_TOP_K; ++rank) {
      if (float_less(actual, selected_distance_bits[rank])) {
        insert = rank;
        break;
      }
    }
    if (insert < NEARN_TOP_K) {
      for (uint32_t rank = NEARN_TOP_K - 1; rank > insert; --rank) {
        selected_distance_bits[rank] = selected_distance_bits[rank - 1];
        selected[rank] = selected[rank - 1];
      }
      selected_distance_bits[insert] = actual;
      selected[insert] = i;
    }
  }

  const uint32_t expected_top[NEARN_TOP_K] = NEARN_EXPECTED_TOP;
  uint32_t top_bad = 0;
  for (uint32_t rank = 0; rank < NEARN_TOP_K; ++rank)
    top_bad += selected[rank] != expected_top[rank];

  const bool guards_ok = guards_intact();
  nh_puts("NEARN_CHECK n="); nh_putu(NEARN_RECORDS);
  nh_puts(" exact="); nh_putu(exact);
  nh_puts(" bad="); nh_putu(bad);
  nh_puts(" max_ulp="); nh_putu(max_ulp);
  nh_puts(" nonfinite="); nh_putu(nonfinite);
  nh_puts(" top_bad="); nh_putu(top_bad);
  nh_puts(" guards="); nh_putu(guards_ok);
  nh_puts("\n");
  const bool passed = completed && guards_ok && bad == 0 &&
                      nonfinite == 0 && top_bad == 0;
  nh_puts(passed ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(passed ? 0 : 1);
}
