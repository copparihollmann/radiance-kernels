#include <stdint.h>
#include <radiance.h>
#include <nightly/host.h>

#include "generated/config.h"
#include "generated/symbols.h"

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
  nightly_host_report(NIGHTLY_CLUSTERS);
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
