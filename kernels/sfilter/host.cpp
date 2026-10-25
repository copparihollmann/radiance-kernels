#include <stdint.h>

#include <nightly/host.h>
#include <radiance.h>

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

static uint64_t digest_words(volatile uint32_t* words, uint32_t count) {
  uint64_t digest = 0xCBF29CE484222325ULL;
  for (uint32_t i = 0; i < count; ++i) {
    digest ^= words[i];
    digest *= 0x100000001B3ULL;
  }
  return digest;
}

static bool guards_intact() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SFILTER_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SFILTER_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16u; ++i) {
    if (before[i] != (0xA5F10000u ^ i) ||
        after[i] != (0x5A0E0000u ^ i))
      return false;
  }
  return true;
}

int main() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SFILTER_GUARD_BEFORE_ADDR));
  volatile uint32_t* output = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SFILTER_OUTPUT_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SFILTER_GUARD_AFTER_ADDR));

  for (uint32_t i = 0; i < 16u; ++i) {
    before[i] = 0xA5F10000u ^ i;
    after[i] = 0x5A0E0000u ^ i;
  }
  // Keep the contract's zero boundary, but poison every interior word. A
  // skipped pixel therefore fails even when its expected result is otherwise
  // close to an old or zero value.
  for (uint32_t y = 0; y < SFILTER_SIZE; ++y) {
    for (uint32_t x = 0; x < SFILTER_SIZE; ++x) {
      const bool boundary = x == 0u || y == 0u ||
                            x + 1u == SFILTER_SIZE ||
                            y + 1u == SFILTER_SIZE;
      output[x + y * SFILTER_SIZE] = boundary ? 0u : 0x7fc00000u;
    }
  }
  asm volatile("fence rw, rw" ::: "memory");

  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int completed = nightly_host_wait(NIGHTLY_CLUSTERS);
  nightly_host_report(NIGHTLY_CLUSTERS);
  asm volatile("fence rw, rw" ::: "memory");

  volatile uint32_t* expected = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SFILTER_EXPECTED_ADDR));
  uint32_t exact = 0u;
  uint32_t bad = 0u;
  uint32_t nonfinite = 0u;
  uint32_t boundary_bad = 0u;
  uint32_t max_ulp = 0u;
  for (uint32_t y = 0; y < SFILTER_SIZE; ++y) {
    for (uint32_t x = 0; x < SFILTER_SIZE; ++x) {
      const uint32_t index = x + y * SFILTER_SIZE;
      const uint32_t actual = output[index];
      const uint32_t reference = expected[index];
      const bool boundary = x == 0u || y == 0u ||
                            x + 1u == SFILTER_SIZE ||
                            y + 1u == SFILTER_SIZE;
      const bool finite = ((actual >> 23u) & 0xffu) != 0xffu;
      const uint32_t ulp = ulp_distance(actual, reference);
      exact += actual == reference;
      nonfinite += !finite;
      boundary_bad += boundary && actual != 0u;
      if (ulp > max_ulp) max_ulp = ulp;
      bad += !finite || ulp > SFILTER_MAX_ULP;
    }
  }

  const bool guards_ok = guards_intact();
  const uint64_t actual_digest = digest_words(output, SFILTER_ELEMENTS);
  const uint64_t reference_digest = digest_words(expected, SFILTER_ELEMENTS);
  const bool reference_digest_ok =
      reference_digest == SFILTER_EXPECTED_DIGEST;
  const bool exact_digest = actual_digest == SFILTER_EXPECTED_DIGEST;

  nh_puts("SFILTER_CHECK n="); nh_putu(SFILTER_ELEMENTS);
  nh_puts(" exact="); nh_putu(exact);
  nh_puts(" bad="); nh_putu(bad);
  nh_puts(" max_ulp="); nh_putu(max_ulp);
  nh_puts(" nonfinite="); nh_putu(nonfinite);
  nh_puts(" boundary_bad="); nh_putu(boundary_bad);
  nh_puts(" guards="); nh_putu(guards_ok);
  nh_puts(" actual_digest="); nh_putx(actual_digest, 16);
  nh_puts(" exact_digest="); nh_putu(exact_digest);
  nh_puts(" reference_digest_ok="); nh_putu(reference_digest_ok);
  nh_puts("\n");

  // The upstream checker accepts up to four ULP, so exact_digest is reported
  // for reproducibility but is not a stricter replacement for that contract.
  const bool passed = completed && guards_ok && reference_digest_ok &&
                      bad == 0u && nonfinite == 0u && boundary_bad == 0u;
  nh_puts(passed ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(passed ? 0 : 1);
}
