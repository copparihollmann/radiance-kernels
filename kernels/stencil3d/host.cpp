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
      rad_device_to_host_address(STENCIL3D_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STENCIL3D_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16; ++i) {
    if (before[i] != (0xA53D0000u ^ i) ||
        after[i] != (0x5AC20000u ^ i)) return false;
  }
  return true;
}

int main() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STENCIL3D_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STENCIL3D_GUARD_AFTER_ADDR));
  volatile uint32_t* got = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STENCIL3D_OUTPUT_ADDR));
  for (uint32_t i = 0; i < 16; ++i) {
    before[i] = 0xA53D0000u ^ i;
    after[i] = 0x5AC20000u ^ i;
  }
  // Every reference output is finite. A NaN prefill therefore makes an
  // omitted store fail independently of the prior contents of GPU memory.
  for (uint32_t i = 0; i < STENCIL3D_ELEMENTS; ++i)
    got[i] = 0x7fc00000u;
  asm volatile("fence rw, rw" ::: "memory");
  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int completed = nightly_host_wait(NIGHTLY_CLUSTERS);
  nightly_host_report(NIGHTLY_CLUSTERS);
  asm volatile("fence rw, rw" ::: "memory");

  volatile uint32_t* expected = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STENCIL3D_EXPECTED_ADDR));
  uint32_t exact = 0;
  uint32_t bad = 0;
  uint32_t nonfinite = 0;
  uint32_t poison = 0;
  uint32_t max_ulp = 0;
  for (uint32_t i = 0; i < STENCIL3D_ELEMENTS; ++i) {
    const uint32_t actual = got[i];
    const uint32_t reference = expected[i];
    exact += actual == reference;
    poison += actual == 0x7fc00000u;
    nonfinite += ((actual >> 23) & 0xffu) == 0xffu;
    const uint32_t ulp = ulp_distance(actual, reference);
    if (ulp > max_ulp) max_ulp = ulp;
    bad += ulp > STENCIL3D_MAX_ULP;
  }
  const bool guards_ok = guards_intact();
  const uint64_t output_digest = digest_words(got, STENCIL3D_ELEMENTS);
  const uint64_t reference_digest = digest_words(expected, STENCIL3D_ELEMENTS);
  const bool reference_digest_ok =
      reference_digest == STENCIL3D_EXPECTED_DIGEST;
  nh_puts("STENCIL3D_CHECK n="); nh_putu(STENCIL3D_ELEMENTS);
  nh_puts(" exact="); nh_putu(exact);
  nh_puts(" bad="); nh_putu(bad);
  nh_puts(" max_ulp="); nh_putu(max_ulp);
  nh_puts(" nonfinite="); nh_putu(nonfinite);
  nh_puts(" poison="); nh_putu(poison);
  nh_puts(" guards="); nh_putu(guards_ok);
  nh_puts(" output_digest="); nh_putx(output_digest, 16);
  nh_puts(" reference_digest_ok="); nh_putu(reference_digest_ok);
  nh_puts("\n");
  // The upstream contract permits six ULP, so output_digest is diagnostic;
  // reference_digest_ok anchors the embedded oracle without strengthening the
  // accepted numerical tolerance to bit identity.
  const bool passed = completed && guards_ok && reference_digest_ok &&
                      bad == 0 && nonfinite == 0 && poison == 0;
  nh_puts(passed ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(passed ? 0 : 1);
}
