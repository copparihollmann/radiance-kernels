#include <stdint.h>
#include <radiance.h>
#include "generated/config.h"
#include <nightly/host.h>
#if defined(RAD_HOST_TIMING) || defined(STREAM_DIAGNOSTIC_READBACK)
#include <rad_host.h>
#endif

#include "generated/symbols.h"

static uint64_t digest_output() {
  volatile uint32_t* output = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_OUTPUT_ADDR));
  uint64_t digest = 0xCBF29CE484222325ULL;
#if STREAM_READBACK_SAMPLES
  for (uint32_t j = 0; j < STREAM_READBACK_SAMPLES; ++j) {
    const uint32_t i = STREAM_READBACK_SAMPLES == 1 ? 0 :
        static_cast<uint32_t>((static_cast<uint64_t>(j) *
                               (STREAM_ELEMENTS - 1u)) /
                              (STREAM_READBACK_SAMPLES - 1u));
    digest ^= output[i];
    digest *= 0x100000001B3ULL;
  }
#else
  for (uint32_t i = 0; i < STREAM_ELEMENTS; ++i) {
    digest ^= output[i];
    digest *= 0x100000001B3ULL;
  }
#endif
  return digest;
}

static bool guards_intact() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16; ++i) {
    if (before[i] != (0xA5A50000u ^ i) ||
        after[i] != (0x5A5A0000u ^ i)) return false;
  }
  return true;
}

int main() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16; ++i) {
    before[i] = 0xA5A50000u ^ i;
    after[i] = 0x5A5A0000u ^ i;
  }
  asm volatile("fence rw, rw" ::: "memory");
#ifdef RAD_HOST_TIMING
  const uint64_t release_cycle = nightly_rdcycle();
#endif
  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int completed = nightly_host_wait(NIGHTLY_CLUSTERS);
#ifdef RAD_HOST_TIMING
  const uint64_t completion_cycle = nightly_rdcycle();
#endif
  nightly_host_report(NIGHTLY_CLUSTERS);
  asm volatile("fence rw, rw" ::: "memory");
  const bool guards_ok = guards_intact();
  const uint64_t actual_digest = digest_output();
  const bool passed = completed && guards_ok &&
      actual_digest == STREAM_EXPECTED_READBACK_DIGEST;
#ifdef STREAM_DIAGNOSTIC_READBACK
  if (!passed) {
    rad_puts("STREAM_DIAG completed="); rad_putu(completed);
    rad_puts(" guards="); rad_putu(guards_ok);
    rad_puts(" digest_actual=");
    rad_putx(static_cast<uint32_t>(actual_digest >> 32));
    rad_putx(static_cast<uint32_t>(actual_digest));
    rad_puts(" expected=");
    rad_putx(static_cast<uint32_t>(STREAM_EXPECTED_READBACK_DIGEST >> 32));
    rad_putx(static_cast<uint32_t>(STREAM_EXPECTED_READBACK_DIGEST));
    rad_putc('\n');
    if (!guards_ok) {
      for (uint32_t i = 0; i < 16; ++i) {
        rad_puts("STREAM_DIAG guard "); rad_putu(i);
        rad_puts(" before="); rad_putx(before[i]);
        rad_puts(" expected_before="); rad_putx(0xA5A50000u ^ i);
        rad_puts(" after="); rad_putx(after[i]);
        rad_puts(" expected_after="); rad_putx(0x5A5A0000u ^ i);
        rad_putc('\n');
      }
    }
    rad_flush();
  }
#endif
#ifdef RAD_HOST_TIMING
  rad_puts("HOST_RELEASE_TO_DONE_CYCLES=");
  rad_putu(completion_cycle - release_cycle);
  rad_putc('\n');
  rad_flush();
#endif
  nh_puts(passed ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(passed ? 0 : 1);
}
