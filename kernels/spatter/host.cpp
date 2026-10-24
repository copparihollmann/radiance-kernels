#include <stdint.h>
#include <radiance.h>
#include "generated/config.h"
#include <nightly/host.h>
#if SPATTER_DIAGNOSTIC_READBACK || defined(RAD_HOST_TIMING)
#include <rad_host.h>
#endif

#include "generated/symbols.h"

static uint64_t output_digest() {
  volatile uint32_t* words = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPATTER_OUTPUT_ADDR));
  uint64_t digest = 0xCBF29CE484222325ULL;
  for (uint32_t i = 0; i < SPATTER_OUTPUT_LENGTH * 2u; ++i) {
    digest ^= words[i];
    digest *= 0x100000001B3ULL;
  }
  return digest;
}

static uint64_t output_sample_digest() {
  volatile uint32_t* words = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPATTER_OUTPUT_ADDR));
  uint64_t digest = 0xCBF29CE484222325ULL;
  for (uint32_t i = 0; i < SPATTER_READBACK_SAMPLES; ++i) {
    const uint32_t pos = SPATTER_READBACK_SAMPLES == 1 ? 0 :
        static_cast<uint32_t>((static_cast<uint64_t>(i) *
                               (SPATTER_OUTPUT_LENGTH - 1u)) /
                              (SPATTER_READBACK_SAMPLES - 1u));
    digest ^= words[2u * pos];
    digest *= 0x100000001B3ULL;
    digest ^= words[2u * pos + 1u];
    digest *= 0x100000001B3ULL;
  }
  return digest;
}

static bool guards_intact() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPATTER_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPATTER_GUARD_AFTER_ADDR));
  for (unsigned i = 0; i < 16; ++i) {
    if (before[i] != (0xA5A50000u ^ i) ||
        after[i] != (0x5A5A0000u ^ i)) return false;
  }
  return true;
}

static void clear_output() {
  volatile uint32_t* words = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPATTER_OUTPUT_ADDR));
  for (uint64_t i = 0; i < static_cast<uint64_t>(SPATTER_OUTPUT_LENGTH) * 2u; ++i) {
    words[i] = 0;
  }
}

int main() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPATTER_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPATTER_GUARD_AFTER_ADDR));
  for (unsigned i = 0; i < 16; ++i) {
    before[i] = 0xA5A50000u ^ i;
    after[i] = 0x5A5A0000u ^ i;
  }
  clear_output();
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

  bool passed = completed && guards_intact();
#if SPATTER_EXPLORATORY
  // Collisions make the final values schedule-dependent, but every source
  // payload is nonzero. Check exact destination coverage, not one probe: a
  // wrong SM partition can otherwise skip half the tasks and still pass.
  volatile uint32_t* output = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPATTER_OUTPUT_ADDR));
  uint32_t nonzero_outputs = 0;
  for (uint32_t i = 0; i < SPATTER_OUTPUT_LENGTH; ++i)
    nonzero_outputs += (output[2u * i] | output[2u * i + 1u]) != 0u;
  passed = passed && nonzero_outputs == SPATTER_EXPECTED_NONZERO_OUTPUTS;
#elif SPATTER_FORCE_FULL_READBACK
  const uint64_t actual_digest = output_digest();
  passed = passed && actual_digest == SPATTER_EXPECTED_DIGEST;
#if SPATTER_DIAGNOSTIC_READBACK
  if (!passed) {
    rad_puts("SPATTER_DIAG full_digest_actual=");
    rad_putx(static_cast<uint32_t>(actual_digest >> 32));
    rad_putx(static_cast<uint32_t>(actual_digest));
    rad_puts(" expected=");
    rad_putx(static_cast<uint32_t>(SPATTER_EXPECTED_DIGEST >> 32));
    rad_putx(static_cast<uint32_t>(SPATTER_EXPECTED_DIGEST));
    rad_putc('\n'); rad_flush();
  }
#endif
#elif SPATTER_READBACK_SAMPLES
  const uint64_t actual_sample_digest = output_sample_digest();
  passed = passed && actual_sample_digest == SPATTER_EXPECTED_SAMPLE_DIGEST;
#if SPATTER_DIAGNOSTIC_READBACK
  if (!passed) {
    rad_puts("SPATTER_DIAG guards="); rad_putu(guards_intact());
    rad_puts(" actual="); rad_putx(static_cast<uint32_t>(actual_sample_digest >> 32));
    rad_putx(static_cast<uint32_t>(actual_sample_digest));
    rad_puts(" expected="); rad_putx(static_cast<uint32_t>(SPATTER_EXPECTED_SAMPLE_DIGEST >> 32));
    rad_putx(static_cast<uint32_t>(SPATTER_EXPECTED_SAMPLE_DIGEST));
    rad_putc('\n');
    volatile uint32_t* words = reinterpret_cast<volatile uint32_t*>(
        rad_device_to_host_address(SPATTER_OUTPUT_ADDR));
    for (uint32_t i = 0; i < SPATTER_READBACK_SAMPLES; ++i) {
      const uint32_t pos = SPATTER_READBACK_SAMPLES == 1 ? 0 :
          static_cast<uint32_t>((static_cast<uint64_t>(i) *
                                 (SPATTER_OUTPUT_LENGTH - 1u)) /
                                (SPATTER_READBACK_SAMPLES - 1u));
      rad_putu(i); rad_putc(' '); rad_putu(pos); rad_putc(' ');
      rad_putx(words[2u * pos]); rad_putc(' '); rad_putx(words[2u * pos + 1u]);
      rad_putc('\n');
    }
    rad_flush();
  }
#endif
#else
  passed = passed && output_digest() == SPATTER_EXPECTED_DIGEST;
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
