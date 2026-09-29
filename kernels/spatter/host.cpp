#include <stdint.h>
#include <radiance.h>

#include "generated/config.h"
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

int main() {
  tohost = 0;
  *tocpu = 0;
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPATTER_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPATTER_GUARD_AFTER_ADDR));
  for (unsigned i = 0; i < 16; ++i) {
    before[i] = 0xA5A50000u ^ i;
    after[i] = 0x5A5A0000u ^ i;
  }
  asm volatile("fence rw, rw" ::: "memory");
  WRITE_MMIO_32(RAD_HOST_GPU_RESET, 0);

  while (!READ_MMIO_32(RAD_HOST_GPU_ALL_FINISHED)) {}
  asm volatile("fence rw, rw" ::: "memory");

  bool passed = guards_intact();
#if SPATTER_EXPLORATORY
  // Read a destination known to be written. Its final value depends on
  // scheduling when several threads target the same address.
  volatile uint32_t* output = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(SPATTER_OUTPUT_ADDR));
  const uint32_t probe = SPATTER_EXPLORATORY_PROBE_INDEX * 2u;
  passed = passed && ((output[probe] | output[probe + 1u]) != 0u);
#else
#if SPATTER_READBACK_SAMPLES
  passed = passed && output_sample_digest() == SPATTER_EXPECTED_SAMPLE_DIGEST;
#else
  passed = passed && output_digest() == SPATTER_EXPECTED_DIGEST;
#endif
#endif
  // The fused startup loop ignores main()'s return value. HTIF uses 1 for
  // success and 3 for failure; publish only after the full readback check.
  tohost = passed ? 1 : 3;
  asm volatile("fence rw, rw" ::: "memory");
  for (;;) {}
}
