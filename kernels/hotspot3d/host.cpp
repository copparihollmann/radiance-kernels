#include <stdint.h>
#include <radiance.h>
#include <nightly/host.h>

#include "generated/config.h"
#include "generated/symbols.h"

static uint64_t fnv_word(uint64_t digest, uint32_t word) {
  return (digest ^ word) * 0x100000001B3ULL;
}

static uint32_t ordered_float(uint32_t bits) {
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

static uint32_t ulp_distance(uint32_t lhs, uint32_t rhs) {
  const uint32_t a = ordered_float(lhs);
  const uint32_t b = ordered_float(rhs);
  return a > b ? a - b : b - a;
}

static bool finite_word(uint32_t word) {
  return ((word >> 23) & 0xffu) != 0xffu;
}

static uint32_t expected_owner(uint32_t column) {
  const uint32_t chunks = HOTSPOT3D_COLUMNS / HOTSPOT3D_COLUMNS_PER_LINE;
  const uint32_t base = chunks / NIGHTLY_CLUSTERS;
  const uint32_t extra = chunks % NIGHTLY_CLUSTERS;
  const uint32_t chunk = column / HOTSPOT3D_COLUMNS_PER_LINE;
  uint32_t cursor = 0u;
  for (uint32_t sm = 0u; sm < NIGHTLY_CLUSTERS; ++sm) {
    cursor += base + (sm < extra ? 1u : 0u);
    if (chunk < cursor) return sm;
  }
  return 0xffffffffu;
}

static void fill_guard(uint32_t address, uint32_t pattern) {
  volatile uint32_t* guard = nightly_gpu_ptr<uint32_t>(address);
  for (uint32_t i = 0u; i < 16u; ++i) guard[i] = pattern ^ i;
}

static bool check_guard(uint32_t address, uint32_t pattern) {
  volatile uint32_t* guard = nightly_gpu_ptr<uint32_t>(address);
  for (uint32_t i = 0u; i < 16u; ++i)
    if (guard[i] != (pattern ^ i)) return false;
  return true;
}

static void hold_gpu_in_reset() {
  for (uint32_t cl = 0u; cl < NIGHTLY_CLUSTERS; ++cl)
    nightly_pb_write(cl, NIGHTLY_PB_LAUNCH, 0u);
  asm volatile("fence rw, rw" ::: "memory");
  *reinterpret_cast<volatile uint32_t*>(NIGHTLY_HOST_GPU_RESET) = 1u;
  asm volatile("fence rw, rw" ::: "memory");
  for (volatile uint32_t settle = 0u; settle < 256u; ++settle)
    asm volatile("nop");
}

int main() {
  fill_guard(HOTSPOT3D_GUARD_BEFORE_ADDR, HOTSPOT3D_GUARD_BEFORE_PATTERN);
  fill_guard(HOTSPOT3D_GUARD_MIDDLE_A_ADDR,
             HOTSPOT3D_GUARD_MIDDLE_A_PATTERN);
  fill_guard(HOTSPOT3D_GUARD_MIDDLE_B_ADDR,
             HOTSPOT3D_GUARD_MIDDLE_B_PATTERN);
  fill_guard(HOTSPOT3D_GUARD_AFTER_ADDR, HOTSPOT3D_GUARD_AFTER_PATTERN);
  volatile uint32_t* output = nightly_gpu_ptr<uint32_t>(
      HOTSPOT3D_OUTPUT_ADDR);
  volatile uint32_t* owners = nightly_gpu_ptr<uint32_t>(
      HOTSPOT3D_OWNERS_ADDR);
  for (uint32_t i = 0u; i < HOTSPOT3D_ELEMENTS; ++i)
    output[i] = HOTSPOT3D_OUTPUT_POISON;
  for (uint32_t i = 0u; i < HOTSPOT3D_COLUMNS; ++i)
    owners[i] = HOTSPOT3D_OWNER_POISON;
  asm volatile("fence rw, rw" ::: "memory");

  int completed = 1;
  uint64_t total_cycles = 0u;
  for (uint32_t iteration = 0u; iteration < HOTSPOT3D_ITERATIONS;
       ++iteration) {
    if (iteration != 0u) hold_gpu_in_reset();
    nightly_host_launch(NIGHTLY_CLUSTERS, iteration);
    completed &= nightly_host_wait(NIGHTLY_CLUSTERS);
    total_cycles += nightly_host_report(NIGHTLY_CLUSTERS);
    if (!completed) break;
  }
  asm volatile("fence rw, rw" ::: "memory");

  volatile uint32_t* expected = nightly_gpu_ptr<uint32_t>(
      HOTSPOT3D_EXPECTED_ADDR);
  uint32_t exact = 0u;
  uint32_t bad = 0u;
  uint32_t nonfinite = 0u;
  uint32_t max_ulp = 0u;
  uint64_t output_digest = 0xCBF29CE484222325ULL;
  uint64_t expected_digest = 0xCBF29CE484222325ULL;
  for (uint32_t i = 0u; i < HOTSPOT3D_ELEMENTS; ++i) {
    const uint32_t actual = output[i];
    const uint32_t reference = expected[i];
    output_digest = fnv_word(output_digest, actual);
    expected_digest = fnv_word(expected_digest, reference);
    exact += actual == reference;
    if (!finite_word(actual)) {
      ++nonfinite;
      ++bad;
    } else {
      const uint32_t ulp = ulp_distance(actual, reference);
      if (ulp > max_ulp) max_ulp = ulp;
      bad += ulp > HOTSPOT3D_MAX_ULP;
    }
  }

  uint32_t owner_bad = 0u;
  uint64_t owner_digest = 0xCBF29CE484222325ULL;
  for (uint32_t column = 0u; column < HOTSPOT3D_COLUMNS; ++column) {
    const uint32_t actual = owners[column];
    owner_digest = fnv_word(owner_digest, actual);
    owner_bad += actual != expected_owner(column);
  }
  const bool guards_ok =
      check_guard(HOTSPOT3D_GUARD_BEFORE_ADDR,
                  HOTSPOT3D_GUARD_BEFORE_PATTERN) &&
      check_guard(HOTSPOT3D_GUARD_MIDDLE_A_ADDR,
                  HOTSPOT3D_GUARD_MIDDLE_A_PATTERN) &&
      check_guard(HOTSPOT3D_GUARD_MIDDLE_B_ADDR,
                  HOTSPOT3D_GUARD_MIDDLE_B_PATTERN) &&
      check_guard(HOTSPOT3D_GUARD_AFTER_ADDR,
                  HOTSPOT3D_GUARD_AFTER_PATTERN);
  const bool expected_digest_ok =
      expected_digest == HOTSPOT3D_EXPECTED_DIGEST;
  const bool owner_digest_ok = owner_digest == HOTSPOT3D_OWNER_DIGEST;

  nh_puts("HOTSPOT3D_CHECK n="); nh_putu(HOTSPOT3D_ELEMENTS);
  nh_puts(" exact="); nh_putu(exact);
  nh_puts(" bad="); nh_putu(bad);
  nh_puts(" max_ulp="); nh_putu(max_ulp);
  nh_puts(" nonfinite="); nh_putu(nonfinite);
  nh_puts(" owner_bad="); nh_putu(owner_bad);
  nh_puts(" guards="); nh_putu(guards_ok);
  nh_puts(" output_digest="); nh_putx(output_digest, 16);
  nh_puts(" expected_digest_ok="); nh_putu(expected_digest_ok);
  nh_puts(" owner_digest="); nh_putx(owner_digest, 16);
  nh_puts(" owner_digest_ok="); nh_putu(owner_digest_ok);
  nh_puts(" total_kernel_cycles="); nh_putu(total_cycles);
  nh_puts("\n");
  const bool passed = completed && bad == 0u && nonfinite == 0u &&
      owner_bad == 0u && guards_ok && expected_digest_ok && owner_digest_ok;
  nh_puts(passed ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(passed ? 0 : 1);
}
