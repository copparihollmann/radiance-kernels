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
  const uint32_t a = ordered_float(lhs), b = ordered_float(rhs);
  return a > b ? a - b : b - a;
}
static bool finite_word(uint32_t word) {
  return ((word >> 23) & 0xffu) != 0xffu;
}

struct Check {
  uint32_t exact, bad, nonfinite, max_ulp;
  uint64_t actual_digest, expected_digest;
};

static Check check_floats(uint32_t actual_addr, uint32_t expected_addr,
                          uint32_t words) {
  volatile uint32_t* actual = nightly_gpu_ptr<uint32_t>(actual_addr);
  volatile uint32_t* expected = nightly_gpu_ptr<uint32_t>(expected_addr);
  Check c = {0u, 0u, 0u, 0u, 0xCBF29CE484222325ULL,
             0xCBF29CE484222325ULL};
  for (uint32_t i = 0u; i < words; ++i) {
    const uint32_t got = actual[i], want = expected[i];
    c.actual_digest = fnv_word(c.actual_digest, got);
    c.expected_digest = fnv_word(c.expected_digest, want);
    c.exact += got == want;
    if (!finite_word(got)) { ++c.nonfinite; ++c.bad; continue; }
    const uint32_t ulp = ulp_distance(got, want);
    if (ulp > c.max_ulp) c.max_ulp = ulp;
    c.bad += ulp > GAUSSIAN_MAX_ULP;
  }
  return c;
}

static void print_check(const char* name, const Check& c, uint32_t words) {
  nh_puts(name); nh_puts(" n="); nh_putu(words);
  nh_puts(" exact="); nh_putu(c.exact); nh_puts(" bad="); nh_putu(c.bad);
  nh_puts(" max_ulp="); nh_putu(c.max_ulp);
  nh_puts(" nonfinite="); nh_putu(c.nonfinite);
  nh_puts(" digest="); nh_putx(c.actual_digest, 16);
  nh_puts(" expected_digest="); nh_putx(c.expected_digest, 16);
  nh_puts("\n");
}

static uint32_t expected_owner(uint32_t stage, uint32_t row) {
  const uint32_t t = stage / 2u;
  if (row <= t) return GAUSSIAN_OWNER_POISON;
  return (row / GAUSSIAN_ROWS_PER_LINE) % NIGHTLY_CLUSTERS;
}

static bool check_owners() {
  volatile uint32_t* owners = nightly_gpu_ptr<uint32_t>(GAUSSIAN_OWNERS_ADDR);
  uint32_t bad = 0u;
  uint64_t digest = 0xCBF29CE484222325ULL;
  for (uint32_t stage = 0u; stage < GAUSSIAN_STAGES; ++stage) {
    for (uint32_t row = 0u; row < GAUSSIAN_OWNER_STRIDE; ++row) {
      const uint32_t want = row < GAUSSIAN_SIZE
          ? expected_owner(stage, row) : GAUSSIAN_OWNER_POISON;
      const uint32_t got = owners[stage * GAUSSIAN_OWNER_STRIDE + row];
      digest = fnv_word(digest, got);
      bad += got != want;
    }
  }
  nh_puts("GAUSSIAN_OWNERS n=");
  nh_putu(GAUSSIAN_STAGES * GAUSSIAN_OWNER_STRIDE);
  nh_puts(" bad="); nh_putu(bad);
  nh_puts(" digest="); nh_putx(digest, 16); nh_puts("\n");
  return bad == 0u && digest == GAUSSIAN_OWNER_DIGEST;
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
  int completed = 1;
  uint64_t total_cycles = 0u;
  // Host waits are the explicit device-wide edges Fan1(t)->Fan2(t) and
  // Fan2(t)->Fan1(t+1). DRAM state survives each inter-stage soft reset.
  for (uint32_t stage = 0u; stage < GAUSSIAN_STAGES; ++stage) {
    if (stage != 0u) hold_gpu_in_reset();
    nightly_host_launch(NIGHTLY_CLUSTERS, stage);
    completed &= nightly_host_wait(NIGHTLY_CLUSTERS);
    total_cycles += nightly_host_report(NIGHTLY_CLUSTERS);
    if (!completed) break;
  }
  asm volatile("fence rw, rw" ::: "memory");

  const Check a = check_floats(
      GAUSSIAN_A_ADDR, GAUSSIAN_EXPECTED_A_ADDR, GAUSSIAN_MATRIX_WORDS);
  const Check b = check_floats(
      GAUSSIAN_B_ADDR, GAUSSIAN_EXPECTED_B_ADDR, GAUSSIAN_SIZE);
  const Check m = check_floats(
      GAUSSIAN_M_ADDR, GAUSSIAN_EXPECTED_M_ADDR, GAUSSIAN_MATRIX_WORDS);
  print_check("GAUSSIAN_A", a, GAUSSIAN_MATRIX_WORDS);
  print_check("GAUSSIAN_B", b, GAUSSIAN_SIZE);
  print_check("GAUSSIAN_M", m, GAUSSIAN_MATRIX_WORDS);
  const bool owners_ok = check_owners();
  const bool guards_ok =
      check_guard(GAUSSIAN_GUARD_BEFORE_ADDR, GAUSSIAN_GUARD_BEFORE_PATTERN) &&
      check_guard(GAUSSIAN_GUARD_MIDDLE_ADDR, GAUSSIAN_GUARD_MIDDLE_PATTERN) &&
      check_guard(GAUSSIAN_GUARD_MODEL_ADDR, GAUSSIAN_GUARD_MODEL_PATTERN) &&
      check_guard(GAUSSIAN_GUARD_AFTER_ADDR, GAUSSIAN_GUARD_AFTER_PATTERN);
  nh_puts("GAUSSIAN_RESIDUAL relative_backward_error_e18=");
  nh_putu(GAUSSIAN_RESIDUAL_E18);
  nh_puts(" tolerance_e18="); nh_putu(GAUSSIAN_RESIDUAL_TOLERANCE_E18);
  nh_puts(" oracle_within_tolerance=");
  nh_putu(GAUSSIAN_RESIDUAL_E18 <= GAUSSIAN_RESIDUAL_TOLERANCE_E18);
  nh_puts(" guards="); nh_putu(guards_ok);
  nh_puts(" total_kernel_cycles="); nh_putu(total_cycles); nh_puts("\n");

  const bool passed = completed && a.bad == 0u && b.bad == 0u &&
      m.bad == 0u && a.nonfinite == 0u && b.nonfinite == 0u &&
      m.nonfinite == 0u && owners_ok && guards_ok &&
      GAUSSIAN_RESIDUAL_E18 <= GAUSSIAN_RESIDUAL_TOLERANCE_E18;
  nh_puts(passed ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(passed ? 0 : 1);
}
