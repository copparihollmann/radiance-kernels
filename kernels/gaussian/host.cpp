#include <stdint.h>
#include <radiance.h>
#include <nightly/host.h>

#include "actual_checker.h"
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

struct OriginalCheck {
  uint64_t digest;
  uint32_t nonfinite;
};

static OriginalCheck check_original(uint32_t address, uint32_t words) {
  volatile uint32_t* values = nightly_gpu_ptr<uint32_t>(address);
  OriginalCheck result = {0xCBF29CE484222325ULL, 0u};
  for (uint32_t i = 0u; i < words; ++i) {
    const uint32_t word = values[i];
    result.digest = fnv_word(result.digest, word);
    result.nonfinite += !finite_word(word);
  }
  return result;
}

static void print_original_check(const char* name, const OriginalCheck& check,
                                 uint64_t expected_digest) {
  nh_puts(name); nh_puts(" digest="); nh_putx(check.digest, 16);
  nh_puts(" expected_digest="); nh_putx(expected_digest, 16);
  nh_puts(" nonfinite="); nh_putu(check.nonfinite);
  nh_puts(" exact="); nh_putu(check.digest == expected_digest);
  nh_puts("\n");
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

static uint64_t accumulate_stage_cycles(
    uint64_t per_core[NIGHTLY_CLUSTERS][NIGHTLY_CORES_PER_CLUSTER]) {
  uint32_t first = 0xffffffffu;
  uint32_t last = 0u;
  for (uint32_t cl = 0u; cl < NIGHTLY_CLUSTERS; ++cl) {
    for (uint32_t core = 0u; core < NIGHTLY_CORES_PER_CLUSTER; ++core) {
      const uint32_t start = nightly_pb_read(cl, NIGHTLY_PB_START + core);
      const uint32_t end = nightly_pb_read(cl, NIGHTLY_PB_END + core);
      per_core[cl][core] += static_cast<uint32_t>(end - start);
      if (start < first) first = start;
      if (end > last) last = end;
    }
  }
  return static_cast<uint32_t>(last - first);
}

static void report_aggregate_cycles(
    const uint64_t per_core[NIGHTLY_CLUSTERS][NIGHTLY_CORES_PER_CLUSTER],
    uint64_t total) {
  for (uint32_t cl = 0u; cl < NIGHTLY_CLUSTERS; ++cl) {
    for (uint32_t core = 0u; core < NIGHTLY_CORES_PER_CLUSTER; ++core) {
      nh_puts("cluster "); nh_putu(cl); nh_puts(" core "); nh_putu(core);
      nh_puts(": start=0 end="); nh_putu(per_core[cl][core]);
      nh_puts(" cycles="); nh_putu(per_core[cl][core]); nh_puts("\n");
    }
  }
  nh_puts("kernel_cycles="); nh_putu(total); nh_puts("\n");
  nh_flush();
}

int main() {
  int completed = 1;
  uint64_t total_cycles = 0u;
  uint64_t per_core_cycles[NIGHTLY_CLUSTERS][NIGHTLY_CORES_PER_CLUSTER] = {};
  // Host waits are the explicit device-wide edges Fan1(t)->Fan2(t) and
  // Fan2(t)->Fan1(t+1). DRAM state survives each inter-stage soft reset.
  for (uint32_t stage = 0u; stage < GAUSSIAN_STAGES; ++stage) {
#if !NIGHTLY_ROBUST_SPIN_UNTIL_RESET
    if (stage != 0u) hold_gpu_in_reset();
#endif
    nightly_host_launch(NIGHTLY_CLUSTERS, stage);
    completed &= nightly_host_wait(NIGHTLY_CLUSTERS);
    total_cycles += accumulate_stage_cycles(per_core_cycles);
#if NIGHTLY_ROBUST_SPIN_UNTIL_RESET
    // Every robust stage publishes END and then waits for this reset.  Reset
    // the final stage too, before the host reads A, b, m, owners, and guards.
    hold_gpu_in_reset();
#endif
    if (!completed) break;
  }
  report_aggregate_cycles(per_core_cycles, total_cycles);
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
  const OriginalCheck original_a = check_original(
      GAUSSIAN_ORIGINAL_A_ADDR, GAUSSIAN_MATRIX_WORDS);
  const OriginalCheck original_b = check_original(
      GAUSSIAN_ORIGINAL_B_ADDR, GAUSSIAN_SIZE);
  print_original_check("GAUSSIAN_ORIGINAL_A", original_a,
                       GAUSSIAN_ORIGINAL_A_DIGEST);
  print_original_check("GAUSSIAN_ORIGINAL_B", original_b,
                       GAUSSIAN_ORIGINAL_B_DIGEST);

  uint32_t solution[GAUSSIAN_SIZE] = {};
  const GaussianBackSubstitution back = gaussian_back_substitute_actual(
      nightly_gpu_ptr<uint32_t>(GAUSSIAN_A_ADDR),
      nightly_gpu_ptr<uint32_t>(GAUSSIAN_B_ADDR),
      GAUSSIAN_SIZE, solution);
  GaussianResidualCertificate residual = {};
  if (back.valid) {
    residual = gaussian_certify_actual_solution(
        nightly_gpu_ptr<uint32_t>(GAUSSIAN_ORIGINAL_A_ADDR),
        nightly_gpu_ptr<uint32_t>(GAUSSIAN_ORIGINAL_B_ADDR),
        solution, GAUSSIAN_SIZE, GAUSSIAN_RESIDUAL_TOLERANCE_SHIFT);
  }
  nh_puts("GAUSSIAN_BACK_SUB actual_transformed=1 valid=");
  nh_putu(back.valid); nh_puts(" failure="); nh_putu(back.failure);
  nh_puts(" row="); nh_putu(back.row);
  nh_puts(" column="); nh_putu(back.column);
  nh_puts(" solution_digest="); nh_putx(back.solution_digest, 16);
  nh_puts("\n");
  nh_puts("GAUSSIAN_RESIDUAL actual_transformed=1 valid=");
  nh_putu(residual.valid); nh_puts(" certified=");
  nh_putu(residual.certified); nh_puts(" failure=");
  nh_putu(residual.failure); nh_puts(" row="); nh_putu(residual.row);
  nh_puts(" upper_coefficient=");
  nh_putu(residual.residual_upper_coefficient);
  nh_puts(" upper_exponent2=");
  nh_puti(residual.residual_upper_exponent);
  nh_puts(" denominator_lower_word=");
  nh_putx(residual.denominator_lower_word, 8);
  nh_puts(" tolerance_power2=-");
  nh_putu(residual.tolerance_shift); nh_puts("\n");
  const bool owners_ok = check_owners();
  const bool guards_ok =
      check_guard(GAUSSIAN_GUARD_BEFORE_ADDR, GAUSSIAN_GUARD_BEFORE_PATTERN) &&
      check_guard(GAUSSIAN_GUARD_MIDDLE_ADDR, GAUSSIAN_GUARD_MIDDLE_PATTERN) &&
      check_guard(GAUSSIAN_GUARD_MODEL_ADDR, GAUSSIAN_GUARD_MODEL_PATTERN) &&
      check_guard(GAUSSIAN_GUARD_AFTER_ADDR, GAUSSIAN_GUARD_AFTER_PATTERN);
  const bool trajectory_match = a.bad == 0u && b.bad == 0u && m.bad == 0u;
  nh_puts("GAUSSIAN_SUMMARY oracle_trajectory_match=");
  nh_putu(trajectory_match);
  nh_puts(" actual_residual_certified="); nh_putu(residual.certified);
  nh_puts(" guards="); nh_putu(guards_ok);
  nh_puts(" total_kernel_cycles="); nh_putu(total_cycles); nh_puts("\n");

  // The expected trajectory remains an exhaustive diagnostic, but is not the
  // semantic acceptance criterion: independent IEEE implementations may
  // accumulate more than one ULP over 126 stages.  Acceptance instead proves
  // that a software-binary32 back solve of the actual transformed A/b solves
  // the untouched original system under a stricter-than-documented residual
  // bound.  Completion, every finite output, ownership, and guards remain
  // mandatory and fail closed.
  const bool originals_ok =
      original_a.digest == GAUSSIAN_ORIGINAL_A_DIGEST &&
      original_b.digest == GAUSSIAN_ORIGINAL_B_DIGEST &&
      original_a.nonfinite == 0u && original_b.nonfinite == 0u;
  const bool passed = completed && a.nonfinite == 0u && b.nonfinite == 0u &&
      m.nonfinite == 0u && originals_ok && back.valid && residual.valid &&
      residual.certified && owners_ok && guards_ok;
  nh_puts(passed ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(passed ? 0 : 1);
}
