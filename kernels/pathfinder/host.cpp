#include <stdint.h>
#include <radiance.h>
#include <nightly/host.h>

#include "generated/config.h"
#include "generated/symbols.h"

static uint64_t fnv_word(uint64_t digest, uint32_t word) {
  digest ^= word;
  return digest * 0x100000001B3ULL;
}

static uint32_t expected_owner(uint32_t column) {
  const uint32_t chunks = PATHFINDER_COLS / PATHFINDER_INTS_PER_LINE;
  const uint32_t base = chunks / NIGHTLY_CLUSTERS;
  const uint32_t extra = chunks % NIGHTLY_CLUSTERS;
  const uint32_t chunk = column / PATHFINDER_INTS_PER_LINE;
  uint32_t cursor = 0u;
  for (uint32_t sm = 0u; sm < NIGHTLY_CLUSTERS; ++sm) {
    cursor += base + (sm < extra ? 1u : 0u);
    if (chunk < cursor) return sm;
  }
  return 0xffffffffu;
}

static void fill_guard(uint32_t address, uint32_t pattern) {
  volatile uint32_t* guard = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(address));
  for (uint32_t i = 0u; i < 16u; ++i) guard[i] = pattern ^ i;
}

static bool check_guard(uint32_t address, uint32_t pattern) {
  volatile uint32_t* guard = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(address));
  for (uint32_t i = 0u; i < 16u; ++i)
    if (guard[i] != (pattern ^ i)) return false;
  return true;
}

int main() {
  fill_guard(PATHFINDER_GUARD_BEFORE_ADDR, 0xFAEE0000u);
  fill_guard(PATHFINDER_GUARD_MIDDLE_ADDR, 0xFADD0000u);
  fill_guard(PATHFINDER_GUARD_AFTER_ADDR, 0xFAAA0000u);

  volatile int32_t* output = reinterpret_cast<volatile int32_t*>(
      rad_device_to_host_address(PATHFINDER_OUTPUT_ADDR));
  volatile uint32_t* owners = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(PATHFINDER_OWNERS_ADDR));
  for (uint32_t i = 0u; i < PATHFINDER_COLS; ++i) {
    output[i] = (int32_t)0xeeeeeeeeu;
    owners[i] = 0xffffffffu;
  }
  asm volatile("fence rw, rw" ::: "memory");

  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int completed = nightly_host_wait(NIGHTLY_CLUSTERS);
  const uint32_t kernel_cycles = nightly_host_report(NIGHTLY_CLUSTERS);
  // Compute axis.  Each dynamic-programming cell update is 2 signed compares + 1 add.
  //
  // useful   = the semantic DP: (ROWS-1) * COLS updates, each counted once.
  // executed = what this partitioning actually runs.  Every SM owns a cache-line aligned
  //            column range widened by the remaining row count, so its contracting cone never
  //            needs a neighbour's value; the overlap between adjacent cones is recomputed by
  //            both SMs.  The loop below replays the generator's own recurrence
  //            (kernels/pathfinder/run.py partition_boundaries) and costs ROWS*SMS host
  //            iterations, so the overhead is counted, not predicted.
  {
    const uint64_t rows = (uint64_t)PATHFINDER_ROWS;
    const uint64_t cols = (uint64_t)PATHFINDER_COLS;
    const uint32_t chunks = PATHFINDER_COLS / PATHFINDER_INTS_PER_LINE;
    const uint32_t base = chunks / NIGHTLY_CLUSTERS;
    const uint32_t extra = chunks % NIGHTLY_CLUSTERS;
    uint64_t executed_updates = 0u;
    uint32_t cursor = 0u;
    for (uint32_t sm = 0u; sm < NIGHTLY_CLUSTERS; ++sm) {
      const uint64_t begin = (uint64_t)cursor * PATHFINDER_INTS_PER_LINE;
      cursor += base + (sm < extra ? 1u : 0u);
      const uint64_t end = (uint64_t)cursor * PATHFINDER_INTS_PER_LINE;
      for (uint64_t row = 1u; row < rows; ++row) {
        const uint64_t remaining = rows - 1u - row;
        const uint64_t lo = begin > remaining ? begin - remaining : 0u;
        const uint64_t hi = end + remaining < cols ? end + remaining : cols;
        executed_updates += hi - lo;
      }
    }
    const uint64_t useful_updates = (rows - 1u) * cols;
    const uint64_t ops_per_update = 3u;   // 2 signed compares + 1 add
    nh_puts("pathfinder_useful_updates="); nh_putu(useful_updates); nh_puts("\n");
    nh_puts("pathfinder_executed_updates="); nh_putu(executed_updates); nh_puts("\n");
    nightly_report_compute_scalar("integer-compare-add", useful_updates * ops_per_update,
                                  executed_updates * ops_per_update, 1,
                                  NIGHTLY_CLUSTERS, kernel_cycles);
  }
  asm volatile("fence rw, rw" ::: "memory");

  volatile int32_t* expected = reinterpret_cast<volatile int32_t*>(
      rad_device_to_host_address(PATHFINDER_EXPECTED_ADDR));
  uint32_t value_bad = 0u;
  uint32_t owner_bad = 0u;
  uint64_t value_digest = 0xCBF29CE484222325ULL;
  uint64_t owner_digest = 0xCBF29CE484222325ULL;
  for (uint32_t column = 0u; column < PATHFINDER_COLS; ++column) {
    value_bad += output[column] != expected[column];
    owner_bad += owners[column] != expected_owner(column);
    value_digest = fnv_word(value_digest, (uint32_t)output[column]);
    owner_digest = fnv_word(owner_digest, owners[column]);
  }

  const bool guards_ok =
      check_guard(PATHFINDER_GUARD_BEFORE_ADDR, 0xFAEE0000u) &&
      check_guard(PATHFINDER_GUARD_MIDDLE_ADDR, 0xFADD0000u) &&
      check_guard(PATHFINDER_GUARD_AFTER_ADDR, 0xFAAA0000u);
  const bool value_digest_ok = value_digest == PATHFINDER_OUTPUT_DIGEST;
  const bool owner_digest_ok = owner_digest == PATHFINDER_OWNER_DIGEST;
  nh_puts("PATHFINDER_CHECK rows="); nh_putu(PATHFINDER_ROWS);
  nh_puts(" cols="); nh_putu(PATHFINDER_COLS);
  nh_puts(" value_bad="); nh_putu(value_bad);
  nh_puts(" owner_bad="); nh_putu(owner_bad);
  nh_puts(" value_digest="); nh_putx(value_digest, 16);
  nh_puts(" value_digest_ok="); nh_putu(value_digest_ok);
  nh_puts(" owner_digest="); nh_putx(owner_digest, 16);
  nh_puts(" owner_digest_ok="); nh_putu(owner_digest_ok);
  nh_puts(" guards="); nh_putu(guards_ok);
  nh_puts("\n");
  const bool passed = completed && value_bad == 0u && owner_bad == 0u &&
                      value_digest_ok && owner_digest_ok && guards_ok;
  nh_puts(passed ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(passed ? 0 : 1);
}
