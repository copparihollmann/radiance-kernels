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
  nightly_host_report(NIGHTLY_CLUSTERS);
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
