#include <stdint.h>
#include <radiance.h>
#include <nightly/host.h>

#include "generated/config.h"
#include "generated/symbols.h"
#include "wire_format.hpp"

static uint64_t fnv_word(uint64_t digest, uint32_t word) {
  digest ^= word;
  return digest * 0x100000001B3ULL;
}

static uint32_t expected_owner(uint32_t index, uint32_t total) {
  const uint32_t chunks =
      (total + BTREE_OUTPUT_RECORDS_PER_LINE - 1u) /
      BTREE_OUTPUT_RECORDS_PER_LINE;
  const uint32_t base = chunks / NIGHTLY_CLUSTERS;
  const uint32_t extra = chunks % NIGHTLY_CLUSTERS;
  const uint32_t chunk = index / BTREE_OUTPUT_RECORDS_PER_LINE;
  uint32_t cursor = 0u;
  for (uint32_t sm = 0u; sm < NIGHTLY_CLUSTERS; ++sm) {
    cursor += base + (sm < extra ? 1u : 0u);
    if (chunk < cursor) return sm;
  }
  return 0xffffffffu;
}

static bool check_guard(uint32_t address, uint32_t pattern) {
  volatile uint32_t* guard = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(address));
  for (uint32_t i = 0u; i < 16u; ++i)
    if (guard[i] != (pattern ^ i)) return false;
  return true;
}

static void fill_guard(uint32_t address, uint32_t pattern) {
  volatile uint32_t* guard = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(address));
  for (uint32_t i = 0u; i < 16u; ++i) guard[i] = pattern ^ i;
}

int main() {
  fill_guard(BTREE_GUARD_BEFORE_ADDR, 0xB7EE0000u);
  fill_guard(BTREE_GUARD_MIDDLE_ADDR, 0xB7DD0000u);
  fill_guard(BTREE_GUARD_AFTER_ADDR, 0xB7AA0000u);

  volatile uint32_t* point_words = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(BTREE_POINT_OUTPUTS_ADDR));
  volatile uint32_t* range_words = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(BTREE_RANGE_OUTPUTS_ADDR));
  for (uint32_t i = 0u; i < BTREE_POINT_QUERY_COUNT * 4u; ++i)
    point_words[i] = 0xeeeeeeeeu;
  for (uint32_t i = 0u; i < BTREE_RANGE_QUERY_COUNT * 4u; ++i)
    range_words[i] = 0xeeeeeeeeu;
  asm volatile("fence rw, rw" ::: "memory");

  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int completed = nightly_host_wait(NIGHTLY_CLUSTERS);
  nightly_host_report(NIGHTLY_CLUSTERS);
  asm volatile("fence rw, rw" ::: "memory");

  volatile int32_t* point_expected = reinterpret_cast<volatile int32_t*>(
      rad_device_to_host_address(BTREE_POINT_EXPECTED_ADDR));
  volatile int32_t* range_expected = reinterpret_cast<volatile int32_t*>(
      rad_device_to_host_address(BTREE_RANGE_EXPECTED_ADDR));
  volatile BTreePointOutput* points =
      reinterpret_cast<volatile BTreePointOutput*>(point_words);
  volatile BTreeRangeOutput* ranges =
      reinterpret_cast<volatile BTreeRangeOutput*>(range_words);

  uint32_t point_bad = 0u;
  uint32_t range_bad = 0u;
  uint64_t point_digest = 0xCBF29CE484222325ULL;
  uint64_t range_digest = 0xCBF29CE484222325ULL;
  for (uint32_t i = 0u; i < BTREE_POINT_QUERY_COUNT; ++i) {
    const uint32_t expected_sm = expected_owner(i, BTREE_POINT_QUERY_COUNT);
    point_bad += points[i].value != point_expected[i];
    point_bad += points[i].query_index != i;
    point_bad += points[i].owner_sm != expected_sm;
    point_bad += points[i].tag != (BTREE_POINT_TAG ^ i);
    point_digest = fnv_word(point_digest, (uint32_t)points[i].value);
    point_digest = fnv_word(point_digest, points[i].query_index);
    point_digest = fnv_word(point_digest, points[i].owner_sm);
    point_digest = fnv_word(point_digest, points[i].tag);
  }
  for (uint32_t i = 0u; i < BTREE_RANGE_QUERY_COUNT; ++i) {
    const uint32_t expected_sm = expected_owner(i, BTREE_RANGE_QUERY_COUNT);
    range_bad += ranges[i].start != range_expected[2u * i];
    range_bad += ranges[i].length != range_expected[2u * i + 1u];
    range_bad += ranges[i].owner_sm != expected_sm;
    range_bad += ranges[i].tag != (BTREE_RANGE_TAG ^ i);
    range_digest = fnv_word(range_digest, (uint32_t)ranges[i].start);
    range_digest = fnv_word(range_digest, (uint32_t)ranges[i].length);
    range_digest = fnv_word(range_digest, ranges[i].owner_sm);
    range_digest = fnv_word(range_digest, ranges[i].tag);
  }

  const bool guards_ok =
      check_guard(BTREE_GUARD_BEFORE_ADDR, 0xB7EE0000u) &&
      check_guard(BTREE_GUARD_MIDDLE_ADDR, 0xB7DD0000u) &&
      check_guard(BTREE_GUARD_AFTER_ADDR, 0xB7AA0000u);
  const bool point_digest_ok = point_digest == BTREE_POINT_OUTPUT_DIGEST;
  const bool range_digest_ok = range_digest == BTREE_RANGE_OUTPUT_DIGEST;
  nh_puts("BTREE_CHECK point_n="); nh_putu(BTREE_POINT_QUERY_COUNT);
  nh_puts(" point_bad="); nh_putu(point_bad);
  nh_puts(" point_digest="); nh_putx(point_digest, 16);
  nh_puts(" point_digest_ok="); nh_putu(point_digest_ok);
  nh_puts(" range_n="); nh_putu(BTREE_RANGE_QUERY_COUNT);
  nh_puts(" range_bad="); nh_putu(range_bad);
  nh_puts(" range_digest="); nh_putx(range_digest, 16);
  nh_puts(" range_digest_ok="); nh_putu(range_digest_ok);
  nh_puts(" guards="); nh_putu(guards_ok);
  nh_puts("\n");
  const bool passed = completed && point_bad == 0u && range_bad == 0u &&
                      point_digest_ok && range_digest_ok && guards_ok;
  nh_puts(passed ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(passed ? 0 : 1);
}
