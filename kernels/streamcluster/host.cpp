#include <stdint.h>

#include <nightly/host.h>
#include <radiance.h>

#include "generated/config.h"
#include "generated/symbols.h"

static float as_float(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } value = {bits};
  return value.f;
}

static float abs_float(float value) { return value < 0.0f ? -value : value; }

static bool finite_word(uint32_t bits) {
  return ((bits >> 23u) & 0xffu) != 0xffu;
}

static uint64_t digest_words(volatile uint32_t* values, uint32_t count) {
  uint64_t digest = 0xCBF29CE484222325ULL;
  for (uint32_t i = 0; i < count; ++i) {
    digest ^= values[i];
    digest *= 0x100000001B3ULL;
  }
  return digest;
}

static uint64_t digest_bytes_logical(volatile uint8_t* storage) {
  static constexpr uint32_t boundaries[HPC_SM_COUNT + 1u] =
      STREAMCLUSTER_SM_POINT_BOUNDARIES;
  uint64_t digest = 0xCBF29CE484222325ULL;
  for (uint32_t sm = 0; sm < HPC_SM_COUNT; ++sm) {
    const uint32_t count = boundaries[sm + 1u] - boundaries[sm];
    const uint32_t base = sm * STREAMCLUSTER_SWITCH_SEGMENT_STRIDE;
    for (uint32_t local = 0; local < count; ++local) {
      digest ^= storage[base + local];
      digest *= 0x100000001B3ULL;
    }
  }
  return digest;
}

static bool guards_intact() {
  volatile uint32_t* work_before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAMCLUSTER_WORK_GUARD_BEFORE_ADDR));
  volatile uint32_t* work_after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAMCLUSTER_WORK_GUARD_AFTER_ADDR));
  volatile uint32_t* switch_before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAMCLUSTER_SWITCH_GUARD_BEFORE_ADDR));
  volatile uint32_t* switch_after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAMCLUSTER_SWITCH_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16u; ++i) {
    if (work_before[i] != (0xA5C10000u ^ i) ||
        work_after[i] != (0x5A3E0000u ^ i) ||
        switch_before[i] != (0xC35A0000u ^ i) ||
        switch_after[i] != (0x3CA50000u ^ i))
      return false;
  }
  return true;
}

int main() {
  volatile uint32_t* work_before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAMCLUSTER_WORK_GUARD_BEFORE_ADDR));
  volatile uint32_t* work = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAMCLUSTER_WORK_OUTPUT_ADDR));
  volatile uint32_t* work_after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAMCLUSTER_WORK_GUARD_AFTER_ADDR));
  volatile uint32_t* switch_before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAMCLUSTER_SWITCH_GUARD_BEFORE_ADDR));
  volatile uint8_t* switches = reinterpret_cast<volatile uint8_t*>(
      rad_device_to_host_address(STREAMCLUSTER_SWITCH_OUTPUT_ADDR));
  volatile uint32_t* switch_after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAMCLUSTER_SWITCH_GUARD_AFTER_ADDR));

  for (uint32_t i = 0; i < 16u; ++i) {
    work_before[i] = 0xA5C10000u ^ i;
    work_after[i] = 0x5A3E0000u ^ i;
    switch_before[i] = 0xC35A0000u ^ i;
    switch_after[i] = 0x3CA50000u ^ i;
  }
  for (uint32_t i = 0; i < STREAMCLUSTER_WORK_FLOATS; ++i)
    work[i] = 0x7fc00000u;
  for (uint32_t i = 0; i < STREAMCLUSTER_SWITCH_STORAGE_BYTES; ++i)
    switches[i] = 0xA5u;
  asm volatile("fence rw, rw" ::: "memory");

  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int completed = nightly_host_wait(NIGHTLY_CLUSTERS);
  nightly_host_report(NIGHTLY_CLUSTERS);
  asm volatile("fence rw, rw" ::: "memory");

  volatile uint32_t* expected_work = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAMCLUSTER_WORK_EXPECTED_ADDR));
  volatile uint8_t* expected_switch = reinterpret_cast<volatile uint8_t*>(
      rad_device_to_host_address(STREAMCLUSTER_SWITCH_EXPECTED_ADDR));
  const float relative_tolerance = as_float(STREAMCLUSTER_REL_TOL_BITS);
  uint32_t work_exact = 0u;
  uint32_t work_bad = 0u;
  uint32_t work_nonfinite = 0u;
  for (uint32_t i = 0; i < STREAMCLUSTER_WORK_FLOATS; ++i) {
    const uint32_t actual_word = work[i];
    const uint32_t expected_word = expected_work[i];
    work_exact += actual_word == expected_word;
    if (!finite_word(actual_word)) {
      ++work_nonfinite;
      ++work_bad;
      continue;
    }
    const float actual = as_float(actual_word);
    const float reference = as_float(expected_word);
    const float denominator = abs_float(reference) > 1.0f
                                  ? abs_float(reference)
                                  : 1.0f;
    if (abs_float(actual - reference) / denominator > relative_tolerance)
      ++work_bad;
  }

  static constexpr uint32_t boundaries[HPC_SM_COUNT + 1u] =
      STREAMCLUSTER_SM_POINT_BOUNDARIES;
  uint32_t switch_exact = 0u;
  uint32_t switch_bad = 0u;
  uint32_t switch_padding_bad = 0u;
  for (uint32_t sm = 0; sm < HPC_SM_COUNT; ++sm) {
    const uint32_t begin = boundaries[sm];
    const uint32_t count = boundaries[sm + 1u] - begin;
    const uint32_t base = sm * STREAMCLUSTER_SWITCH_SEGMENT_STRIDE;
    for (uint32_t local = 0; local < count; ++local) {
      const uint8_t actual = switches[base + local];
      const uint8_t reference = expected_switch[begin + local];
      switch_exact += actual == reference;
      switch_bad += actual != reference;
    }
    for (uint32_t local = count;
         local < STREAMCLUSTER_SWITCH_SEGMENT_STRIDE; ++local)
      switch_padding_bad += switches[base + local] != 0xA5u;
  }

  const bool guards_ok = guards_intact();
  const uint64_t actual_work_digest =
      digest_words(work, STREAMCLUSTER_WORK_FLOATS);
  const uint64_t reference_work_digest =
      digest_words(expected_work, STREAMCLUSTER_WORK_FLOATS);
  const uint64_t actual_switch_digest = digest_bytes_logical(switches);
  uint64_t reference_switch_digest = 0xCBF29CE484222325ULL;
  for (uint32_t i = 0; i < STREAMCLUSTER_NUM; ++i) {
    reference_switch_digest ^= expected_switch[i];
    reference_switch_digest *= 0x100000001B3ULL;
  }
  const bool work_digest_ok =
      actual_work_digest == STREAMCLUSTER_WORK_EXPECTED_DIGEST;
  const bool work_reference_ok =
      reference_work_digest == STREAMCLUSTER_WORK_EXPECTED_DIGEST;
  const bool switch_digest_ok =
      actual_switch_digest == STREAMCLUSTER_SWITCH_EXPECTED_DIGEST;
  const bool switch_reference_ok =
      reference_switch_digest == STREAMCLUSTER_SWITCH_EXPECTED_DIGEST;

  nh_puts("STREAMCLUSTER_CHECK work="); nh_putu(STREAMCLUSTER_WORK_FLOATS);
  nh_puts(" work_exact="); nh_putu(work_exact);
  nh_puts(" work_bad="); nh_putu(work_bad);
  nh_puts(" work_nonfinite="); nh_putu(work_nonfinite);
  nh_puts(" switch="); nh_putu(STREAMCLUSTER_NUM);
  nh_puts(" switch_exact="); nh_putu(switch_exact);
  nh_puts(" switch_bad="); nh_putu(switch_bad);
  nh_puts(" switch_padding_bad="); nh_putu(switch_padding_bad);
  nh_puts(" guards="); nh_putu(guards_ok);
  nh_puts(" work_digest="); nh_putx(actual_work_digest, 16);
  nh_puts(" switch_digest="); nh_putx(actual_switch_digest, 16);
  nh_puts("\n");

  const bool passed = completed && guards_ok && work_bad == 0u &&
                      work_nonfinite == 0u && switch_bad == 0u &&
                      switch_padding_bad == 0u && work_digest_ok &&
                      work_reference_ok && switch_digest_ok &&
                      switch_reference_ok;
  nh_puts(passed ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(passed ? 0 : 1);
}
