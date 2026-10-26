#include <stdint.h>
#include <radiance.h>
#include <rad_host.h>

#include "generated/config.h"
#include "generated/symbols.h"
#include <nightly/host.h>

union FloatWord {
  uint32_t word;
  float value;
};

static bool guards_intact() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_DOT_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_DOT_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16; ++i) {
    if (before[i] != (0xA5A50000u ^ i) ||
        after[i] != (0x5A5A0000u ^ i)) return false;
  }
  return true;
}

int main() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_DOT_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_DOT_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16; ++i) {
    before[i] = 0xA5A50000u ^ i;
    after[i] = 0x5A5A0000u ^ i;
  }
  asm volatile("fence rw, rw" ::: "memory");
  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int completed = nightly_host_wait(NIGHTLY_CLUSTERS);
  nightly_host_report(NIGHTLY_CLUSTERS);
  asm volatile("fence rw, rw" ::: "memory");

  volatile uint32_t* partial = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_DOT_PARTIAL_ADDR));
  static const uint32_t expected[HPC_SM_COUNT] = STREAM_DOT_EXPECTED_WORDS;
  bool partials_ok = true;
  uint64_t digest = 0xCBF29CE484222325ULL;
  // BabelStream's CUDA backend returns one partial per device block and the
  // host combines those partials in block order.  Preserve that final stage
  // explicitly and gate the returned scalar, not only the intermediate words.
  volatile float final_sum = 0.0f;
  for (uint32_t sm = 0; sm < HPC_SM_COUNT; ++sm) {
    const uint32_t word = partial[sm];
    partials_ok &= word == expected[sm];
    FloatWord decoded;
    decoded.word = word;
    final_sum = final_sum + decoded.value;
    digest ^= word;
    digest *= 0x100000001B3ULL;
  }
  FloatWord final_word;
  final_word.value = final_sum;
  const bool final_ok = final_word.word == STREAM_DOT_EXPECTED_FINAL_WORD;
  const bool passed = completed && guards_intact() && partials_ok &&
      digest == STREAM_DOT_EXPECTED_DIGEST && final_ok;
  if (!passed) {
    rad_puts("STREAM_DOT_DIAG completed="); rad_putu(completed);
    rad_puts(" guards="); rad_putu(guards_intact());
    rad_puts(" digest=");
    rad_putx(static_cast<uint32_t>(digest >> 32));
    rad_putx(static_cast<uint32_t>(digest));
    rad_puts(" final="); rad_putx(final_word.word);
    rad_puts(" final_expected="); rad_putx(STREAM_DOT_EXPECTED_FINAL_WORD);
    rad_putc('\n');
    for (uint32_t sm = 0; sm < HPC_SM_COUNT; ++sm) {
      rad_puts("STREAM_DOT_DIAG sm="); rad_putu(sm);
      rad_puts(" actual="); rad_putx(partial[sm]);
      rad_puts(" expected="); rad_putx(expected[sm]);
      rad_putc('\n');
    }
    rad_flush();
  }
  nh_puts(passed ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(passed ? 0 : 1);
}
