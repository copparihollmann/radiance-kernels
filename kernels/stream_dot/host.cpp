#include <stdint.h>
#include <radiance.h>
#include <rad_host.h>

#include "generated/config.h"
#include "generated/symbols.h"
#include "softfloat_positive_f32.h"
#include <nightly/host.h>

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
  const uint32_t kernel_cycles = nightly_host_report(NIGHTLY_CLUSTERS);
  // Compute axis, and this kernel really can distinguish executed from useful.
  //
  // useful   = N multiplies + (N-1) reduction adds: the BabelStream dot product.
  // executed = the device accumulates with N multiplies and N adds (every thread's first
  //            `sum += a[i]*b[i]` is a real add against the zero-initialised accumulator),
  //            then each threadblock runs the full shared-memory tree of
  //            (threads_per_threadblock - 1) adds whether or not it was given any elements.
  //            threads_per_threadblock = Muon cores/cluster x NIGHTLY_OCC warps x lanes.
  // The chunk loop's `end = min(begin + chunk, N)` clamp means every element is touched
  // exactly once, so the whole executed/useful gap is the reduction tree, not padding.
  // The final cross-SM combination is HPC_SM_COUNT-1 adds done on the host in software
  // float, not on the device, and is reported separately rather than folded in.
  {
    const uint64_t n = (uint64_t)STREAM_DOT_ELEMENTS;
    const uint64_t threads_per_block = NIGHTLY_MUON_CORES_PER_CLUSTER *
        (uint64_t)NIGHTLY_OCC * NIGHTLY_SIMT_LANES_PER_CORE;
    const uint64_t useful_ops = n + (n - 1ull);
    const uint64_t device_mul = n;
    const uint64_t device_add = n + (uint64_t)NIGHTLY_CLUSTERS * (threads_per_block - 1ull);
    const uint64_t executed_ops = device_mul + device_add;
    nh_puts("dot_threads_per_threadblock="); nh_putu(threads_per_block); nh_puts("\n");
    nh_puts("dot_device_fp32_multiplies="); nh_putu(device_mul); nh_puts("\n");
    nh_puts("dot_device_fp32_adds="); nh_putu(device_add); nh_puts("\n");
    nh_puts("dot_host_software_float_adds="); nh_putu((uint64_t)HPC_SM_COUNT - 1ull);
    nh_puts("\n");
    nightly_report_compute_scalar("flop", useful_ops, executed_ops, 1,
                                  NIGHTLY_CLUSTERS, kernel_cycles);
  }
  asm volatile("fence rw, rw" ::: "memory");

  // One partial per SM, one 64-byte line per partial: SM n's word is at
  // element n * STREAM_DOT_PARTIAL_STRIDE_FLOATS, not element n.
  volatile uint32_t* partial = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(STREAM_DOT_PARTIAL_ADDR));
  static const uint32_t expected[HPC_SM_COUNT] = STREAM_DOT_EXPECTED_WORDS;
  bool partials_ok = true;
  uint64_t digest = 0xCBF29CE484222325ULL;
  // BabelStream's CUDA backend returns one partial per device block and the
  // host combines those partials in block order.  Preserve that final stage
  // explicitly and gate the returned scalar, not only the intermediate words.
  uint32_t final_word = 0u;
  bool final_inputs_valid = true;
  for (uint32_t sm = 0; sm < HPC_SM_COUNT; ++sm) {
    const uint32_t word = partial[sm * STREAM_DOT_PARTIAL_STRIDE_FLOATS];
    partials_ok &= word == expected[sm];
    final_word = stream_dot_add_positive_finite_f32_rne(
        final_word, word, &final_inputs_valid);
    digest ^= word;
    digest *= 0x100000001B3ULL;
  }
  const bool final_ok = final_inputs_valid &&
      final_word == STREAM_DOT_EXPECTED_FINAL_WORD;
  const bool passed = completed && guards_intact() && partials_ok &&
      digest == STREAM_DOT_EXPECTED_DIGEST && final_ok;
  if (!passed) {
    rad_puts("STREAM_DOT_DIAG completed="); rad_putu(completed);
    rad_puts(" guards="); rad_putu(guards_intact());
    rad_puts(" digest=");
    rad_putx(static_cast<uint32_t>(digest >> 32));
    rad_putx(static_cast<uint32_t>(digest));
    rad_puts(" final="); rad_putx(final_word);
    rad_puts(" final_expected="); rad_putx(STREAM_DOT_EXPECTED_FINAL_WORD);
    rad_putc('\n');
    for (uint32_t sm = 0; sm < HPC_SM_COUNT; ++sm) {
      rad_puts("STREAM_DOT_DIAG sm="); rad_putu(sm);
      rad_puts(" actual=");
      rad_putx(partial[sm * STREAM_DOT_PARTIAL_STRIDE_FLOATS]);
      rad_puts(" expected="); rad_putx(expected[sm]);
      rad_putc('\n');
    }
    rad_flush();
  }
  nh_puts(passed ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(passed ? 0 : 1);
}
