/* Whole-workload publication and completion for profiling-enabled builds. */
#ifndef NIGHTLY_PROFILE_DEVICE_H
#define NIGHTLY_PROFILE_DEVICE_H
#include <mu_intrinsics.h>
#include <nightly/profile.h>

#define NIGHTLY_PROFILE_DONE_SLOT 59u
#define NIGHTLY_PROFILE_CODE_SLOT 58u
#define NIGHTLY_PROFILE_DONE 0x50444f4eu

static inline void nightly_profile_wait_launch() {
  uint32_t value;
  const uint32_t address = 0x80000u;
  do {
    asm volatile("lw.shared %0, 0(%1)" : "=r"(value) : "r"(address) : "memory");
  } while (value != 0x6e1a0c11u);
}

/* GEMM calls its end hook inside a callback, before its verification loop.
 * The begin snapshot therefore lives in uncached, per-core printBuf slots.
 * These begin stores are instrumentation inside its measured interval. */
static inline void nightly_profile_begin_mailbox() {
  const auto begin = nightly_profile_begin();
  const unsigned base = NIGHTLY_PROFILE_SLOT + unsigned(vx_core_id()) * 8;
  nightly_profile_word(base, uint32_t(begin.cycle));
  nightly_profile_word(base + 1, uint32_t(begin.cycle >> 32));
  nightly_profile_word(base + 4, uint32_t(begin.instructions));
  nightly_profile_word(base + 5, uint32_t(begin.instructions >> 32));
}

static inline uint32_t nightly_profile_read_word(unsigned slot) {
  const uint32_t address = 0x80000u + slot * 8u;
  uint32_t value;
  asm volatile("lw.shared %0, 0(%1)" : "=r"(value) : "r"(address) : "memory");
  return value;
}

static inline void nightly_profile_end_mailbox() {
  NightlyProfileSpan span;
  span.end = nightly_profile_end();
  const unsigned core = unsigned(vx_core_id()), base = NIGHTLY_PROFILE_SLOT + core * 8;
  span.begin.cycle = uint64_t(nightly_profile_read_word(base)) |
                    (uint64_t(nightly_profile_read_word(base + 1)) << 32);
  span.begin.instructions = uint64_t(nightly_profile_read_word(base + 4)) |
                           (uint64_t(nightly_profile_read_word(base + 5)) << 32);
  nightly_profile_publish(core, span);
}

/* This is device completion, not an independent host numerical check. The
 * original tohost instruction remains after the post-run status publication. */
static inline void nightly_profile_complete(uint32_t code) {
  mu_fence();
  nightly_profile_word(NIGHTLY_PROFILE_CODE_SLOT, code);
  nightly_profile_word(NIGHTLY_PROFILE_DONE_SLOT, NIGHTLY_PROFILE_DONE);
  asm volatile(".insn i 0x73, 0, x0, %0, 0" :: "r"(code) : "memory");
}
#endif
