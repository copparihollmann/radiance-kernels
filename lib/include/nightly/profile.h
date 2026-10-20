/* Optional Muon counter snapshots. No printing, barriers, DMA fences, or allocation.
 * Call from one lane of each core's boot warp. Read PROFILE.md before using the
 * timestamps as a performance result. Disabled macros do not evaluate arguments.
 */
#ifndef NIGHTLY_PROFILE_H
#define NIGHTLY_PROFILE_H

#include <stdint.h>

#ifndef NIGHTLY_PROFILE
#define NIGHTLY_PROFILE 0
#endif
/* Explicit opt-in: Cyclotron currently exposes minstret but does not update it. */
#ifndef NIGHTLY_PROFILE_HAS_INSTRET
#define NIGHTLY_PROFILE_HAS_INSTRET 0
#endif

#define NIGHTLY_PROFILE_SLOT 32u
#define NIGHTLY_PROFILE_READY_SLOT 56u
#define NIGHTLY_PROFILE_READY 0x50524631u

struct NightlyProfileSample { uint64_t cycle, instructions; };
struct NightlyProfileSpan { NightlyProfileSample begin, end; };

#if NIGHTLY_PROFILE
/* The low half is read between equal high halves. A wrap retries the sample. */
template <unsigned Low, unsigned High>
static inline uint64_t nightly_profile_counter() {
  uint32_t before, low, after;
  do {
    asm volatile("csrr %0, %1" : "=r"(before) : "i"(High) : "memory");
    asm volatile("csrr %0, %1" : "=r"(low) : "i"(Low) : "memory");
    asm volatile("csrr %0, %1" : "=r"(after) : "i"(High) : "memory");
  } while (before != after);
  return (uint64_t(after) << 32) | low;
}

static inline NightlyProfileSample nightly_profile_begin() {
  NightlyProfileSample s = {};
#if NIGHTLY_PROFILE_HAS_INSTRET
  s.instructions = nightly_profile_counter<0xb02, 0xb82>();
#endif
  s.cycle = nightly_profile_counter<0xb00, 0xb80>();
  return s;
}

static inline NightlyProfileSample nightly_profile_end() {
  NightlyProfileSample s = {};
  s.cycle = nightly_profile_counter<0xb00, 0xb80>();
#if NIGHTLY_PROFILE_HAS_INSTRET
  s.instructions = nightly_profile_counter<0xb02, 0xb82>();
#endif
  return s;
}

static inline void nightly_profile_word(unsigned slot, uint32_t value) {
  const uint32_t address = 0x80000u + slot * 8u;
  asm volatile("sw.shared %0, 0(%1)\nsw.shared %0, 4(%1)"
               :: "r"(value), "r"(address) : "memory");
}

/* Publish AFTER taking the end sample and after the workload's own required
 * completion fence. Slots 32..47 and 56..57 must be reserved by this workload.
 * The host clears READY before launch and waits for both cores' committed data.
 * This function is not a completion fence for MX/VPU or cached output stores. */
static inline void nightly_profile_publish(unsigned core, const NightlyProfileSpan &span) {
  if (core >= 2) return;
  const uint64_t values[] = {span.begin.cycle, span.end.cycle,
                            span.begin.instructions, span.end.instructions};
  for (unsigned i = 0; i < 4; ++i) {
    nightly_profile_word(NIGHTLY_PROFILE_SLOT + core * 8 + i * 2, uint32_t(values[i]));
    nightly_profile_word(NIGHTLY_PROFILE_SLOT + core * 8 + i * 2 + 1, uint32_t(values[i] >> 32));
  }
  asm volatile("fence" ::: "memory");
  nightly_profile_word(NIGHTLY_PROFILE_READY_SLOT + core,
      NIGHTLY_PROFILE_READY | (NIGHTLY_PROFILE_HAS_INSTRET ? 0x100u : 0u));
}

#define NIGHTLY_PROFILE_BEGIN(span) ((span).begin = nightly_profile_begin())
#define NIGHTLY_PROFILE_END(span) ((span).end = nightly_profile_end())
#define NIGHTLY_PROFILE_PUBLISH(core, span) nightly_profile_publish((core), (span))
#else
#define NIGHTLY_PROFILE_BEGIN(span) ((void)0)
#define NIGHTLY_PROFILE_END(span) ((void)0)
#define NIGHTLY_PROFILE_PUBLISH(core, span) ((void)0)
#endif

#endif
