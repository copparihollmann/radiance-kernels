/* Rocket-side collection. Include after nightly/host.h. Emit only after all GPU
 * work has finished: even a host MMIO read or console write can cause contention. */
#ifndef NIGHTLY_PROFILE_HOST_H
#define NIGHTLY_PROFILE_HOST_H
#include <nightly/profile.h>

static inline void nightly_profile_host_clear(unsigned clusters) {
  for (unsigned sm = 0; sm < clusters; ++sm)
    for (unsigned core = 0; core < 2; ++core)
      nightly_pb_write(sm, NIGHTLY_PROFILE_READY_SLOT + core, 0);
  asm volatile("fence" ::: "memory");
}

static inline uint64_t nightly_profile_host_u64(unsigned sm, unsigned slot) {
  /* Committed snapshot words are immutable, so a second high read is unnecessary. */
  const uint64_t low = nightly_pb_read(sm, slot);
  return low | (uint64_t(nightly_pb_read(sm, slot + 1)) << 32);
}

/* 0 means at least one record is missing; never print an incomplete snapshot. */
static inline int nightly_profile_host_ready(unsigned clusters) {
  for (unsigned sm = 0; sm < clusters; ++sm)
    for (unsigned core = 0; core < 2; ++core)
      if ((nightly_pb_read(sm, NIGHTLY_PROFILE_READY_SLOT + core) & ~0x100u)
           != NIGHTLY_PROFILE_READY) return 0;
  return 1;
}

static inline int nightly_profile_host_report(unsigned clusters) {
  if (!nightly_profile_host_ready(clusters)) return 0;
  for (unsigned sm = 0; sm < clusters; ++sm) {
    for (unsigned core = 0; core < 2; ++core) {
      const unsigned base = NIGHTLY_PROFILE_SLOT + core * 8;
      const bool instret = nightly_pb_read(sm, NIGHTLY_PROFILE_READY_SLOT + core) & 0x100u;
      nh_puts("RADIANCE_PROFILE {\"schema\":1,\"sm\":"); nh_putu(sm);
      nh_puts(",\"core\":"); nh_putu(core);
      const char *names[] = {",\"cycle_begin\":", ",\"cycle_end\":",
                            ",\"instructions_begin\":", ",\"instructions_end\":"};
      for (unsigned i = 0; i < 4; ++i) {
        nh_puts(names[i]);
        if (i >= 2 && !instret) nh_puts("null");
        else nh_putu(nightly_profile_host_u64(sm, base + i * 2));
      }
      nh_puts("}\n");
    }
  }
  nh_flush();
  return 1;
}
#endif
