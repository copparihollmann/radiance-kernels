#ifndef NIGHTLY_CLUSTER_ID_H
#define NIGHTLY_CLUSTER_ID_H

#include <stdint.h>
#include <mu_intrinsics.h>
#include <nightly/pb.h>

/* Include in the device entry translation unit only. Four-SM launchers write
 * this tag before releasing GPU reset; older hosts retain the CSR fallback. */
#if (defined(MU_CLUSTER_ID_FROM_PRINTBUF) && MU_CLUSTER_ID_FROM_PRINTBUF) || \
    (defined(NIGHTLY_CLUSTERS) && NIGHTLY_CLUSTERS > 2)
extern "C" __attribute__((noinline)) uint32_t mu_cluster_id(void) {
  uint32_t value;
  const uint32_t address = NIGHTLY_PB_DEV_BASE + 8u * NIGHTLY_PB_CLUSTER;
  asm volatile("lw.shared %0, 0(%1)" : "=r"(value) : "r"(address) : "memory");
  if ((value & 0xffff0000u) == NIGHTLY_CLUSTER_TAG) return value & 0xffffu;
  asm volatile("csrr %0, %1" : "=r"(value) : "i"(MU_CSR_CLUSTER_ID));
  return value;
}
#endif

#endif
