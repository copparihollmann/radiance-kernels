#ifndef MODEL_CHAIN_CLUSTER_ID_H
#define MODEL_CHAIN_CLUSTER_ID_H

#include <stdint.h>
#include <mu_intrinsics.h>
#include <nightly/cluster_id.h>

static inline uint32_t model_chain_cluster_id(void) {
#if defined(MU_CLUSTER_ID_FROM_PRINTBUF) && MU_CLUSTER_ID_FROM_PRINTBUF
  return mu_cluster_id();
#else
  uint32_t id;
  asm volatile("csrr %0, %1" : "=r"(id) : "i"(MU_CSR_CLUSTER_ID));
  return id;
#endif
}

#endif
