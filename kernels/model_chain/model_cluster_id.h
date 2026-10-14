#ifndef MODEL_CHAIN_CLUSTER_ID_H
#define MODEL_CHAIN_CLUSTER_ID_H

#include <stdint.h>
#include <mu_intrinsics.h>

static inline uint32_t model_chain_cluster_id(void) {
  uint32_t id;
  asm volatile("csrr %0, %1" : "=r"(id) : "i"(MU_CSR_CLUSTER_ID));
  return id;
}

#endif
