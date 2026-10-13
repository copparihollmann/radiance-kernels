/* PrintBuf completion protocol for model-chain programs on Radiance HBM RTL. */
#ifndef MODEL_HBM_BRIDGE_H
#define MODEL_HBM_BRIDGE_H

#include <stdint.h>
#include <mu_intrinsics.h>
#include <nightly/pb.h>

#define MODEL_HBM_PASS 0x600d0000u

static inline uint32_t model_hbm_cycle(void) {
  uint32_t value;
  asm volatile("csrr %0, mcycle" : "=r"(value));
  return value;
}

static inline void model_hbm_put(uint32_t slot, uint32_t value) {
  const uint32_t address = NIGHTLY_PB_DEV_BASE + slot * 8u;
  asm volatile("sw.shared %0, 0(%1)" :: "r"(value), "r"(address) : "memory");
  asm volatile("sw.shared %0, 4(%1)" :: "r"(value), "r"(address) : "memory");
}

static inline uint32_t model_hbm_get(uint32_t slot) {
  const uint32_t address = NIGHTLY_PB_DEV_BASE + slot * 8u;
  uint32_t value;
  asm volatile("lw.shared %0, 0(%1)" : "=r"(value) : "r"(address) : "memory");
  return value;
}

static inline void model_hbm_wait_launch(void) {
  while (model_hbm_get(NIGHTLY_PB_LAUNCH) != NIGHTLY_LAUNCH_MAGIC)
    asm volatile("nop");
  model_hbm_put(NIGHTLY_PB_START + (uint32_t)vx_core_id(), model_hbm_cycle());
}

static inline void model_hbm_status(uint32_t code) {
  model_hbm_put(NIGHTLY_PB_STATUS, code ? code : MODEL_HBM_PASS);
}

static inline void model_hbm_end(void) {
  mu_fence();
  model_hbm_put(NIGHTLY_PB_END + (uint32_t)vx_core_id(), model_hbm_cycle());
}

#endif
