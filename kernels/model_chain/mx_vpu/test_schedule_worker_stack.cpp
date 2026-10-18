// Repeated mu_schedule launches must restart every worker with the same SP.
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <kernel_verify.h>
#include "model_cluster_id.h"

extern "C" uint32_t __mu_num_warps = 4;
constexpr unsigned participants = 4 * MU_NUM_CORES * MU_NUM_THREADS;
alignas(64) __global volatile uint32_t first_sp[2 * participants] = {};
alignas(64) __global volatile uint32_t last_sp[2 * participants] = {};
alignas(64) __global volatile uint32_t calls[2 * participants] = {};
alignas(64) __global volatile uint32_t errors[2 * participants] = {};

static void stage_probe(void*, uint32_t tid, uint32_t, uint32_t) {
  uint32_t sp;
  asm volatile("mv %0, sp" : "=r"(sp));
  const unsigned slot = model_chain_cluster_id() * participants + tid;
  if (!calls[slot]) first_sp[slot] = sp;
  if (first_sp[slot] != sp) ++errors[slot];
  last_sp[slot] = sp;
  ++calls[slot];
}

int main() {
  const unsigned base = model_chain_cluster_id() * participants;
  for (unsigned round = 0; round < 4096; ++round) {
    mu_schedule(stage_probe, nullptr, 4);
    mu_barrier(0, MU_NUM_CORES);
    mu_fence();
    // Both manager cores observe the same completed launch and fail before
    // another launch can corrupt a neighbouring warp's saved return address.
    for (unsigned i = 0; i < participants; ++i)
      if (errors[base+i]) { mu_tohost(1); return 0; }
  }
  if (vx_core_id() != 0) { for (;;) {} }
  for (unsigned i = 0; i < participants; ++i)
    if (calls[base+i] != 4096 || first_sp[base+i] != last_sp[base+i]) {
      mu_tohost(1); return 0;
    }
  mu_tohost(0);
  return 0;
}
