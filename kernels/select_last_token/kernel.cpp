// Select the final prefill hidden row for generation's LM head.
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
#include "data"

extern "C" uint32_t __mu_num_warps = 1;
__global float selected_raw[BATCH * WIDTH] = {0};

static void select_last(void*, uint32_t tid, uint32_t tpb, uint32_t) {
  for (uint32_t i = tid; i < BATCH * WIDTH; i += tpb) {
    const uint32_t batch = i / WIDTH;
    const uint32_t feature = i % WIDTH;
    selected_raw[i] = input_raw[(batch * TOKENS + TOKENS - 1) * WIDTH + feature];
  }
}

int main() {
  mu_schedule(select_last, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES);
  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  if (mu_hart_id() != 0) { for (;;) {} }
  uint32_t errors = 0;
  for (uint32_t i = 0; i < BATCH * WIDTH; ++i)
    if (selected_raw[i] != gold_raw[i]) ++errors;
  mu_tohost(errors ? ((errors << 1) | 1u) : 0u);
  return 0;
}
