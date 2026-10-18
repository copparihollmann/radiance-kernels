// Device regression: mu_schedule must join every worker before returning.
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <kernel_verify.h>

extern "C" uint32_t __mu_num_warps = 8;
alignas(64) __global volatile uint32_t completed[512] = {};

static uint32_t cluster_id() {
  uint32_t value;
  asm volatile("csrr %0, %1" : "=r"(value) : "i"(MU_CSR_CLUSTER_ID));
  return value;
}

static void stage_delayed_worker(void*, uint32_t tid, uint32_t threads, uint32_t) {
  // Warp zero returns early; later warps still have observable writes pending.
  const unsigned delay = 2000 * (unsigned(vx_warp_id()) + 1);
  for (unsigned i = 0; i < delay; ++i) asm volatile("nop");
  completed[cluster_id() * threads + tid] = tid + 1;
}

int main() {
  mu_schedule(stage_delayed_worker, nullptr, 8);
  mu_barrier(0, MU_NUM_CORES);
  mu_fence();
  if (vx_core_id() != 0) { for (;;) {} }
  const unsigned threads = 8 * MU_NUM_THREADS * MU_NUM_CORES;
  const unsigned base = cluster_id() * threads;
  unsigned errors = 0;
  for (unsigned i = 0; i < threads; ++i)
    if (completed[base + i] != i + 1) ++errors;
  mu_tohost(errors ? (errors * 2 + 1) : 0);
  return 0;
}
