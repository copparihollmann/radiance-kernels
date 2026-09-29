#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

#include "generated/config.h"

#define STREAM_WARPS 4
extern "C" uint32_t __mu_num_warps = STREAM_WARPS;

extern "C" {
extern __global float stream_a[];
extern __global float stream_b[];
extern __global float stream_c[];
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  const uint32_t global_tid = block_id * threads_per_block + tid;
  const uint32_t global_threads = MU_NUM_CLUSTERS * threads_per_block;
  for (uint32_t i = global_tid; i < STREAM_ELEMENTS;) {
#if STREAM_KIND == 0
    stream_c[i] = stream_a[i];
#elif STREAM_KIND == 1
    stream_b[i] = 2.0f * stream_c[i];
#elif STREAM_KIND == 2
    stream_c[i] = stream_a[i] + stream_b[i];
#elif STREAM_KIND == 3
    stream_a[i] = stream_b[i] + 2.0f * stream_c[i];
#endif
    if (STREAM_ELEMENTS - i <= global_threads) break;
    i += global_threads;
  }
}

int main() {
  mu_schedule(kernel_body, nullptr, STREAM_WARPS);
  mu_fence();
  return 0;
}
