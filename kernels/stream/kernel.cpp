#include <stdint.h>

#include "generated/config.h"
#include <mu_partition.h>
#include <nightly/device.h>

static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "host and device cluster counts must match");

extern "C" {
extern __global float stream_a[];
extern __global float stream_b[];
extern __global float stream_c[];
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  // float32 has 16 elements per 64-byte line.  Give each SM whole,
  // contiguous lines so no line ping-pongs between SM-private caches.
  const MuPartition range = mu_contiguous_partition(
      STREAM_ELEMENTS, NIGHTLY_CLUSTERS, block_id, 16u);
  for (uint32_t i = range.begin + tid; i < range.end;
       i += threads_per_block) {
#if STREAM_KIND == 0
    stream_c[i] = stream_a[i];
#elif STREAM_KIND == 1
    stream_b[i] = STREAM_SCALAR * stream_c[i];
#elif STREAM_KIND == 2
    stream_c[i] = stream_a[i] + stream_b[i];
#elif STREAM_KIND == 3
    stream_a[i] = stream_b[i] + STREAM_SCALAR * stream_c[i];
#endif
  }
  nightly_kernel_end(tid, threads_per_block);
}

int main() { return nightly_main(kernel_body, nullptr); }
