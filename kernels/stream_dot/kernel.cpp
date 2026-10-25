#include <stdint.h>

#include "generated/config.h"
#include <mu_partition.h>
#include <nightly/device.h>
#include <shared_mem.h>

static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "host and device cluster counts must match");

extern "C" {
extern __global const float stream_dot_a[];
extern __global const float stream_dot_b[];
extern __global float stream_dot_partial[];
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  const MuPartition range = mu_contiguous_partition(
      STREAM_DOT_ELEMENTS, NIGHTLY_CLUSTERS, block_id, 16u);
  float sum = 0.0f;
  for (uint32_t i = range.begin + tid; i < range.end;
       i += threads_per_block) {
    sum += stream_dot_a[i] * stream_dot_b[i];
  }

  // Match BabelStream's CUDA reduction structure: every thread first forms a
  // strided partial, then the whole block performs a power-of-two tree in
  // shared memory. The host combines one partial per physical SM, just as the
  // upstream CUDA host combines one partial per CUDA block.
  volatile __shared float* scratch =
      reinterpret_cast<volatile __shared float*>(0x0);
  scratch[tid] = sum;
  mu_fence_smem();
  for (uint32_t offset = threads_per_block / 2u; offset > 0u; offset /= 2u) {
    mu_barrier(1u, threads_per_block / MU_NUM_THREADS);
    if (tid < offset) scratch[tid] += scratch[tid + offset];
    mu_fence_smem();
  }
  mu_barrier(1u, threads_per_block / MU_NUM_THREADS);
  if (tid == 0u) stream_dot_partial[block_id] = scratch[0];
  nightly_kernel_end(tid, threads_per_block);
}

int main() { return nightly_main(kernel_body, nullptr); }
