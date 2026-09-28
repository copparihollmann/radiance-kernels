#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

#include "generated/config.h"
#include "spatter_ops.hpp"

#define SPATTER_NUM_WARPS 4
extern "C" uint32_t __mu_num_warps = SPATTER_NUM_WARPS;

extern "C" {
extern __global uint32_t spatter_pattern[];
extern __global uint32_t spatter_pattern_gather[];
extern __global uint32_t spatter_pattern_scatter[];
extern __global uint32_t spatter_sparse[];
extern __global uint32_t spatter_dense[];
extern __global uint32_t spatter_sparse_gather[];
extern __global uint32_t spatter_sparse_scatter[];
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  const uint32_t global_tid = block_id * threads_per_block + tid;
  const uint32_t global_threads = MU_NUM_CLUSTERS * threads_per_block;
  constexpr uint32_t length = SPATTER_PATTERN_LENGTH;

#if SPATTER_KIND == 0 || SPATTER_KIND == 3
  // A dense output slot has one owner. It processes the iterations that wrap
  // to that slot in order, so the full Spatter gather result is deterministic.
  const uint32_t owners = length * SPATTER_WRAP;
  for (uint32_t owner = global_tid; owner < owners;) {
    const uint32_t r = owner / length;
    const uint32_t j = owner - r * length;
    for (uint32_t i = r; i < SPATTER_COUNT; i += SPATTER_WRAP) {
#if SPATTER_KIND == 0
      const uint32_t src = spatter::affine_index(
          spatter::pattern_index(spatter_pattern, j), SPATTER_DELTA, i);
#else
      const uint32_t src = spatter::affine_index(
          spatter::nested_pattern_index(spatter_pattern, spatter_pattern_gather, j),
          SPATTER_DELTA, i);
#endif
      spatter::copy_value(spatter_dense, owner, spatter_sparse, src);
    }
    if (owners - owner <= global_threads) break;
    owner += global_threads;
  }
#else
  constexpr uint32_t tasks = length * SPATTER_COUNT;
  for (uint32_t task = global_tid; task < tasks;) {
    const uint32_t i = task / length;
    const uint32_t j = task - i * length;
#if SPATTER_KIND == 1
    const uint32_t dst = spatter::affine_index(
        spatter::pattern_index(spatter_pattern, j), SPATTER_DELTA, i);
    const uint32_t src = spatter::dense_index(j, length, i, SPATTER_WRAP);
    spatter::copy_value(spatter_sparse, dst, spatter_dense, src);
#elif SPATTER_KIND == 2
    const uint32_t dst = spatter::affine_index(
        spatter::pattern_index(spatter_pattern_scatter, j), SPATTER_DELTA_SCATTER, i);
    const uint32_t src = spatter::affine_index(
        spatter::pattern_index(spatter_pattern_gather, j), SPATTER_DELTA_GATHER, i);
    spatter::copy_value(spatter_sparse_scatter, dst, spatter_sparse_gather, src);
#elif SPATTER_KIND == 4
    const uint32_t dst = spatter::affine_index(
        spatter::nested_pattern_index(spatter_pattern, spatter_pattern_scatter, j),
        SPATTER_DELTA, i);
    const uint32_t src = spatter::dense_index(j, length, i, SPATTER_WRAP);
    spatter::copy_value(spatter_sparse, dst, spatter_dense, src);
#endif
    if (tasks - task <= global_threads) break;
    task += global_threads;
  }
#endif
}

int main() {
  mu_schedule(kernel_body, nullptr, SPATTER_NUM_WARPS);
  mu_fence();
  return 0;
}
