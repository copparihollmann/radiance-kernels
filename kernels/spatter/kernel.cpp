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
extern __global uint32_t spatter_group_offsets[];
extern __global uint32_t spatter_group_tasks[];
}

#if SPATTER_CHAIN
static void gather_stage(void*, uint32_t tid, uint32_t threads_per_block,
                         uint32_t block_id) {
  const uint32_t global_tid = block_id * threads_per_block + tid;
  const uint32_t global_threads = MU_NUM_CLUSTERS * threads_per_block;
  constexpr uint32_t length = SPATTER_GATHER_LENGTH;
  const uint32_t owners = length * SPATTER_GATHER_WRAP;
  for (uint32_t owner = global_tid; owner < owners;) {
    const uint32_t r = owner / length;
    const uint32_t j = owner - r * length;
    for (uint32_t i = r; i < SPATTER_GATHER_COUNT; i += SPATTER_GATHER_WRAP) {
      const uint32_t src = spatter::affine_index(
          spatter::pattern_index(spatter_pattern_gather, j),
          SPATTER_GATHER_DELTA, i);
      spatter::copy_value(spatter_dense, owner, spatter_sparse_gather, src);
    }
    if (owners - owner <= global_threads) break;
    owner += global_threads;
  }
}

static void scatter_stage(void*, uint32_t tid, uint32_t threads_per_block,
                          uint32_t block_id) {
  const uint32_t global_tid = block_id * threads_per_block + tid;
  const uint32_t global_threads = MU_NUM_CLUSTERS * threads_per_block;
  constexpr uint32_t length = SPATTER_SCATTER_LENGTH;
  constexpr uint32_t tasks = length * SPATTER_SCATTER_COUNT;
  for (uint32_t task = global_tid; task < tasks;) {
    const uint32_t i = task / length;
    const uint32_t j = task - i * length;
    const uint32_t dst = spatter::affine_index(
        spatter::pattern_index(spatter_pattern_scatter, j),
        SPATTER_SCATTER_DELTA, i);
    const uint32_t src = spatter::dense_index(
        j, length, i, SPATTER_SCATTER_WRAP);
    spatter::copy_value(spatter_sparse_scatter, dst, spatter_dense, src);
    if (tasks - task <= global_threads) break;
    task += global_threads;
  }
}

int main() {
  mu_schedule(gather_stage, nullptr, SPATTER_NUM_WARPS);
  mu_fence();
  mu_barrier(0, MU_NUM_CORES);
  mu_schedule(scatter_stage, nullptr, SPATTER_NUM_WARPS);
  mu_fence();
  return 0;
}

#else
#if SPATTER_KIND == 1 || SPATTER_KIND == 2 || SPATTER_KIND == 4
static inline void scatter_task(uint32_t task, uint32_t owned_destination) {
  constexpr uint32_t length = SPATTER_PATTERN_LENGTH;
  const uint32_t i = task / length;
  const uint32_t j = task - i * length;
#if SPATTER_KIND == 1
#if SPATTER_ORDERED_COLLISIONS
  const uint32_t dst = owned_destination;
#else
  const uint32_t dst = spatter::affine_index(
      spatter::pattern_index(spatter_pattern, j), SPATTER_DELTA, i);
#endif
  const uint32_t src = spatter::dense_index(j, length, i, SPATTER_WRAP);
  spatter::copy_value(spatter_sparse, dst, spatter_dense, src);
#elif SPATTER_KIND == 2
#if SPATTER_ORDERED_COLLISIONS
  const uint32_t dst = owned_destination;
#else
  const uint32_t dst = spatter::affine_index(
      spatter::pattern_index(spatter_pattern_scatter, j), SPATTER_DELTA_SCATTER, i);
#endif
#if SPATTER_GATHER_FINAL_WRAP
  const uint32_t residue = i % SPATTER_GATHER_FINAL_WRAP;
  const uint32_t source_iteration = residue +
      ((SPATTER_COUNT - 1u - residue) / SPATTER_GATHER_FINAL_WRAP) *
          SPATTER_GATHER_FINAL_WRAP;
#else
  const uint32_t source_iteration = i;
#endif
  const uint32_t src = spatter::affine_index(
      spatter::pattern_index(spatter_pattern_gather, j), SPATTER_DELTA_GATHER,
      source_iteration);
  spatter::copy_value(spatter_sparse_scatter, dst, spatter_sparse_gather, src);
#elif SPATTER_KIND == 4
#if SPATTER_ORDERED_COLLISIONS
  const uint32_t dst = owned_destination;
#else
  const uint32_t dst = spatter::affine_index(
      spatter::nested_pattern_index(spatter_pattern, spatter_pattern_scatter, j),
      SPATTER_DELTA, i);
#endif
  const uint32_t src = spatter::dense_index(j, length, i, SPATTER_WRAP);
  spatter::copy_value(spatter_sparse, dst, spatter_dense, src);
#endif
}
#endif

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
#if SPATTER_ORDERED_COLLISIONS
  // The host groups tasks by destination in their original order. One lane
  // owns each destination, so colliding 64-bit stores cannot interleave.
  for (uint32_t owner = global_tid; owner < SPATTER_OUTPUT_LENGTH;) {
    const uint32_t end = spatter_group_offsets[owner + 1u];
    for (uint32_t k = spatter_group_offsets[owner]; k < end; ++k) {
      scatter_task(spatter_group_tasks[k], owner);
    }
    if (SPATTER_OUTPUT_LENGTH - owner <= global_threads) break;
    owner += global_threads;
  }
#else
  constexpr uint32_t tasks = length * SPATTER_COUNT;
  for (uint32_t task = global_tid; task < tasks;) {
    scatter_task(task, 0);
    if (tasks - task <= global_threads) break;
    task += global_threads;
  }
#endif
#endif
}

int main() {
  mu_schedule(kernel_body, nullptr, SPATTER_NUM_WARPS);
  mu_fence();
  return 0;
}
#endif
