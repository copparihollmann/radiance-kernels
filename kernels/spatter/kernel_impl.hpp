// Spatter on Muon: Gather, Scatter, GatherScatter, MultiGather, MultiScatter,
// plus the materialized and fused Gather->Scatter chains.
//
// SPATTER_ILP is the unroll factor; kernel.cpp builds the ILP=1 baseline and
// kernel_ilp{2,4}.cpp the unrolled variants, the same arrangement as
// kernels/vecadd.  These are the purest memory kernels in the set -- no
// floating-point arithmetic at all, just indexed loads and stores -- so the
// only thing that moves them is more independent accesses in flight.
#ifndef SPATTER_KERNEL_IMPL_HPP
#define SPATTER_KERNEL_IMPL_HPP

#include <stdint.h>

#include "generated/config.h"
#include <mu_partition.h>
#include <nightly/device.h>
#include "spatter_ops.hpp"

// hpc.mk picks NIGHTLY_CLUSTERS from the preset; HPC_SM_COUNT is the cluster
// count the generated data was partitioned for.  A mismatch means the ELF
// would read someone else's share, so fail the build rather than the run.
static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "preset cluster count must match the generated data");

#ifndef SPATTER_ILP
#define SPATTER_ILP 1
#endif

extern "C" {
extern __global uint32_t spatter_pattern[];
extern __global uint32_t spatter_pattern_gather[];
extern __global uint32_t spatter_pattern_scatter[];
extern __global uint32_t spatter_sparse[];
extern __global uint32_t spatter_dense[];
extern __global uint32_t spatter_sparse_gather[];
extern __global uint32_t spatter_sparse_scatter[];
extern __global uint32_t spatter_group_destinations[];
extern __global uint32_t spatter_group_offsets[];
extern __global uint32_t spatter_group_tasks[];
}

#if SPATTER_DESTINATION_OWNER_GROUPS || (SPATTER_CHAIN && SPATTER_CHAIN_ORDERED_COLLISIONS)
// mu_contiguous_partition splits the compacted destination-group list by RANK.
// spatter_group_destinations is an indirection: the list holds only the
// *occupied* destinations, so eight adjacent ranks are not eight adjacent
// addresses and a rank-aligned boundary can land inside a 64-byte destination
// line.  Two SMs would then issue the non-atomic 32-bit halves of different
// payloads into the same line, which is the tearing the destination-owner
// repair exists to prevent.  Derive the boundary from the destination itself:
// advance a candidate rank until it is the first rank of a new destination
// line.  The map is monotone and every SM computes it identically, so
// partition p's end is still partition p+1's begin and the groups stay
// contiguous, disjoint and exhaustive.  It walks at most kLineElements - 1
// entries because a line holds at most that many occupied destinations.
static inline uint32_t spatter_destination_line_boundary(uint32_t rank,
                                                         uint32_t total) {
  if (rank >= total)
    return total;
  while (rank != 0u &&
         spatter::destination_line(spatter_group_destinations[rank]) ==
             spatter::destination_line(spatter_group_destinations[rank - 1u])) {
    ++rank;
    if (rank >= total)
      return total;
  }
  return rank;
}
#endif

#if SPATTER_CHAIN
template <uint32_t ILP>
static void gather_stage_impl(void*, uint32_t tid, uint32_t threads_per_block,
                         uint32_t block_id) {
  constexpr uint32_t length = SPATTER_GATHER_LENGTH;
  const uint32_t owners = length * SPATTER_GATHER_WRAP;
  const MuPartition range = mu_contiguous_partition(
      owners, NIGHTLY_CLUSTERS, block_id, 8u);
  mu_grid_stride<ILP>(range.begin, range.end, tid, threads_per_block,
                      [&](uint32_t owner) {
    NIGHTLY_LANE_SAMPLE();
    const uint32_t r = owner / length;
    const uint32_t j = owner - r * length;
    for (uint32_t i = r; i < SPATTER_GATHER_COUNT; i += SPATTER_GATHER_WRAP) {
      const uint32_t src = spatter::affine_index(
          spatter::pattern_index(spatter_pattern_gather, j),
          SPATTER_GATHER_DELTA, i);
      spatter::copy_value(spatter_dense, owner, spatter_sparse_gather, src);
    }
                      });
}

static inline void scatter_chain_task(uint32_t task, uint32_t owned_destination) {
  constexpr uint32_t length = SPATTER_SCATTER_LENGTH;
  const uint32_t i = task / length;
  const uint32_t j = task - i * length;
#if SPATTER_CHAIN_ORDERED_COLLISIONS
  const uint32_t dst = owned_destination;
#else
  const uint32_t dst = spatter::affine_index(
      spatter::pattern_index(spatter_pattern_scatter, j),
      SPATTER_SCATTER_DELTA, i);
#endif
  const uint32_t src = spatter::dense_index(
      j, length, i, SPATTER_SCATTER_WRAP);
  spatter::copy_value(spatter_sparse_scatter, dst, spatter_dense, src);
}

template <uint32_t ILP>
static void scatter_stage_impl(void*, uint32_t tid, uint32_t threads_per_block,
                          uint32_t block_id) {
#if SPATTER_CHAIN_ORDERED_COLLISIONS
  const MuPartition ranks = mu_contiguous_partition(
      SPATTER_GROUP_COUNT, NIGHTLY_CLUSTERS, block_id, 8u);
  // Snap both rank boundaries onto destination-line boundaries so no 64-byte
  // destination line is written by two SMs.
  const MuPartition range = {
      spatter_destination_line_boundary(ranks.begin, SPATTER_GROUP_COUNT),
      spatter_destination_line_boundary(ranks.end, SPATTER_GROUP_COUNT)};
  mu_grid_stride<ILP>(range.begin, range.end, tid, threads_per_block,
                      [&](uint32_t group) {
    NIGHTLY_LANE_SAMPLE();
    const uint32_t end = spatter_group_offsets[group + 1u];
    for (uint32_t k = spatter_group_offsets[group]; k < end; ++k) {
      scatter_chain_task(spatter_group_tasks[k], spatter_group_destinations[group]);
    }
                      });
#else
  constexpr uint32_t tasks = SPATTER_SCATTER_LENGTH * SPATTER_SCATTER_COUNT;
  const MuPartition range = mu_contiguous_partition(
      tasks, NIGHTLY_CLUSTERS, block_id, 16u);
  mu_grid_stride<ILP>(range.begin, range.end, tid, threads_per_block,
                      [&](uint32_t task) {
    NIGHTLY_LANE_SAMPLE();
    scatter_chain_task(task, 0);
                      });
#endif
}

template <uint32_t ILP>
static void chain_body_impl(void* arg, uint32_t tid, uint32_t threads_per_block,
                       uint32_t block_id) {
  gather_stage(arg, tid, threads_per_block, block_id);
  // Keep both stages in one launch. A second mu_schedule left some Scatter
  // worker output blocks unwritten in the full-size timing simulation.
  mu_barrier(1, NIGHTLY_OCC * MU_NUM_CORES);
  scatter_stage(arg, tid, threads_per_block, block_id);
  nightly_kernel_end(tid, threads_per_block);
}

static void chain_body(void* arg, uint32_t tid,
                        uint32_t threads_per_block,
                        uint32_t block_id) {
  chain_body_impl<SPATTER_ILP>(arg, tid, threads_per_block, block_id);
}

int main() { return nightly_main(chain_body, nullptr); }

#else
#if SPATTER_KIND == 0 || SPATTER_KIND == 3
static inline uint32_t gather_source(uint32_t j, uint32_t i) {
#if SPATTER_KIND == 0
  return spatter::affine_index(
      spatter::pattern_index(spatter_pattern, j), SPATTER_DELTA, i);
#else
  return spatter::affine_index(
      spatter::nested_pattern_index(spatter_pattern, spatter_pattern_gather, j),
      SPATTER_DELTA, i);
#endif
}
#endif

#if SPATTER_KIND == 1 || SPATTER_KIND == 2 || SPATTER_KIND == 4
static inline void scatter_task(uint32_t task, uint32_t owned_destination) {
  constexpr uint32_t length = SPATTER_PATTERN_LENGTH;
  const uint32_t i = task / length;
  const uint32_t j = task - i * length;
#if SPATTER_KIND == 1
#if SPATTER_DESTINATION_OWNER_GROUPS
  const uint32_t dst = owned_destination;
#else
  const uint32_t dst = spatter::affine_index(
      spatter::pattern_index(spatter_pattern, j), SPATTER_DELTA, i);
#endif
  const uint32_t src = spatter::dense_index(j, length, i, SPATTER_WRAP);
  spatter::copy_value(spatter_sparse, dst, spatter_dense, src);
#elif SPATTER_KIND == 2
#if SPATTER_DESTINATION_OWNER_GROUPS
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
#if SPATTER_DESTINATION_OWNER_GROUPS
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

template <uint32_t ILP>
static void kernel_body_impl(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  constexpr uint32_t length = SPATTER_PATTERN_LENGTH;

#if SPATTER_KIND == 0 || SPATTER_KIND == 3
#if SPATTER_GATHER_BLOCK_CYCLIC_ELEMENTS
  // Assign complete, cache-line-aligned output blocks round-robin across SMs.
  // Larger blocks amortize the block-cyclic scheduling overhead while still
  // mixing distant sections of an irregular pattern to reduce per-SM skew.
  constexpr uint32_t owners = length * SPATTER_WRAP;
  constexpr uint32_t block_elements = SPATTER_GATHER_BLOCK_CYCLIC_ELEMENTS;
  constexpr uint32_t chunks =
      (owners + block_elements - 1u) / block_elements;
  for (uint32_t chunk = block_id; chunk < chunks;
       chunk += NIGHTLY_CLUSTERS) {
    const uint32_t begin = chunk * block_elements;
    const uint32_t candidate_end = begin + block_elements;
    const uint32_t end = candidate_end < owners ? candidate_end : owners;
    mu_grid_stride<ILP>(begin, end, tid, threads_per_block,
                        [&](uint32_t owner) {
      NIGHTLY_LANE_SAMPLE();
      const uint32_t r = owner / length;
      const uint32_t j = owner - r * length;
      for (uint32_t i = r; i < SPATTER_COUNT; i += SPATTER_WRAP) {
        spatter::copy_value(spatter_dense, owner, spatter_sparse,
                            gather_source(j, i));
      }
                        });
  }
#elif SPATTER_GATHER_BLOCK_CYCLIC_16
  // Assign complete 16-element (two cache-line) output chunks round-robin
  // across SMs.  This retains exclusive cache-line ownership while mixing
  // distant sections of an irregular pattern to reduce per-SM load skew.
  constexpr uint32_t owners = length * SPATTER_WRAP;
  constexpr uint32_t block_elements = 16u;
  constexpr uint32_t chunks =
      (owners + block_elements - 1u) / block_elements;
  const uint32_t owned_chunks = block_id < chunks
      ? (chunks + NIGHTLY_CLUSTERS - 1u - block_id) / NIGHTLY_CLUSTERS
      : 0u;
  const uint32_t chunk_stride = threads_per_block / block_elements;
  const uint32_t lane = tid % block_elements;
  for (uint32_t chunk_rank = tid / block_elements;
       chunk_rank < owned_chunks; chunk_rank += chunk_stride) {
    const uint32_t chunk = block_id + chunk_rank * NIGHTLY_CLUSTERS;
    const uint32_t owner = chunk * block_elements + lane;
    if (owner >= owners)
      continue;
    const uint32_t r = owner / length;
    const uint32_t j = owner - r * length;
    for (uint32_t i = r; i < SPATTER_COUNT; i += SPATTER_WRAP) {
      spatter::copy_value(spatter_dense, owner, spatter_sparse,
                          gather_source(j, i));
    }
  }
#else
  // A dense output slot has one owner. It processes the iterations that wrap
  // to that slot in order, so the full Spatter gather result is deterministic.
  constexpr uint32_t owners = length * SPATTER_WRAP;
  const MuPartition range = mu_contiguous_partition(
      owners, NIGHTLY_CLUSTERS, block_id, 8u);
  mu_grid_stride<ILP>(range.begin, range.end, tid, threads_per_block,
                      [&](uint32_t owner) {
    NIGHTLY_LANE_SAMPLE();
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
                      });
#endif
#else
#if SPATTER_DESTINATION_OWNER_GROUPS
  // The host groups tasks by destination in their original order. One lane
  // owns each destination, so colliding 64-bit stores cannot interleave. This
  // is also used by the parallel-Scatter physical repair: destinations still
  // run in parallel and every original transfer executes, but one destination
  // cannot be written by two RV32 lanes at once.
  const MuPartition ranks = mu_contiguous_partition(
      SPATTER_GROUP_COUNT, NIGHTLY_CLUSTERS, block_id, 8u);
  // Snap both rank boundaries onto destination-line boundaries so no 64-byte
  // destination line is written by two SMs.
  const MuPartition range = {
      spatter_destination_line_boundary(ranks.begin, SPATTER_GROUP_COUNT),
      spatter_destination_line_boundary(ranks.end, SPATTER_GROUP_COUNT)};
  mu_grid_stride<ILP>(range.begin, range.end, tid, threads_per_block,
                      [&](uint32_t group) {
    NIGHTLY_LANE_SAMPLE();
    const uint32_t end = spatter_group_offsets[group + 1u];
    for (uint32_t k = spatter_group_offsets[group]; k < end; ++k) {
      scatter_task(spatter_group_tasks[k], spatter_group_destinations[group]);
    }
                      });
#else
  constexpr uint32_t tasks = length * SPATTER_COUNT;
  const MuPartition range = mu_contiguous_partition(
      tasks, NIGHTLY_CLUSTERS, block_id, 16u);
  mu_grid_stride<ILP>(range.begin, range.end, tid, threads_per_block,
                      [&](uint32_t task) {
    NIGHTLY_LANE_SAMPLE();
    scatter_task(task, 0);
                      });
#endif
#endif
  nightly_kernel_end(tid, threads_per_block);
}

static void kernel_body(void* arg, uint32_t tid,
                        uint32_t threads_per_block,
                        uint32_t block_id) {
  kernel_body_impl<SPATTER_ILP>(arg, tid, threads_per_block, block_id);
}

int main() { return nightly_main(kernel_body, nullptr); }
#endif

#endif
