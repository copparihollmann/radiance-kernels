#pragma once

#include <stdint.h>

// Rodinia streamcluster pgain on Muon: one gain evaluation per point.
//
// STREAMCLUSTER_ILP is the unroll factor; kernel.cpp builds the ILP=1 baseline
// and kernel_ilp{2,4}.cpp the unrolled variants, the same arrangement as
// kernels/vecadd.  Each unrolled slot is an independent point, so the strided
// coordinate gathers of several points overlap.
//
// Point ownership is NOT one of mu_partition.h's policies: the boundaries come
// from the generator (STREAMCLUSTER_SM_POINT_BOUNDARIES).

#include "generated/config.h"
#include <mu_partition.h>
#include <nightly/device.h>

#ifndef STREAMCLUSTER_ILP
#define STREAMCLUSTER_ILP 1
#endif

struct StreamClusterPoint {
  float weight;
  int32_t assign;
  float cost;
};

static_assert(sizeof(StreamClusterPoint) == 12u,
              "pinned point ABI is float/int/float");

struct StreamClusterArgs {
  __global const StreamClusterPoint* points;
  __global const float* coordinates;
  __global const int32_t* center_table;
  __global float* work;
  __global uint8_t* switches;
};

static constexpr uint32_t kStreamClusterBoundaries[HPC_SM_COUNT + 1u] =
    STREAMCLUSTER_SM_POINT_BOUNDARIES;

static_assert(STREAMCLUSTER_LOCAL_SIZE == 16u,
              "pinned Vortex local size must remain 16");
static_assert((STREAMCLUSTER_WORK_ROW_BYTES * STREAMCLUSTER_POINT_GRANULE) %
                  64u == 0u,
              "point ownership must not split a work output cache line");
static_assert(STREAMCLUSTER_SWITCH_SEGMENT_STRIDE % 64u == 0u,
              "switch ownership segments must begin on cache lines");

#pragma clang fp contract(off)
template <uint32_t ILP>
static inline void streamcluster_pgain_impl(
    StreamClusterArgs* args, uint32_t tid, uint32_t threads_per_threadblock,
    uint32_t threadblock_id) {
  const uint32_t begin = kStreamClusterBoundaries[threadblock_id];
  const uint32_t end = kStreamClusterBoundaries[threadblock_id + 1u];
  const uint32_t switch_base =
      threadblock_id * STREAMCLUSTER_SWITCH_SEGMENT_STRIDE;

  mu_grid_stride<ILP>(begin, end, tid, threads_per_threadblock,
                      [&](uint32_t point) {
    NIGHTLY_LANE_SAMPLE();
    float x_cost = 0.0f;
    for (uint32_t dimension = 0; dimension < STREAMCLUSTER_DIM; ++dimension) {
      const float difference =
          args->coordinates[dimension * STREAMCLUSTER_NUM + point] -
          args->coordinates[dimension * STREAMCLUSTER_NUM +
                            STREAMCLUSTER_CANDIDATE];
      x_cost = x_cost + difference * difference;
    }
    x_cost = x_cost * args->points[point].weight;
    const float current_cost = args->points[point].cost;
    __global float* row =
        args->work + point * STREAMCLUSTER_WORK_ROW_FLOATS;

    // This is the pinned memset final state, fused into point ownership so a
    // skipped point remains poisoned and every output row has one writer.
    for (uint32_t column = 0; column < STREAMCLUSTER_WORK_ROW_FLOATS;
         ++column)
      row[column] = 0.0f;

    const uint32_t local_point = point - begin;
    if (x_cost < current_cost) {
      args->switches[switch_base + local_point] = static_cast<uint8_t>('1');
      row[STREAMCLUSTER_K] = x_cost - current_cost;
    } else {
      args->switches[switch_base + local_point] = 0u;
      const int32_t assign = args->points[point].assign;
      const int32_t compact = args->center_table[assign];
      row[compact] += current_cost - x_cost;
    }
  });
}

static inline void streamcluster_pgain(
    StreamClusterArgs* args, uint32_t tid, uint32_t threads_per_threadblock,
    uint32_t threadblock_id) {
  streamcluster_pgain_impl<STREAMCLUSTER_ILP>(
      args, tid, threads_per_threadblock, threadblock_id);
}

// hpc.mk picks NIGHTLY_CLUSTERS from the preset; HPC_SM_COUNT is the cluster
// count the generated data was partitioned for.  A mismatch means the ELF
// would read someone else's share, so fail the build rather than the run.
static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "preset cluster count must match the generated data");

extern "C" {
extern __global const StreamClusterPoint streamcluster_points[];
extern __global const float streamcluster_coordinates[];
extern __global const int32_t streamcluster_center_table[];
extern __global float streamcluster_work_output[];
extern __global uint8_t streamcluster_switch_output[];
}

static StreamClusterArgs streamcluster_args = {
    .points = streamcluster_points,
    .coordinates = streamcluster_coordinates,
    .center_table = streamcluster_center_table,
    .work = streamcluster_work_output,
    .switches = streamcluster_switch_output,
};

static void kernel_body(void* opaque, uint32_t tid,
                        uint32_t threads_per_threadblock,
                        uint32_t physical_threadblock_id) {
  streamcluster_pgain(
      reinterpret_cast<StreamClusterArgs*>(opaque), tid,
      threads_per_threadblock, physical_threadblock_id);
  nightly_kernel_end(tid, threads_per_threadblock);
}

int main() { return nightly_main(kernel_body, &streamcluster_args); }
