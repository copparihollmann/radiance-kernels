#pragma once

#include <stdint.h>

#include "generated/config.h"

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
static inline void streamcluster_pgain(
    StreamClusterArgs* args, uint32_t tid, uint32_t threads_per_threadblock,
    uint32_t threadblock_id) {
  const uint32_t begin = kStreamClusterBoundaries[threadblock_id];
  const uint32_t end = kStreamClusterBoundaries[threadblock_id + 1u];
  const uint32_t switch_base =
      threadblock_id * STREAMCLUSTER_SWITCH_SEGMENT_STRIDE;

  for (uint32_t point = begin + tid; point < end;
       point += threads_per_threadblock) {
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
  }
}
