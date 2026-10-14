#pragma once

#include <stdint.h>
#include "spatter_data.h"
#include "spatter_ops.hpp"

#if SPATTER_KIND == 1 || SPATTER_KIND == 2 || SPATTER_KIND == 4
static inline void spatter_scatter_task(uint32_t task, uint32_t owned_destination) {
  const uint32_t i = task / SPATTER_LENGTH;
  const uint32_t j = task - i * SPATTER_LENGTH;
  uint32_t src_index, dst_index;
#if SPATTER_KIND == 1
  src_index = spatter::dense_index(j, SPATTER_LENGTH, i, SPATTER_WRAP);
  dst_index = ((const __global uint32_t *)SPATTER_PATTERN_ADDR)[j] +
              SPATTER_DELTA * i;
#elif SPATTER_KIND == 2
#if SPATTER_GATHER_FINAL_WRAP
  const uint32_t residue = i % SPATTER_GATHER_FINAL_WRAP;
  const uint32_t source_iteration = residue +
      ((SPATTER_COUNT - 1u - residue) / SPATTER_GATHER_FINAL_WRAP) *
          SPATTER_GATHER_FINAL_WRAP;
#else
  const uint32_t source_iteration = i;
#endif
  src_index = ((const __global uint32_t *)SPATTER_GATHER_ADDR)[j] +
              SPATTER_DELTA_GATHER * source_iteration;
  dst_index = ((const __global uint32_t *)SPATTER_SCATTER_ADDR)[j] +
              SPATTER_DELTA_SCATTER * i;
#elif SPATTER_KIND == 4
  src_index = spatter::dense_index(j, SPATTER_LENGTH, i, SPATTER_WRAP);
  dst_index = ((const __global uint32_t *)SPATTER_PATTERN_ADDR)[
      ((const __global uint32_t *)SPATTER_SCATTER_ADDR)[j]] + SPATTER_DELTA * i;
#endif
#if SPATTER_ORDERED
  dst_index = owned_destination;
#else
  (void)owned_destination;
#endif
  spatter::copy_value((volatile __global uint32_t *)SPATTER_OUT_ADDR, dst_index,
                      (const volatile __global uint32_t *)SPATTER_SRC_ADDR,
                      src_index);
}
#endif

template <uint32_t Clusters>
static inline void spatter_apply(uint32_t tid, uint32_t tpb, uint32_t cluster) {
  const uint32_t global_tid = cluster * tpb + tid;
  const uint32_t step = Clusters * tpb;
#if SPATTER_KIND == 0 || SPATTER_KIND == 3
  const uint32_t owners = SPATTER_LENGTH * SPATTER_WRAP;
  for (uint32_t owner = global_tid; owner < owners; owner += step) {
    const uint32_t residue = owner / SPATTER_LENGTH;
    const uint32_t j = owner - residue * SPATTER_LENGTH;
    for (uint32_t i = residue; i < SPATTER_COUNT; i += SPATTER_WRAP) {
#if SPATTER_KIND == 0
      const uint32_t base = ((const __global uint32_t *)SPATTER_PATTERN_ADDR)[j];
#else
      const uint32_t base = ((const __global uint32_t *)SPATTER_PATTERN_ADDR)[
          ((const __global uint32_t *)SPATTER_GATHER_ADDR)[j]];
#endif
      spatter::copy_value((volatile __global uint32_t *)SPATTER_OUT_ADDR, owner,
                          (const volatile __global uint32_t *)SPATTER_SRC_ADDR,
                          base + SPATTER_DELTA * i);
    }
  }
#elif SPATTER_ORDERED
  const __global uint32_t *offsets =
      (const __global uint32_t *)SPATTER_OFFSETS_ADDR;
  const __global uint32_t *tasks = (const __global uint32_t *)SPATTER_TASKS_ADDR;
  for (uint32_t owner = global_tid; owner < SPATTER_OUTPUT_LENGTH;
       owner += step) {
    for (uint32_t k = offsets[owner]; k < offsets[owner + 1u]; k++)
      spatter_scatter_task(tasks[k], owner);
  }
#else
  const uint32_t task_count = SPATTER_COUNT * SPATTER_LENGTH;
  for (uint32_t task = global_tid; task < task_count; task += step)
    spatter_scatter_task(task, 0);
#endif
}
