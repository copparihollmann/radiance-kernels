#include <stdint.h>

#include "generated/config.h"
#include <nightly/device.h>

static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "host and device cluster counts must match");
static_assert(GAUSSIAN_SIZE <= GAUSSIAN_OWNER_STRIDE,
              "owner stride must cover every row");
static_assert(GAUSSIAN_ROWS_PER_LINE * sizeof(float) == 64u,
              "row granule must be one float32 cache line");

extern "C" {
extern __global float gaussian_a[];
extern __global float gaussian_b[];
extern __global float gaussian_m[];
extern __global float gaussian_model_a[];
extern __global float gaussian_model_b[];
extern __global float gaussian_model_m[];
extern __global uint32_t gaussian_owners[];
}

static inline bool owns_row(uint32_t row, uint32_t cluster) {
  return ((row / GAUSSIAN_ROWS_PER_LINE) % NIGHTLY_CLUSTERS) == cluster;
}

#pragma clang fp contract(off)
static void gaussian_phase(void* raw, uint32_t tid,
                           uint32_t threads_per_block, uint32_t block_id) {
  (void)raw;
  // Do not pass this state on the boot warp's stack: mu_schedule worker
  // stacks are lane-interleaved and may overlap such a live frame.  Every
  // worker independently reads the same uncached launch argument instead.
  const uint32_t argument = nightly_pb_get(NIGHTLY_PB_ARG);
  const bool model_stage = (argument & GAUSSIAN_MODEL_STAGE_FLAG) != 0u;
  const uint32_t stage = argument & ~GAUSSIAN_MODEL_STAGE_FLAG;
  const uint32_t t = stage / 2u;
  __global float* a = model_stage
      ? gaussian_model_a + stage * GAUSSIAN_MATRIX_WORDS : gaussian_a;
  __global float* b = model_stage
      ? gaussian_model_b + stage * GAUSSIAN_SIZE : gaussian_b;
  __global float* m = model_stage
      ? gaussian_model_m + stage * GAUSSIAN_MATRIX_WORDS : gaussian_m;
  const uint32_t first = t + 1u;
  __global uint32_t* stage_owners =
      gaussian_owners + stage * GAUSSIAN_OWNER_STRIDE;

  if ((stage & 1u) == 0u) {
    // Pinned Vortex/Rodinia Fan1: one multiplier per row below the pivot.
    for (uint32_t row = first + tid; row < GAUSSIAN_SIZE;
         row += threads_per_block) {
      if (!owns_row(row, block_id)) continue;
      m[row * GAUSSIAN_SIZE + t] =
          a[row * GAUSSIAN_SIZE + t] / a[t * GAUSSIAN_SIZE + t];
      stage_owners[row] = block_id;
    }
  } else {
    // Pinned Fan2's 2-D work-items, flattened inside each owned row block.
    // Exactly the col==t work-item updates b[row].
    const uint32_t columns = GAUSSIAN_SIZE - t;
    for (uint32_t block = block_id;
         block * GAUSSIAN_ROWS_PER_LINE < GAUSSIAN_SIZE;
         block += NIGHTLY_CLUSTERS) {
      uint32_t begin = block * GAUSSIAN_ROWS_PER_LINE;
      uint32_t end = begin + GAUSSIAN_ROWS_PER_LINE;
      if (begin < first) begin = first;
      if (end > GAUSSIAN_SIZE) end = GAUSSIAN_SIZE;
      if (begin >= end) continue;
      const uint32_t cells = (end - begin) * columns;
      for (uint32_t item = tid; item < cells; item += threads_per_block) {
        const uint32_t row = begin + item / columns;
        const uint32_t col = t + item % columns;
        const float factor = m[row * GAUSSIAN_SIZE + t];
        a[row * GAUSSIAN_SIZE + col] =
            a[row * GAUSSIAN_SIZE + col] - factor * a[t * GAUSSIAN_SIZE + col];
        if (col == t) {
          b[row] = b[row] - factor * b[t];
          stage_owners[row] = block_id;
        }
      }
    }
  }
  nightly_kernel_end(tid, threads_per_block);
}

int main() {
  nightly_wait_launch();
  const uint32_t argument = nightly_pb_get(NIGHTLY_PB_ARG);
  const uint32_t stage = argument & ~GAUSSIAN_MODEL_STAGE_FLAG;
  if (stage >= GAUSSIAN_STAGES) return 1;
  mu_schedule(gaussian_phase, nullptr, NIGHTLY_OCC);
  return 0;
}
