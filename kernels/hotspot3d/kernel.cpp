#include <stdint.h>

#include "generated/config.h"
#include <mu_partition.h>
#include <nightly/device.h>

static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "host and device cluster counts must match");
static_assert(HOTSPOT3D_COLUMNS % HOTSPOT3D_COLUMNS_PER_LINE == 0u,
              "every z-plane must contain complete cache lines");
static_assert(HOTSPOT3D_COLUMNS_PER_LINE * sizeof(float) == 64u,
              "partition granule must be exactly one cache line");

extern "C" {
extern __global const float hotspot3d_power[];
extern __global const float hotspot3d_initial[];
extern __global const float hotspot3d_model_inputs[];
extern __global float hotspot3d_output[];
extern __global float hotspot3d_scratch[];
extern __global uint32_t hotspot3d_owners[];
}

#pragma clang fp contract(off)
static inline float update_cell(const __global float* input,
                                uint32_t center, uint32_t west,
                                uint32_t east, uint32_t north,
                                uint32_t south, float bottom, float self,
                                float top) {
  float value = HOTSPOT3D_CC * self;
  value = value + HOTSPOT3D_CW * input[west];
  value = value + HOTSPOT3D_CE * input[east];
  value = value + HOTSPOT3D_CS * input[south];
  value = value + HOTSPOT3D_CN * input[north];
  value = value + HOTSPOT3D_CB * bottom;
  value = value + HOTSPOT3D_CT * top;
  value = value + HOTSPOT3D_STEP_DIV_CAP * hotspot3d_power[center];
  value = value + HOTSPOT3D_CT * HOTSPOT3D_AMBIENT;
  return value;
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  const uint32_t argument = nightly_pb_get(NIGHTLY_PB_ARG);
  const bool model_stage = (argument & HOTSPOT3D_MODEL_STAGE_FLAG) != 0u;
  const uint32_t iteration = argument & ~HOTSPOT3D_MODEL_STAGE_FLAG;
  const bool write_final =
      ((HOTSPOT3D_ITERATIONS - iteration) & 1u) != 0u;
  const __global float* input = model_stage
      ? hotspot3d_model_inputs + iteration * HOTSPOT3D_ELEMENTS
      : (iteration == 0u
          ? hotspot3d_initial
          : (write_final ? hotspot3d_scratch : hotspot3d_output));
  __global float* output = model_stage
      ? hotspot3d_output
      : (write_final ? hotspot3d_output : hotspot3d_scratch);

  // The OpenCL kernel assigns one work-item to each (x,y) column and sweeps
  // z.  Grouping 16 adjacent columns gives every SM exclusive ownership of
  // complete float32 cache lines in every plane.
  const MuPartition columns = mu_contiguous_partition(
      HOTSPOT3D_COLUMNS, NIGHTLY_CLUSTERS, block_id,
      HOTSPOT3D_COLUMNS_PER_LINE);
  for (uint32_t column = columns.begin + tid; column < columns.end;
       column += threads_per_block) {
    const uint32_t x = column % HOTSPOT3D_NX;
    const uint32_t y = column / HOTSPOT3D_NX;
    uint32_t center = column;
    uint32_t west = x == 0u ? center : center - 1u;
    uint32_t east = x + 1u == HOTSPOT3D_NX ? center : center + 1u;
    uint32_t north = y == 0u ? center : center - HOTSPOT3D_NX;
    uint32_t south = y + 1u == HOTSPOT3D_NY
        ? center : center + HOTSPOT3D_NX;

    float bottom = input[center];
    float self = bottom;
    float top = input[center + HOTSPOT3D_COLUMNS];
    output[center] = update_cell(
        input, center, west, east, north, south, bottom, self, top);

    center += HOTSPOT3D_COLUMNS;
    west += HOTSPOT3D_COLUMNS;
    east += HOTSPOT3D_COLUMNS;
    north += HOTSPOT3D_COLUMNS;
    south += HOTSPOT3D_COLUMNS;
    for (uint32_t z = 1u; z + 1u < HOTSPOT3D_NZ; ++z) {
      bottom = self;
      self = top;
      top = input[center + HOTSPOT3D_COLUMNS];
      output[center] = update_cell(
          input, center, west, east, north, south, bottom, self, top);
      center += HOTSPOT3D_COLUMNS;
      west += HOTSPOT3D_COLUMNS;
      east += HOTSPOT3D_COLUMNS;
      north += HOTSPOT3D_COLUMNS;
      south += HOTSPOT3D_COLUMNS;
    }
    bottom = self;
    self = top;
    output[center] = update_cell(
        input, center, west, east, north, south, bottom, self, self);

    if (model_stage || iteration + 1u == HOTSPOT3D_ITERATIONS)
      hotspot3d_owners[column] = block_id;
  }
  nightly_kernel_end(tid, threads_per_block);
}

int main() { return nightly_main(kernel_body, nullptr); }
