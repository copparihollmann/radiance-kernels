#include <stdint.h>

#include "generated/config.h"
#include <mu_partition.h>
#include <nightly/device.h>

static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "host and device cluster counts must match");

extern "C" {
extern __global float stencil3d_input[];
extern __global float stencil3d_output[];
}

static inline uint32_t clamp_coord(int32_t value) {
  if (value < 0) return 0;
  if (value >= static_cast<int32_t>(STENCIL3D_SIZE)) return STENCIL3D_SIZE - 1u;
  return static_cast<uint32_t>(value);
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  // Preserve the Vortex stencil3d equation and boundary replication while
  // assigning complete 64-byte output lines to one physical SM.
  const MuPartition range = mu_contiguous_partition(
      STENCIL3D_ELEMENTS, NIGHTLY_CLUSTERS, block_id, 16u);
  const uint32_t plane = STENCIL3D_SIZE * STENCIL3D_SIZE;
  for (uint32_t linear = range.begin + tid; linear < range.end;
       linear += threads_per_block) {
    const uint32_t z = linear / plane;
    const uint32_t rem = linear - z * plane;
    const uint32_t y = rem / STENCIL3D_SIZE;
    const uint32_t x = rem - y * STENCIL3D_SIZE;
    float sum = 0.0f;
    for (int32_t dz = -1; dz <= 1; ++dz) {
      const uint32_t nz = clamp_coord(static_cast<int32_t>(z) + dz);
      for (int32_t dy = -1; dy <= 1; ++dy) {
        const uint32_t ny = clamp_coord(static_cast<int32_t>(y) + dy);
        for (int32_t dx = -1; dx <= 1; ++dx) {
          const uint32_t nx = clamp_coord(static_cast<int32_t>(x) + dx);
          sum += stencil3d_input[nz * plane + ny * STENCIL3D_SIZE + nx];
        }
      }
    }
    stencil3d_output[linear] = sum / 27;
  }
  nightly_kernel_end(tid, threads_per_block);
}

int main() { return nightly_main(kernel_body, nullptr); }
