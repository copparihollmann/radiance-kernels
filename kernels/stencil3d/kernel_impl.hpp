// Vortex stencil3d on Muon: a 27-point box filter with boundary replication.
//
// STENCIL3D_ILP is the unroll factor; kernel.cpp builds the ILP=1 baseline and
// kernel_ilp{2,4}.cpp the unrolled variants, the same arrangement as
// kernels/vecadd.  Each unrolled slot is an independent output cell, so the
// 27 loads of one cell overlap the 27 of the next instead of waiting for them.
#ifndef STENCIL3D_KERNEL_IMPL_HPP
#define STENCIL3D_KERNEL_IMPL_HPP

#include <stdint.h>

#include "generated/config.h"
#include <mu_partition.h>
#include <nightly/device.h>

// hpc.mk picks NIGHTLY_CLUSTERS from the preset; HPC_SM_COUNT is the cluster
// count the generated data was partitioned for.  A mismatch means the ELF would
// read someone else's share, so fail the build rather than the run.
static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "preset cluster count must match the generated data");

#ifndef STENCIL3D_ILP
#define STENCIL3D_ILP 1
#endif

extern "C" {
extern __global float stencil3d_input[];
extern __global float stencil3d_output[];
}

static inline uint32_t clamp_coord(int32_t value) {
  if (value < 0) return 0;
  if (value >= static_cast<int32_t>(STENCIL3D_SIZE)) return STENCIL3D_SIZE - 1u;
  return static_cast<uint32_t>(value);
}

static inline void stencil3d_cell(uint32_t linear, uint32_t plane) {
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

template <uint32_t ILP>
static inline void stencil3d_kernel(uint32_t tid, uint32_t threads_per_block,
                                    uint32_t block_id) {
  // Preserve the Vortex stencil3d equation and boundary replication while
  // assigning complete 64-byte output lines to one physical SM.
  const MuOwnedSpace owned =
      MuOwnedSpace::of<MuOwnership::kContiguous, 16u>(
          STENCIL3D_ELEMENTS, NIGHTLY_CLUSTERS, block_id);
  const uint32_t plane = STENCIL3D_SIZE * STENCIL3D_SIZE;
  mu_grid_stride<ILP>(owned, tid, threads_per_block, [&](uint32_t linear) {
    NIGHTLY_LANE_SAMPLE();
    stencil3d_cell(linear, plane);
  });
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  stencil3d_kernel<STENCIL3D_ILP>(tid, threads_per_block, block_id);
  nightly_kernel_end(tid, threads_per_block);
}

int main() { return nightly_main(kernel_body, nullptr); }

#endif
