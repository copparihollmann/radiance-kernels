// GEMV on Muon: one warp per output row, a BF16 dot product reduced in shared
// memory.
//
// GEMV_ILP is the unroll factor; kernel.cpp builds the ILP=1 baseline and
// kernel_ilp{2,4}.cpp the unrolled variants, the same arrangement as
// kernels/vecadd.  It used to come from the generated config.h, which meant
// one ELF per ILP per data set and no way to compare two unrolls of the same
// data.  GEMV_UNROLL stays in config.h: it is how far the row loop unrolls,
// and must be a multiple of GEMV_ILP, which is how many independent
// accumulator chains that unroll is split across.
#ifndef GEMV_KERNEL_IMPL_HPP
#define GEMV_KERNEL_IMPL_HPP

#include <stdint.h>

#include "generated/config.h"
#include <mu_partition.h>
#include <nightly/device.h>

// hpc.mk picks NIGHTLY_CLUSTERS from the preset; HPC_SM_COUNT is the cluster
// count the generated data was partitioned for.  A mismatch means the ELF
// would read someone else's share, so fail the build rather than the run.
static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "preset cluster count must match the generated data");

#ifndef GEMV_ILP
#define GEMV_ILP 1
#endif
static_assert(GEMV_ILP > 0u && GEMV_UNROLL > 0u);
static_assert((GEMV_UNROLL % GEMV_ILP) == 0u,
              "the row unroll must divide evenly across the accumulator chains");
static_assert((MU_NUM_THREADS & (MU_NUM_THREADS - 1u)) == 0u);

extern "C" {
extern __global uint16_t gemv_a[];
extern __global uint16_t gemv_x[];
extern __global uint16_t gemv_y[];
}

__shared _Float16* const gemv_sdata =
    reinterpret_cast<__shared _Float16*>(0x0);

template <uint32_t MaxStride>
static inline void reduce(uint32_t tid, uint32_t lane) {
  for (uint32_t stride = 2; stride <= MaxStride; stride *= 2) {
    if ((lane % stride) == 0u)
      gemv_sdata[tid] = gemv_sdata[tid] +
                        gemv_sdata[tid + (stride >> 1u)];
  }
}

template <uint32_t Ilp>
static inline void keep_accum_ordered(_Float16 (&accum)[Ilp]) {
#pragma unroll
  for (uint32_t i = 0; i < Ilp; ++i)
    asm volatile("" : "+r"(accum[i]) :: "memory");
}

template <uint32_t ILP>
static inline void gemv_kernel(uint32_t tid, uint32_t threads_per_block,
                               uint32_t block_id) {
  constexpr uint32_t width = MU_NUM_THREADS;
  const uint32_t lane = tid % width;
  const uint32_t warp = tid / width;
  const uint32_t warps_per_block = threads_per_block / width;
  const MuPartition owned = mu_contiguous_partition(
      GEMV_ROWS, NIGHTLY_CLUSTERS, block_id);
  __global _Float16* const a =
      reinterpret_cast<__global _Float16*>(gemv_a);
  __global _Float16* const x =
      reinterpret_cast<__global _Float16*>(gemv_x);
  __global _Float16* const y =
      reinterpret_cast<__global _Float16*>(gemv_y);

  for (uint32_t row = owned.begin + warp; row < owned.end;
       row += warps_per_block) {
    _Float16 accum[ILP] = {};
    const uint32_t unroll_stride = width * GEMV_UNROLL;
    const uint32_t unroll_span = width * (GEMV_UNROLL - 1u);
    const __global _Float16* const row_a = a + row * GEMV_COLS;
    for (uint32_t base = lane; base + unroll_span < GEMV_COLS;
         base += unroll_stride) {
#pragma unroll
      for (uint32_t group = 0; group < GEMV_UNROLL; group += ILP) {
#pragma unroll
        for (uint32_t chain = 0; chain < ILP; ++chain) {
          const uint32_t index = base + (group + chain) * width;
          accum[chain] += row_a[index] * x[index];
        }
        keep_accum_ordered(accum);
      }
    }
    _Float16 sum = accum[0];
#pragma unroll
    for (uint32_t chain = 1; chain < ILP; ++chain)
      sum += accum[chain];
    gemv_sdata[tid] = sum;
    reduce<width>(tid, lane);
    if (lane == 0u) y[row] = gemv_sdata[tid];
  }
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  gemv_kernel<GEMV_ILP>(tid, threads_per_block, block_id);
  nightly_kernel_end(tid, threads_per_block);
}

int main() { return nightly_main(kernel_body, nullptr); }

#endif
