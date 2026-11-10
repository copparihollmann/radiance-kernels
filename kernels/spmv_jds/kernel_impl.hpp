// Parboil spmv on Muon: JDS-format sparse matrix-vector product.
//
// SPMV_JDS_ILP is the unroll factor; kernel.cpp builds the ILP=1 baseline and
// kernel_ilp{2,4}.cpp the unrolled variants, the same arrangement as
// kernels/vecadd.  Each unrolled slot is an independent row, so one row's
// gather through spmv_jds_indices overlaps the next instead of serializing
// behind it -- which matters here, because the whole kernel is a chain of
// dependent loads.
//
// Row ownership is NOT one of mu_partition.h's policies: the boundaries come
// from the generator (SPMV_SM_BOUNDARIES), which balances nonzeros rather than
// rows, so the two arms below stay explicit.
#ifndef SPMV_JDS_KERNEL_IMPL_HPP
#define SPMV_JDS_KERNEL_IMPL_HPP

#include <stdint.h>

#include "generated/config.h"
#include <mu_partition.h>
#include <nightly/device.h>

// hpc.mk picks NIGHTLY_CLUSTERS from the preset; HPC_SM_COUNT is the cluster
// count the generated data was partitioned for.  A mismatch means the ELF
// would read someone else's share, so fail the build rather than the run.
static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "preset cluster count must match the generated data");

#ifndef SPMV_JDS_ILP
#define SPMV_JDS_ILP 1
#endif
static_assert(SPMV_JDS_WARP_SIZE == 32u,
              "the pinned Parboil JDS layout uses groups of 32 rows");

extern "C" {
extern __global const float spmv_jds_data[];
extern __global const uint32_t spmv_jds_indices[];
extern __global const uint32_t spmv_jds_row_to_jds[];
extern __global const float spmv_jds_x[];
extern __global const uint32_t spmv_jds_ptr[];
extern __global const uint32_t spmv_jds_group_bounds[];
extern __global float spmv_jds_output[];
}

static constexpr uint32_t kBoundaries[HPC_SM_COUNT + 1u] =
    SPMV_SM_BOUNDARIES;

#pragma clang fp contract(off)
template <uint32_t ILP>
static __attribute__((always_inline)) void process_rows(
    uint32_t begin, uint32_t end, uint32_t tid,
    uint32_t threads_per_block) {
  // The upstream kernel enumerates JDS rows and scatters through perm[ix].
  // Inverting that permutation lets each physical SM own complete 64-byte
  // output lines without changing any row's JDS index or accumulation order.
  mu_grid_stride<ILP>(begin, end, tid, threads_per_block, [&](uint32_t row) {
    NIGHTLY_LANE_SAMPLE();
    const uint32_t jds_row = spmv_jds_row_to_jds[row];
    const uint32_t bound =
        spmv_jds_group_bounds[jds_row / SPMV_JDS_WARP_SIZE];
    float sum = 0.0f;
    for (uint32_t k = 0; k < bound; ++k) {
      const uint32_t j = spmv_jds_ptr[k] + jds_row;
      const uint32_t in = spmv_jds_indices[j];
      const float d = spmv_jds_data[j];
      const float t = spmv_jds_x[in];
      sum = sum + d * t;
    }
    spmv_jds_output[row] = sum;
  });
}

template <uint32_t ILP>
static inline void spmv_jds_kernel(uint32_t tid, uint32_t threads_per_block,
                                   uint32_t block_id) {
#if SPMV_BLOCK_CYCLIC_ROWS
  constexpr uint32_t chunk_rows = SPMV_BLOCK_CYCLIC_ROWS;
  constexpr uint32_t chunks =
      (SPMV_DIMENSION + chunk_rows - 1u) / chunk_rows;
  for (uint32_t chunk = block_id; chunk < chunks;
       chunk += NIGHTLY_CLUSTERS) {
    const uint32_t begin = chunk * chunk_rows;
    const uint32_t candidate_end = begin + chunk_rows;
    const uint32_t end = candidate_end < SPMV_DIMENSION
        ? candidate_end : SPMV_DIMENSION;
    process_rows<ILP>(begin, end, tid, threads_per_block);
  }
#else
  process_rows<ILP>(kBoundaries[block_id], kBoundaries[block_id + 1u],
                    tid, threads_per_block);
#endif
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  spmv_jds_kernel<SPMV_JDS_ILP>(tid, threads_per_block, block_id);
  nightly_kernel_end(tid, threads_per_block);
}

int main() { return nightly_main(kernel_body, nullptr); }

#endif
