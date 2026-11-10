// BabelStream Dot on Muon: a strided per-thread partial, then a shared-memory
// tree, then one partial per physical SM for the host to combine.
//
// STREAM_DOT_ILP is the unroll factor; kernel.cpp builds the ILP=1 baseline
// and kernel_ilp{2,4}.cpp the unrolled variants, the same arrangement as
// kernels/vecadd.  Unrolling here gives the cluster several independent loads
// in flight; it deliberately keeps ONE accumulator, because splitting the sum
// across ILP accumulators would reassociate the floating-point addition and
// change the result.
#ifndef STREAM_DOT_KERNEL_IMPL_HPP
#define STREAM_DOT_KERNEL_IMPL_HPP

#include <stdint.h>

#include "generated/config.h"
#include <mu_partition.h>
#include <nightly/device.h>
#include <shared_mem.h>

// hpc.mk picks NIGHTLY_CLUSTERS from the preset; HPC_SM_COUNT is the cluster
// count the generated data was partitioned for.  A mismatch means the ELF
// would read someone else's share, so fail the build rather than the run.
static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "preset cluster count must match the generated data");

#ifndef STREAM_DOT_ILP
#define STREAM_DOT_ILP 1
#endif

extern "C" {
extern __global const float stream_dot_a[];
extern __global const float stream_dot_b[];
extern __global float stream_dot_partial[];
}

template <uint32_t ILP>
static inline void stream_dot_kernel(uint32_t tid, uint32_t threads_per_block,
                                     uint32_t block_id) {
  // Round-robin chunks stagger the physical SMs over different DRAM banks;
  // contiguous gives each SM one dense run.  Either way a 64-byte line belongs
  // to exactly one SM, and the accumulation order within a thread is the same.
#if STREAM_DOT_BLOCK_CYCLIC_CHUNK_ELEMENTS
  constexpr MuOwnership kPolicy = MuOwnership::kCyclicChunks;
  constexpr uint32_t kChunkLines =
      STREAM_DOT_BLOCK_CYCLIC_CHUNK_ELEMENTS / 16u;
#else
  constexpr MuOwnership kPolicy = MuOwnership::kContiguous;
  constexpr uint32_t kChunkLines = 1u;
#endif
  const MuOwnedSpace owned = MuOwnedSpace::of<kPolicy, 16u, kChunkLines>(
      STREAM_DOT_ELEMENTS, NIGHTLY_CLUSTERS, block_id);
  float sum = 0.0f;
  mu_grid_stride<ILP>(owned, tid, threads_per_block, [&](uint32_t i) {
    NIGHTLY_LANE_SAMPLE();
    sum += stream_dot_a[i] * stream_dot_b[i];
  });

  // Match BabelStream's CUDA reduction structure: every thread first forms a
  // strided partial, then the whole block performs a power-of-two tree in
  // shared memory. The host combines one partial per physical SM, just as the
  // upstream CUDA host combines one partial per CUDA block.
  volatile __shared float* scratch =
      reinterpret_cast<volatile __shared float*>(0x0);
  scratch[tid] = sum;
  mu_fence_smem();
  for (uint32_t offset = threads_per_block / 2u; offset > 0u; offset /= 2u) {
    mu_barrier(1u, threads_per_block / MU_NUM_THREADS);
    if (tid < offset) scratch[tid] += scratch[tid + offset];
    mu_fence_smem();
  }
  mu_barrier(1u, threads_per_block / MU_NUM_THREADS);
  // stream_dot_partial is padded to one 64-byte line per SM, so this single
  // naturally aligned 32-bit store is the only write to its line.  The packed
  // float-per-SM array it replaces put every SM in one line.
  if (tid == 0u)
    stream_dot_partial[block_id * STREAM_DOT_PARTIAL_STRIDE_FLOATS] =
        scratch[0];
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  stream_dot_kernel<STREAM_DOT_ILP>(tid, threads_per_block, block_id);
  nightly_kernel_end(tid, threads_per_block);
}

int main() { return nightly_main(kernel_body, nullptr); }

#endif
