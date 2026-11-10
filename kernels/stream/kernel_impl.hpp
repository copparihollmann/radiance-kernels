// STREAM on Muon: Copy, Scale, Add and Triad, one operation per ELF.
//
// STREAM_ILP is the unroll factor.  kernel.cpp builds the ILP=1 baseline and
// kernel_ilp{2,4}.cpp the unrolled variants, the same arrangement as
// kernels/vecadd.  Unrolling here is not about issuing wider loads -- Muon has no
// vector instructions at all -- it is about having several independent loads in
// flight at once.  These kernels are limited by how many line fills a cluster
// can keep outstanding, not by arithmetic, so each unrolled step issues its
// loads back to back, then does the arithmetic, then the stores, rather than
// walking one element at a time through a load-use-store chain.
//
// Ownership of the index space comes from MuOwnedSpace (mu_partition.h), which
// hands each cluster whole 64-byte lines under whichever policy the generator
// selected.  The kernel never names a cluster count.
#ifndef STREAM_KERNEL_IMPL_HPP
#define STREAM_KERNEL_IMPL_HPP

#include <stdint.h>

#include "generated/config.h"
#include <mu_partition.h>
#include <nightly/device.h>

// hpc.mk picks NIGHTLY_CLUSTERS from the preset; HPC_SM_COUNT is the cluster
// count the generated data was partitioned for.  A mismatch means the ELF would
// read someone else's share, so fail the build rather than the run.
static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "preset cluster count must match the generated data");

#ifndef STREAM_ILP
#define STREAM_ILP 1
#endif

extern "C" {
extern __global float stream_a[];
extern __global float stream_b[];
extern __global float stream_c[];
}

// float32: 16 elements per 64-byte line.
static constexpr uint32_t kStreamLineElements = 16u;

#if STREAM_BLOCK_CYCLIC_CHUNK_LINES
static constexpr MuOwnership kStreamOwnership = MuOwnership::kCyclicChunks;
static constexpr uint32_t kStreamChunkLines = STREAM_BLOCK_CYCLIC_CHUNK_LINES;
#elif STREAM_BLOCK_CYCLIC_LINES
static constexpr MuOwnership kStreamOwnership = MuOwnership::kCyclicLines;
static constexpr uint32_t kStreamChunkLines = 1u;
#else
static constexpr MuOwnership kStreamOwnership = MuOwnership::kContiguous;
static constexpr uint32_t kStreamChunkLines = 1u;
#endif

// One element, used for the loop tail.
static inline void stream_element(uint32_t i) {
#if STREAM_KIND == 0
  stream_c[i] = stream_a[i];
#elif STREAM_KIND == 1
  stream_b[i] = STREAM_SCALAR * stream_c[i];
#elif STREAM_KIND == 2
  stream_c[i] = stream_a[i] + stream_b[i];
#elif STREAM_KIND == 3
  stream_a[i] = stream_b[i] + STREAM_SCALAR * stream_c[i];
#endif
}

// ILP elements: every load issued before the first use, every store after the
// last arithmetic op, with one independent temporary per unrolled slot.
template <uint32_t ILP>
static inline void stream_step(const uint32_t (&i)[ILP]) {
#if STREAM_KIND == 0            // Copy: c = a
  float a[ILP];
#pragma unroll
  for (uint32_t u = 0; u < ILP; ++u) a[u] = stream_a[i[u]];
#pragma unroll
  for (uint32_t u = 0; u < ILP; ++u) stream_c[i[u]] = a[u];
#elif STREAM_KIND == 1          // Scale: b = scalar * c
  float c[ILP];
#pragma unroll
  for (uint32_t u = 0; u < ILP; ++u) c[u] = stream_c[i[u]];
#pragma unroll
  for (uint32_t u = 0; u < ILP; ++u) c[u] = STREAM_SCALAR * c[u];
#pragma unroll
  for (uint32_t u = 0; u < ILP; ++u) stream_b[i[u]] = c[u];
#elif STREAM_KIND == 2          // Add: c = a + b
  float a[ILP], b[ILP];
#pragma unroll
  for (uint32_t u = 0; u < ILP; ++u) a[u] = stream_a[i[u]];
#pragma unroll
  for (uint32_t u = 0; u < ILP; ++u) b[u] = stream_b[i[u]];
#pragma unroll
  for (uint32_t u = 0; u < ILP; ++u) a[u] = a[u] + b[u];
#pragma unroll
  for (uint32_t u = 0; u < ILP; ++u) stream_c[i[u]] = a[u];
#elif STREAM_KIND == 3          // Triad: a = b + scalar * c
  float b[ILP], c[ILP];
#pragma unroll
  for (uint32_t u = 0; u < ILP; ++u) b[u] = stream_b[i[u]];
#pragma unroll
  for (uint32_t u = 0; u < ILP; ++u) c[u] = stream_c[i[u]];
#pragma unroll
  for (uint32_t u = 0; u < ILP; ++u) b[u] = b[u] + STREAM_SCALAR * c[u];
#pragma unroll
  for (uint32_t u = 0; u < ILP; ++u) stream_a[i[u]] = b[u];
#endif
}

template <uint32_t ILP>
static inline void stream_kernel(uint32_t tid, uint32_t threads_per_block,
                                 uint32_t block_id) {
  static_assert(ILP > 0u, "ILP must be at least 1");

  const MuOwnedSpace owned =
      MuOwnedSpace::of<kStreamOwnership, kStreamLineElements,
                       kStreamChunkLines>(STREAM_ELEMENTS, NIGHTLY_CLUSTERS,
                                          block_id);

  const uint32_t step = threads_per_block * ILP;
  const uint32_t span = threads_per_block * (ILP - 1u);
  uint32_t v = tid;
  for (; v + span < owned.count; v += step) {
    uint32_t i[ILP];
#pragma unroll
    for (uint32_t u = 0; u < ILP; ++u) {
      NIGHTLY_LANE_SAMPLE();
      i[u] = owned[v + u * threads_per_block];
    }
    stream_step<ILP>(i);
  }
  // Tail: fewer than ILP full steps left.
  for (; v < owned.count; v += threads_per_block) {
    NIGHTLY_LANE_SAMPLE();
    stream_element(owned[v]);
  }
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  stream_kernel<STREAM_ILP>(tid, threads_per_block, block_id);
  nightly_kernel_end(tid, threads_per_block);
}

int main() { return nightly_main(kernel_body, nullptr); }

#endif
