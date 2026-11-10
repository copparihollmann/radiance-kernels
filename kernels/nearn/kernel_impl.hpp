// Rodinia nearest-neighbour on Muon: one Euclidean distance per record.
//
// NEARN_ILP is the unroll factor; kernel.cpp builds the ILP=1 baseline and
// kernel_ilp{2,4}.cpp the unrolled variants, the same arrangement as
// kernels/vecadd.  The body is two loads, three FLOPs and a store, so the only
// thing unrolling buys is more loads in flight -- which is the whole point
// here, since a cluster runs out of outstanding line fills long before it runs
// out of arithmetic.
#ifndef NEARN_KERNEL_IMPL_HPP
#define NEARN_KERNEL_IMPL_HPP

#include <math.h>
#include <stdint.h>

#include "generated/config.h"
#include <mu_partition.h>
#include <nightly/device.h>

// hpc.mk picks NIGHTLY_CLUSTERS from the preset; HPC_SM_COUNT is the cluster
// count the generated data was partitioned for.  A mismatch means the ELF would
// read someone else's share, so fail the build rather than the run.
static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "preset cluster count must match the generated data");

#ifndef NEARN_ILP
#define NEARN_ILP 1
#endif

struct LatLong {
  float lat;
  float lng;
};

extern "C" {
extern __global LatLong nearn_locations[];
extern __global float nearn_distances[];
}

template <uint32_t ILP>
static inline void nearn_kernel(uint32_t tid, uint32_t threads_per_block,
                                uint32_t block_id) {
  static_assert(ILP > 0u, "ILP must be at least 1");
  // Preserve the original one-distance-per-record equation.  Complete groups
  // of 16 float outputs (one 64-byte line) belong to one physical SM.
  const MuOwnedSpace owned =
      MuOwnedSpace::of<MuOwnership::kContiguous, 16u>(
          NEARN_RECORDS, NIGHTLY_CLUSTERS, block_id);

  const uint32_t step = threads_per_block * ILP;
  const uint32_t span = threads_per_block * (ILP - 1u);
  uint32_t v = tid;
  for (; v + span < owned.count; v += step) {
    uint32_t i[ILP];
    float dx[ILP], dy[ILP];
#pragma unroll
    for (uint32_t u = 0; u < ILP; ++u) {
      NIGHTLY_LANE_SAMPLE();
      i[u] = owned[v + u * threads_per_block];
    }
#pragma unroll
    for (uint32_t u = 0; u < ILP; ++u) {
      dx[u] = NEARN_QUERY_LAT - nearn_locations[i[u]].lat;
      dy[u] = NEARN_QUERY_LNG - nearn_locations[i[u]].lng;
    }
#pragma unroll
    for (uint32_t u = 0; u < ILP; ++u)
      nearn_distances[i[u]] = sqrtf(dx[u] * dx[u] + dy[u] * dy[u]);
  }
  for (; v < owned.count; v += threads_per_block) {
    NIGHTLY_LANE_SAMPLE();
    const uint32_t i = owned[v];
    const float dx = NEARN_QUERY_LAT - nearn_locations[i].lat;
    const float dy = NEARN_QUERY_LNG - nearn_locations[i].lng;
    nearn_distances[i] = sqrtf(dx * dx + dy * dy);
  }
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  nearn_kernel<NEARN_ILP>(tid, threads_per_block, block_id);
  nightly_kernel_end(tid, threads_per_block);
}

int main() { return nightly_main(kernel_body, nullptr); }

#endif
