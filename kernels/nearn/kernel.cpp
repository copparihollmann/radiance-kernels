#include <math.h>
#include <stdint.h>

#include "generated/config.h"
#include <mu_partition.h>
#include <nightly/device.h>

static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "host and device cluster counts must match");

struct LatLong {
  float lat;
  float lng;
};

extern "C" {
extern __global LatLong nearn_locations[];
extern __global float nearn_distances[];
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  // Preserve the original one-distance-per-record equation. Complete groups
  // of 16 float outputs (one 64-byte line) belong to one physical SM.
  const MuPartition range = mu_contiguous_partition(
      NEARN_RECORDS, NIGHTLY_CLUSTERS, block_id, 16u);
  for (uint32_t i = range.begin + tid; i < range.end;
       i += threads_per_block) {
    const float dx = NEARN_QUERY_LAT - nearn_locations[i].lat;
    const float dy = NEARN_QUERY_LNG - nearn_locations[i].lng;
    nearn_distances[i] = sqrtf(dx * dx + dy * dy);
  }
  nightly_kernel_end(tid, threads_per_block);
}

int main() { return nightly_main(kernel_body, nullptr); }
