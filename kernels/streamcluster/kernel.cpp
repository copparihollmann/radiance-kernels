#include <stdint.h>

#include "generated/config.h"
#include <nightly/device.h>

#include "kernel_impl.hpp"

static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "host and device cluster counts must match");

extern "C" {
extern __global const StreamClusterPoint streamcluster_points[];
extern __global const float streamcluster_coordinates[];
extern __global const int32_t streamcluster_center_table[];
extern __global float streamcluster_work_output[];
extern __global uint8_t streamcluster_switch_output[];
}

static StreamClusterArgs streamcluster_args = {
    .points = streamcluster_points,
    .coordinates = streamcluster_coordinates,
    .center_table = streamcluster_center_table,
    .work = streamcluster_work_output,
    .switches = streamcluster_switch_output,
};

static void kernel_body(void* opaque, uint32_t tid,
                        uint32_t threads_per_threadblock,
                        uint32_t physical_threadblock_id) {
  streamcluster_pgain(
      reinterpret_cast<StreamClusterArgs*>(opaque), tid,
      threads_per_threadblock, physical_threadblock_id);
  nightly_kernel_end(tid, threads_per_threadblock);
}

int main() { return nightly_main(kernel_body, &streamcluster_args); }
