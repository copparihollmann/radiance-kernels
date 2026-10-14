/* Host for a fixed model-chain workload on the nightly HBM launch protocol. */
/* The generated schedule runs on core 0; other cores park after launch. */
#define NIGHTLY_CORES_PER_CLUSTER 1
#include <nightly/host.h>
#include "model_hbm_pb.h"

#define MODEL_HBM_PASS 0x600d0000u
#ifndef MODEL_HBM_CLUSTERS
#define MODEL_HBM_CLUSTERS 1u
#endif

static int model_hbm_report(uint32_t cluster) {
  const uint64_t start = ((uint64_t)nightly_pb_read(cluster, MODEL_HBM_PB_START_HIGH) << 32) |
                         nightly_pb_read(cluster, NIGHTLY_PB_START);
  const uint64_t end = ((uint64_t)nightly_pb_read(cluster, MODEL_HBM_PB_END_HIGH) << 32) |
                       nightly_pb_read(cluster, NIGHTLY_PB_END);
  if (MODEL_HBM_CLUSTERS > 1) { nh_puts("cluster="); nh_putu(cluster); nh_puts("\n"); }
  nh_puts("gpu_start_cycle64="); nh_putu(start); nh_puts("\n");
  nh_puts("gpu_end_cycle64="); nh_putu(end); nh_puts("\n");
  if (end <= start) {
    nh_puts("kernel_cycles64=INVALID\n");
    return 0;
  }
  nh_puts("kernel_cycles64="); nh_putu(end - start); nh_puts("\n");
  return 1;
}

int main() {
  if (MODEL_HBM_CLUSTERS == 1) nh_puts("model_chain_hbm: launch cluster 0\n");
  else { nh_puts("model_chain_hbm: launch clusters="); nh_putu(MODEL_HBM_CLUSTERS); nh_puts("\n"); }
  nh_flush();
  for (uint32_t cl = 0; cl < MODEL_HBM_CLUSTERS; ++cl) {
    nightly_pb_write(cl, MODEL_HBM_PB_START_HIGH, 0);
    nightly_pb_write(cl, MODEL_HBM_PB_END_HIGH, 0);
  }
  nightly_host_launch(MODEL_HBM_CLUSTERS);
  const int finished = nightly_host_wait(MODEL_HBM_CLUSTERS, UINT64_MAX);
  int pass = finished;
  for (uint32_t cl = 0; cl < MODEL_HBM_CLUSTERS; ++cl) {
    const uint32_t status = nightly_pb_read(cl, NIGHTLY_PB_STATUS);
    if (MODEL_HBM_CLUSTERS > 1) { nh_puts("cluster_status="); nh_putu(cl); nh_puts(":"); }
    else nh_puts("model_status=");
    nh_putx(status, 8); nh_puts("\n");
    if (!model_hbm_report(cl) || status != MODEL_HBM_PASS) pass = 0;
  }
  nh_puts(pass ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(pass ? 0 : 1);
}
