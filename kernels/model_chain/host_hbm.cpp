/* Host for a fixed model-chain workload on the nightly HBM launch protocol. */
/* The generated schedule runs on core 0; other cores park after launch. */
#define NIGHTLY_CORES_PER_CLUSTER 1
#include <nightly/host.h>
#include "model_hbm_pb.h"

#define MODEL_HBM_PASS 0x600d0000u

static int model_hbm_report(void) {
  const uint64_t start = ((uint64_t)nightly_pb_read(0, MODEL_HBM_PB_START_HIGH) << 32) |
                         nightly_pb_read(0, NIGHTLY_PB_START);
  const uint64_t end = ((uint64_t)nightly_pb_read(0, MODEL_HBM_PB_END_HIGH) << 32) |
                       nightly_pb_read(0, NIGHTLY_PB_END);
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
  nh_puts("model_chain_hbm: launch cluster 0\n");
  nh_flush();
  nightly_pb_write(0, MODEL_HBM_PB_START_HIGH, 0);
  nightly_pb_write(0, MODEL_HBM_PB_END_HIGH, 0);
  nightly_host_launch(1);
  const int finished = nightly_host_wait(1, UINT64_MAX);
  const uint32_t status = nightly_pb_read(0, NIGHTLY_PB_STATUS);
  nh_puts("model_status="); nh_putx(status, 8); nh_puts("\n");
  const int valid_time = finished && model_hbm_report();
  const int pass = valid_time && status == MODEL_HBM_PASS;
  nh_puts(pass ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(pass ? 0 : 1);
}
