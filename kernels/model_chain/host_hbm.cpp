/* Host for a fixed model-chain workload on the nightly HBM launch protocol. */
/* The generated schedule runs on core 0; other cores park after launch. */
#define NIGHTLY_CORES_PER_CLUSTER 1
#include <nightly/host.h>

#define MODEL_HBM_PASS 0x600d0000u

int main() {
  nh_puts("model_chain_hbm: launch cluster 0\n");
  nh_flush();
  nightly_host_launch(1);
  const int finished = nightly_host_wait(1, UINT64_MAX);
  const uint32_t status = nightly_pb_read(0, NIGHTLY_PB_STATUS);
  nh_puts("model_status="); nh_putx(status, 8); nh_puts("\n");
  if (finished) nightly_host_report(1);
  const int pass = finished && status == MODEL_HBM_PASS;
  nh_puts(pass ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(pass ? 0 : 1);
}
