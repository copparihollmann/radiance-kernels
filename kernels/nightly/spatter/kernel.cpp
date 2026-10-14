// Five Spatter transfer families on one or two nightly HBM clusters.
#include <nightly/device.h>
#include <nightly/verify.h>
#include "spatter_body.hpp"

static void verify_output(uint32_t tid, uint32_t tpb) {
  if (tid == 0) {
    while (nightly_pb_get(NV_GO) != NV_GO_MAGIC) asm volatile("nop");
  } else {
    asm volatile("nop");
  }
  mu_barrier(NV_BAR, tpb / MU_NUM_THREADS);
  const volatile uint32_t *out = (const volatile uint32_t *)SPATTER_OUT_ADDR;
  const uint32_t *gold = (const uint32_t *)SPATTER_GOLD_ADDR;
  uint32_t exact = 0;
  for (uint32_t i = tid; i < SPATTER_OUTPUT_LENGTH; i += tpb) {
    const uint32_t w = 2u * i;
    exact += out[w] == gold[w] && out[w + 1u] == gold[w + 1u];
  }
  volatile __shared uint32_t *slot = (volatile __shared uint32_t *)(tid * 4u);
  *slot = exact;
  mu_fence_smem();
  mu_barrier(NV_BAR, tpb / MU_NUM_THREADS);
  if (tid == 0) {
    uint32_t total = 0;
    for (uint32_t i = 0; i < tpb; i++)
      total += *(const volatile __shared uint32_t *)(i * 4u);
    nightly_pb_put(NV_SLOT + 1, SPATTER_OUTPUT_LENGTH);
    nightly_pb_put(NV_SLOT + 2, total);
    nightly_pb_put(NV_SLOT + 3, 0);
    nightly_pb_put(NV_SLOT + 4, SPATTER_OUTPUT_LENGTH - total);
    nightly_pb_put(NV_SLOT + 5, 0);
    nightly_pb_put(NV_SLOT + 6, 0);
    nightly_pb_put(NV_SLOT + 7, 0);
    nightly_pb_put(NV_SLOT + 0, NV_DONE);
  } else {
    asm volatile("nop");
  }
  mu_barrier(NV_BAR, tpb / MU_NUM_THREADS);
}

static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t cluster) {
  spatter_apply<NIGHTLY_CLUSTERS>(tid, tpb, cluster);
  nightly_kernel_end(tid, tpb);
  verify_output(tid, tpb);
}

int main() { return nightly_main(entry, nullptr); }
