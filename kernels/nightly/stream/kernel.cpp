// FP32 STREAM Copy, Scale, Add, or Triad on one or two nightly HBM clusters.
#include <nightly/device.h>
#include <nightly/verify.h>
#include "stream_ops.hpp"

static void verify_output(uint32_t tid, uint32_t tpb) {
  if (tid == 0) {
    while (nightly_pb_get(NV_GO) != NV_GO_MAGIC) asm volatile("nop");
  } else {
    asm volatile("nop");
  }
  mu_barrier(NV_BAR, tpb / MU_NUM_THREADS);
  const volatile uint32_t *out = (const volatile uint32_t *)STREAM_OUT_ADDR;
  const uint32_t *gold = (const uint32_t *)STREAM_G_ADDR;
  uint32_t exact = 0, nonfinite = 0;
  for (uint32_t i = tid; i < STREAM_N; i += tpb) {
    const uint32_t value = out[i];
    exact += value == gold[i];
    nonfinite += (value & 0x7f800000u) == 0x7f800000u;
  }
  volatile __shared uint32_t *slot = (volatile __shared uint32_t *)(tid * 8u);
  slot[0] = exact;
  slot[1] = nonfinite;
  mu_fence_smem();
  mu_barrier(NV_BAR, tpb / MU_NUM_THREADS);
  if (tid == 0) {
    uint32_t total_exact = 0, total_nonfinite = 0;
    for (uint32_t i = 0; i < tpb; i++) {
      const volatile __shared uint32_t *part =
          (const volatile __shared uint32_t *)(i * 8u);
      total_exact += part[0];
      total_nonfinite += part[1];
    }
    nightly_pb_put(NV_SLOT + 1, STREAM_N);
    nightly_pb_put(NV_SLOT + 2, total_exact);
    nightly_pb_put(NV_SLOT + 3, 0);
    nightly_pb_put(NV_SLOT + 4, STREAM_N - total_exact);
    nightly_pb_put(NV_SLOT + 5, 0);
    nightly_pb_put(NV_SLOT + 6, total_nonfinite);
    nightly_pb_put(NV_SLOT + 7, 0);
    nightly_pb_put(NV_SLOT + 0, NV_DONE);
  } else {
    asm volatile("nop");
  }
  mu_barrier(NV_BAR, tpb / MU_NUM_THREADS);
}

static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t cluster) {
  stream_apply<NIGHTLY_CLUSTERS>(tid, tpb, cluster);
  nightly_kernel_end(tid, tpb);
  verify_output(tid, tpb);
}

int main() { return nightly_main(entry, nullptr); }
