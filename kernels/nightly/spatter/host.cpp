#include <nightly/host.h>
#include "spatter_data.h"

int main() {
  nh_puts("spatter: kind="); nh_putu(SPATTER_KIND);
  nh_puts(" length="); nh_putu(SPATTER_LENGTH);
  nh_puts(" count="); nh_putu(SPATTER_COUNT);
  nh_puts(" clusters="); nh_putu(NIGHTLY_CLUSTERS); nh_puts("\n"); nh_flush();
  nightly_host_clear_verify(NIGHTLY_CLUSTERS);
  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int finished = nightly_host_wait(NIGHTLY_CLUSTERS);
  const uint32_t cycles = nightly_host_report(NIGHTLY_CLUSTERS);
  int good = finished && cycles != 0;
  for (uint32_t cluster = 0; cluster < NIGHTLY_CLUSTERS; cluster++)
    nightly_pb_write(cluster, NV_GO, NV_GO_MAGIC);
  for (uint32_t cluster = 0; cluster < NIGHTLY_CLUSTERS; cluster++) {
    if (!nightly_host_wait_verify(cluster)) {
      nh_puts("spatter: GPU verify timeout\n"); good = 0;
      continue;
    }
    const NhVerify check = nightly_host_read_verify(cluster);
    nh_puts("spatter: cluster="); nh_putu(cluster);
    nh_puts(" exact="); nh_putu(check.exact);
    nh_puts("/"); nh_putu(check.n); nh_puts("\n");
    good &= check.n == SPATTER_OUTPUT_LENGTH &&
            check.exact == SPATTER_OUTPUT_LENGTH && check.bad == 0;
  }
  const volatile uint32_t *out = nightly_gpu_ptr<uint32_t>(SPATTER_OUT_ADDR);
  const volatile uint32_t *gold = nightly_gpu_ptr<uint32_t>(SPATTER_GOLD_ADDR);
  const uint32_t samples = SPATTER_OUTPUT_LENGTH < 64 ?
      SPATTER_OUTPUT_LENGTH : 64;
  uint32_t exact_samples = 0;
  for (uint32_t i = 0; i < samples; i++) {
    const uint32_t index = samples == 1 ? 0 :
        (uint32_t)(((uint64_t)i * (SPATTER_OUTPUT_LENGTH - 1u)) /
                   (samples - 1u));
    const uint32_t w = 2u * index;
    exact_samples += out[w] == gold[w] && out[w + 1u] == gold[w + 1u];
  }
  nh_puts("spatter: host_samples="); nh_putu(exact_samples);
  nh_puts("/"); nh_putu(samples); nh_puts("\n");
  good &= exact_samples == samples;
  nh_puts("logical_bytes="); nh_putu(SPATTER_LOGICAL_BYTES); nh_puts("\n");
  nh_puts("bytes_per_gpu_cycle_x100=");
  nh_putu(cycles ? (SPATTER_LOGICAL_BYTES * 100ull) / cycles : 0);
  nh_puts("\n");
  nh_puts(good ? "PASS\n" : "FAIL\n"); nh_flush();
  exit(good ? 0 : 1);
}
