// The full GPU-side bitwise check runs after the timed STREAM operation.
#include <nightly/host.h>
#include "stream_data.h"

int main() {
  nh_puts("stream: kind="); nh_putu(STREAM_KIND);
  nh_puts(" N="); nh_putu(STREAM_N);
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
      nh_puts("stream: GPU verify timeout\n"); good = 0;
      continue;
    }
    const NhVerify check = nightly_host_read_verify(cluster);
    nh_puts("stream: cluster="); nh_putu(cluster);
    nh_puts(" exact="); nh_putu(check.exact);
    nh_puts("/"); nh_putu(check.n);
    nh_puts(" nonfinite="); nh_putu(check.nonfin); nh_puts("\n");
    good &= check.n == STREAM_N && check.exact == STREAM_N &&
            check.bad == 0 && check.nonfin == 0;
  }
  const volatile uint32_t *out = nightly_gpu_ptr<uint32_t>(STREAM_OUT_ADDR);
  const volatile uint32_t *gold = nightly_gpu_ptr<uint32_t>(STREAM_G_ADDR);
  const uint32_t samples = STREAM_N < 64 ? STREAM_N : 64;
  uint32_t exact_samples = 0;
  for (uint32_t i = 0; i < samples; i++) {
    const uint32_t index = samples == 1 ? 0 :
        (uint32_t)(((uint64_t)i * (STREAM_N - 1u)) / (samples - 1u));
    exact_samples += out[index] == gold[index];
  }
  nh_puts("stream: host_samples="); nh_putu(exact_samples);
  nh_puts("/"); nh_putu(samples); nh_puts("\n");
  good &= exact_samples == samples;
  nh_puts("logical_bytes="); nh_putu(STREAM_LOGICAL_BYTES); nh_puts("\n");
  nh_puts("bytes_per_gpu_cycle_x100=");
  nh_putu(cycles ? (STREAM_LOGICAL_BYTES * 100ull) / cycles : 0);
  nh_puts("\n");
  nh_puts(good ? "PASS\n" : "FAIL\n"); nh_flush();
  exit(good ? 0 : 1);
}
