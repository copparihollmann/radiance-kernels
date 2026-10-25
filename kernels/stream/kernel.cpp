#include <stdint.h>

#include "generated/config.h"
#include <mu_partition.h>
#include <nightly/device.h>

static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "host and device cluster counts must match");

extern "C" {
extern __global float stream_a[];
extern __global float stream_b[];
extern __global float stream_c[];
}

static inline void stream_element(uint32_t i) {
#if STREAM_KIND == 0
  stream_c[i] = stream_a[i];
#elif STREAM_KIND == 1
  stream_b[i] = STREAM_SCALAR * stream_c[i];
#elif STREAM_KIND == 2
  stream_c[i] = stream_a[i] + stream_b[i];
#elif STREAM_KIND == 3
  stream_a[i] = stream_b[i] + STREAM_SCALAR * stream_c[i];
#endif
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
#if STREAM_BLOCK_CYCLIC_CHUNK_LINES
  // Round-robin groups of complete 64-byte lines across physical SMs.  This
  // keeps destination lines exclusive while preserving locality within each
  // chunk and spreading every SM over the full address range.
  constexpr uint32_t line_elements = 16u;
  constexpr uint32_t chunk_elements =
      STREAM_BLOCK_CYCLIC_CHUNK_LINES * line_elements;
  constexpr uint32_t chunks =
      (STREAM_ELEMENTS + chunk_elements - 1u) / chunk_elements;
  for (uint32_t chunk = block_id; chunk < chunks;
       chunk += NIGHTLY_CLUSTERS) {
    const uint32_t begin = chunk * chunk_elements;
    const uint32_t candidate_end = begin + chunk_elements;
    const uint32_t end = candidate_end < STREAM_ELEMENTS
        ? candidate_end : STREAM_ELEMENTS;
    for (uint32_t i = begin + tid; i < end; i += threads_per_block) {
      stream_element(i);
    }
  }
#else
#if STREAM_BLOCK_CYCLIC_LINES
  // Round-robin complete 64-byte lines across physical SMs.  This preserves
  // exclusive destination-line ownership while distributing every SM over
  // the full address range instead of one large contiguous quarter.
  constexpr uint32_t line_elements = 16u;
  constexpr uint32_t lines =
      (STREAM_ELEMENTS + line_elements - 1u) / line_elements;
  const uint32_t owned_lines = block_id < lines
      ? (lines + NIGHTLY_CLUSTERS - 1u - block_id) / NIGHTLY_CLUSTERS
      : 0u;
  const uint32_t line_stride = threads_per_block / line_elements;
  const uint32_t lane = tid % line_elements;
  for (uint32_t line_rank = tid / line_elements;
       line_rank < owned_lines; line_rank += line_stride) {
    const uint32_t line = block_id + line_rank * NIGHTLY_CLUSTERS;
    const uint32_t i = line * line_elements + lane;
    if (i >= STREAM_ELEMENTS)
      continue;
#else
  // float32 has 16 elements per 64-byte line.  Give each SM whole,
  // contiguous lines so no line ping-pongs between SM-private caches.
  const MuPartition range = mu_contiguous_partition(
      STREAM_ELEMENTS, NIGHTLY_CLUSTERS, block_id, 16u);
  for (uint32_t i = range.begin + tid; i < range.end;
       i += threads_per_block) {
#endif
    stream_element(i);
  }
#endif
  nightly_kernel_end(tid, threads_per_block);
}

int main() { return nightly_main(kernel_body, nullptr); }
