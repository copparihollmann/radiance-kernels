#ifndef MU_PARTITION_H
#define MU_PARTITION_H

#include <stdint.h>

// A cache-line-aware, contiguous partition of [0, total).  Boundaries between
// nonempty partitions are multiples of granule; only the final end may be a
// partial granule.  This keeps neighboring cache lines on one cluster instead
// of interleaving their lanes across every cluster.
struct MuPartition {
  uint32_t begin;
  uint32_t end;
};

static constexpr MuPartition mu_contiguous_partition(
    uint32_t total, uint32_t partitions, uint32_t partition,
    uint32_t granule = 1u) {
  if (partitions == 0u || partition >= partitions || total == 0u)
    return {0u, 0u};
  if (granule == 0u) granule = 1u;

  const uint64_t chunks = total / granule + (total % granule != 0u);
  const uint64_t chunks_per_partition = chunks / partitions;
  const uint64_t extra = chunks % partitions;
  const uint64_t begin_chunk =
      partition * chunks_per_partition +
      (partition < extra ? partition : extra);
  const uint64_t owned_chunks =
      chunks_per_partition + (partition < extra ? 1u : 0u);
  uint64_t begin = begin_chunk * granule;
  uint64_t end = (begin_chunk + owned_chunks) * granule;
  if (begin > total) begin = total;
  if (end > total) end = total;
  return {static_cast<uint32_t>(begin), static_cast<uint32_t>(end)};
}

static_assert(mu_contiguous_partition(4102u, 4u, 0u, 16u).begin == 0u);
static_assert(mu_contiguous_partition(4102u, 4u, 0u, 16u).end == 1040u);
static_assert(mu_contiguous_partition(4102u, 4u, 1u, 16u).begin == 1040u);
static_assert(mu_contiguous_partition(4102u, 4u, 3u, 16u).end == 4102u);
static_assert(mu_contiguous_partition(256u, 4u, 2u, 8u).begin == 128u);
static_assert(mu_contiguous_partition(256u, 4u, 2u, 8u).end == 192u);

#endif
