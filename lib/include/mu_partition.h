#ifndef MU_PARTITION_H
#define MU_PARTITION_H

#include <stdint.h>

// How a kernel splits a flat index space across the clusters it runs on.
//
// Every policy here hands each cluster whole cache lines, so a line is never
// shared between two cluster-private L1s.  Which one is right depends on the
// access pattern, not on the kernel:
//
//   kContiguous   one run per cluster.  Best locality within a cluster, but
//                 each cluster only ever touches its own slice of the address
//                 space, so it cannot spread load across DRAM banks and ranks.
//   kCyclicLines  single lines round-robin.  Every cluster walks the whole
//                 address range, which spreads bank and rank pressure.
//   kCyclicChunks runs of CHUNK_LINES lines round-robin.  The middle ground:
//                 locality inside a chunk, clusters still spread over the
//                 whole range.
//
// These used to be three #if arms copied into every kernel body.  They are one
// type now, so a kernel reads as the arithmetic it does and the ownership
// argument is made once, here.
enum class MuOwnership : uint32_t {
  kContiguous = 0,
  kCyclicLines = 1,
  kCyclicChunks = 2,
};

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

// The set of elements one cluster owns, as a dense [0, count) space mapped onto
// the real index space.  A kernel walks the dense space with an ordinary
// grid-stride loop and never has to know which policy is in force:
//
//   const MuOwnedSpace owned = MuOwnedSpace::of<POLICY, 16u>(N, clusters, sm);
//   for (uint32_t v = tid; v < owned.count; v += threads)
//     c[owned[v]] = a[owned[v]];
//
// count is exact -- the mapping never returns an index past total -- so the hot
// loop carries no bounds guard and can be unrolled freely.  LINE_ELEMENTS is
// how many elements fit in a 64-byte line (16 for 32-bit types, 32 for 16-bit).
struct MuOwnedSpace {
  uint32_t count;
  uint32_t base;    // kContiguous: first owned index
  uint32_t stride;  // cyclic: elements skipped per owned chunk, 0 if contiguous
  uint32_t chunk;   // cyclic: elements per owned chunk

  constexpr uint32_t operator[](uint32_t v) const {
    if (stride == 0u) return base + v;
    // Only the final owned chunk can be short, so a plain divide is exact for
    // every v < count.
    return base + (v / chunk) * stride + (v % chunk);
  }

  template <MuOwnership POLICY, uint32_t LINE_ELEMENTS,
            uint32_t CHUNK_LINES = 1u>
  static constexpr MuOwnedSpace of(uint32_t total, uint32_t clusters,
                                   uint32_t cluster) {
    static_assert(LINE_ELEMENTS > 0u, "a cache line holds at least one element");
    static_assert(CHUNK_LINES > 0u, "a chunk is at least one line");
    if (clusters == 0u || cluster >= clusters || total == 0u)
      return {0u, 0u, 0u, 1u};

    if constexpr (POLICY == MuOwnership::kContiguous) {
      const MuPartition range =
          mu_contiguous_partition(total, clusters, cluster, LINE_ELEMENTS);
      return {range.end - range.begin, range.begin, 0u, 1u};
    } else {
      constexpr uint32_t chunk_elements =
          (POLICY == MuOwnership::kCyclicLines ? 1u : CHUNK_LINES) *
          LINE_ELEMENTS;
      const uint32_t chunks =
          total / chunk_elements + (total % chunk_elements != 0u);
      if (cluster >= chunks) return {0u, 0u, 0u, chunk_elements};
      const uint32_t owned = (chunks - cluster + clusters - 1u) / clusters;
      uint32_t count = owned * chunk_elements;
      // The globally last chunk may be partial; subtract the shortfall if it
      // is the one this cluster owns.
      const uint32_t last = chunks - 1u;
      if (last % clusters == cluster) {
        const uint32_t tail = total - last * chunk_elements;
        count -= chunk_elements - tail;
      }
      return {count, cluster * chunk_elements, clusters * chunk_elements,
              chunk_elements};
    }
  }
};

// Contiguous: cluster 1 of 4 over 4102 elements starts at 1040 and owns 1040.
static_assert(MuOwnedSpace::of<MuOwnership::kContiguous, 16u>(4102u, 4u, 1u)[0] ==
              1040u);
static_assert(MuOwnedSpace::of<MuOwnership::kContiguous, 16u>(4102u, 4u, 1u)
                  .count == 1024u);
// Cyclic lines: cluster 1 of 4 owns lines 1, 5, 9, ... of a 256-element space.
static_assert(MuOwnedSpace::of<MuOwnership::kCyclicLines, 16u>(256u, 4u, 1u)
                  .count == 64u);
static_assert(MuOwnedSpace::of<MuOwnership::kCyclicLines, 16u>(256u, 4u, 1u)[0] ==
              16u);
static_assert(MuOwnedSpace::of<MuOwnership::kCyclicLines, 16u>(256u, 4u, 1u)[16] ==
              80u);
// A partial final chunk is counted once, by its owner, and by nobody else.
static_assert(MuOwnedSpace::of<MuOwnership::kCyclicLines, 16u>(40u, 2u, 1u)
                  .count == 16u);
static_assert(MuOwnedSpace::of<MuOwnership::kCyclicLines, 16u>(40u, 2u, 0u)
                  .count == 24u);
static_assert(MuOwnedSpace::of<MuOwnership::kCyclicLines, 16u>(40u, 2u, 1u)[7] ==
              23u);
// Cyclic chunks of 2 lines: cluster 0 of 2 takes [0,32) then [64,96).
static_assert(MuOwnedSpace::of<MuOwnership::kCyclicChunks, 16u, 2u>(128u, 2u, 0u)
                  .count == 64u);
static_assert(MuOwnedSpace::of<MuOwnership::kCyclicChunks, 16u, 2u>(128u, 2u,
                                                                    0u)[32] ==
              64u);

#endif
