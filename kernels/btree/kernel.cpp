#include <stdint.h>

#include "generated/config.h"
#include "wire_format.hpp"
#include <mu_partition.h>
#include <nightly/device.h>

static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "host and device cluster counts must match");
static_assert(BTREE_ORDER == kBTreeOrder, "generator/device order mismatch");
static_assert(BTREE_OUTPUT_RECORDS_PER_LINE * sizeof(BTreePointOutput) == 64u,
              "output partition granule must be one cache line");

extern "C" {
extern __global const BTreeNode btree_nodes[];
extern __global const int32_t btree_records[];
extern __global const int32_t btree_point_queries[];
extern __global const BTreeRangeQuery btree_range_queries[];
extern __global BTreePointOutput btree_point_outputs[];
extern __global BTreeRangeOutput btree_range_outputs[];
}

// Keep all SIMD lanes on one control-flow path while traversing a node.  The
// taped-out Muon path can lose lanes under the deeply nested, data-dependent
// early exits used by the direct OpenCL translation.  Bit selection preserves
// the first-match semantics without changing the tree, queries, or outputs.
__attribute__((always_inline))
static inline uint32_t select_u32(uint32_t take, uint32_t yes, uint32_t no) {
  const uint32_t mask = 0u - (take & 1u);
  return (yes & mask) | (no & ~mask);
}

__attribute__((always_inline))
static inline uint32_t descend_to_leaf(int32_t search_key) {
  uint32_t node_index = 0u;
  for (uint32_t level = 0u; level < BTREE_HEIGHT; ++level) {
    const __global BTreeNode* node = &btree_nodes[node_index];
    uint32_t next = node_index;
    uint32_t found = 0u;
#pragma clang loop unroll(disable)
    for (uint32_t slot = 0u; slot < BTREE_ORDER; ++slot) {
      const uint32_t interval =
          (uint32_t)(node->keys[slot] <= search_key) &
          (uint32_t)(node->keys[slot + 1u] > search_key);
      const int32_t candidate = node->indices[slot];
      const uint32_t candidate_u = (uint32_t)candidate;
      const uint32_t valid = (uint32_t)(candidate >= 0) &
                             (uint32_t)(candidate_u < BTREE_NODE_COUNT);
      const uint32_t take = interval & valid & (found ^ 1u);
      next = select_u32(take, candidate_u, next);
      // The original traversal breaks on the first matching interval even if
      // its child index is invalid, so later intervals must remain ignored.
      found |= interval;
    }
    node_index = next;
  }
  return node_index;
}

__attribute__((always_inline))
static inline int32_t leaf_record_index(int32_t search_key) {
  const __global BTreeNode* leaf = &btree_nodes[descend_to_leaf(search_key)];
  uint32_t result = 0xffffffffu;
  uint32_t found = 0u;
#pragma clang loop unroll(disable)
  for (uint32_t slot = 0u; slot < BTREE_ORDER; ++slot) {
    const uint32_t key_match = (uint32_t)(leaf->keys[slot] == search_key);
    const int32_t record_index = leaf->indices[slot];
    const uint32_t record_u = (uint32_t)record_index;
    const uint32_t valid = (uint32_t)(record_index >= 0) &
                           (uint32_t)(record_u < BTREE_RECORD_COUNT);
    const uint32_t take = key_match & valid & (found ^ 1u);
    result = select_u32(take, record_u, result);
    found |= key_match & valid;
  }
  return (int32_t)result;
}

__attribute__((always_inline))
static inline int32_t point_lookup(int32_t search_key) {
  const int32_t record_index = leaf_record_index(search_key);
  const uint32_t record_u = (uint32_t)record_index;
  const uint32_t valid = (uint32_t)(record_index >= 0) &
                         (uint32_t)(record_u < BTREE_RECORD_COUNT);
  const uint32_t mask = 0u - valid;
  const uint32_t value = (uint32_t)btree_records[record_u & mask];
  return (int32_t)((value & mask) | (0xffffffffu & ~mask));
}

// Construct the global output address only after both tree traversals.  This
// changes neither the traversal nor the output contract; it only keeps the
// output address out of the traversal's live-register set.  Force-inline it
// together with the traversal helpers so multi-SM execution has no nested
// device-call state, while the non-unrolled search loops bound register use.
__attribute__((always_inline))
static inline void write_range_output(uint32_t query, int32_t start,
                                      int32_t end, uint32_t block_id) {
  __global BTreeRangeOutput* output = &btree_range_outputs[query];
  output->start = start;
  output->length = start >= 0 && end >= start ? end - start + 1 : 0;
  output->owner_sm = block_id;
  output->tag = BTREE_RANGE_TAG ^ query;
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  // Four 16-byte result records fill one 64-byte line. Contiguous line-granule
  // partitions give every physical SM exclusive ownership of every line it
  // writes, while each query is evaluated exactly once.
  const MuPartition point_range = mu_contiguous_partition(
      BTREE_POINT_QUERY_COUNT, NIGHTLY_CLUSTERS, block_id,
      BTREE_OUTPUT_RECORDS_PER_LINE);
  for (uint32_t query = point_range.begin + tid; query < point_range.end;
       query += threads_per_block) {
    btree_point_outputs[query].value =
        point_lookup(btree_point_queries[query]);
    btree_point_outputs[query].query_index = query;
    btree_point_outputs[query].owner_sm = block_id;
    btree_point_outputs[query].tag = BTREE_POINT_TAG ^ query;
  }

  const MuPartition range_range = mu_contiguous_partition(
      BTREE_RANGE_QUERY_COUNT, NIGHTLY_CLUSTERS, block_id,
      BTREE_OUTPUT_RECORDS_PER_LINE);
  for (uint32_t query = range_range.begin + tid; query < range_range.end;
       query += threads_per_block) {
    const int32_t range_start = btree_range_queries[query].start;
    const int32_t range_end = btree_range_queries[query].end;
    const int32_t start = leaf_record_index(range_start);
    const int32_t end = leaf_record_index(range_end);
    write_range_output(query, start, end, block_id);
  }

  nightly_kernel_end(tid, threads_per_block);
}

int main() { return nightly_main(kernel_body, nullptr); }
