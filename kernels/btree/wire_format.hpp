#ifndef RADIANCE_BTREE_WIRE_FORMAT_HPP
#define RADIANCE_BTREE_WIRE_FORMAT_HPP

#include <stdint.h>

// Explicit wire ABI for the pinned OpenCL B+Tree fixture.
//
// OpenCL int / C++ int       -> int32_t
// OpenCL long / cl_long      -> uint32_t after generator bounds validation
// OpenCL bool in knode       -> uint32_t containing 0 or 1
//
// The reference bool is followed by three ABI padding bytes before num_keys.
// Replacing it with uint32_t preserves every subsequent offset and makes the
// complete node representation explicit: 37 little-endian words, 148 bytes.
static constexpr uint32_t kBTreeOrder = 16u;

struct BTreeNode {
  int32_t location;
  int32_t indices[kBTreeOrder + 1u];
  int32_t keys[kBTreeOrder + 1u];
  uint32_t is_leaf;
  uint32_t num_keys;
};

struct BTreePointOutput {
  int32_t value;
  uint32_t query_index;
  uint32_t owner_sm;
  uint32_t tag;
};

struct BTreeRangeQuery {
  int32_t start;
  int32_t end;
};

struct BTreeRangeOutput {
  int32_t start;
  int32_t length;
  uint32_t owner_sm;
  uint32_t tag;
};

static_assert(sizeof(BTreeNode) == 148u, "B+Tree node wire ABI changed");
static_assert(__builtin_offsetof(BTreeNode, indices) == 4u,
              "B+Tree indices offset changed");
static_assert(__builtin_offsetof(BTreeNode, keys) == 72u,
              "B+Tree keys offset changed");
static_assert(__builtin_offsetof(BTreeNode, is_leaf) == 140u,
              "B+Tree leaf flag offset changed");
static_assert(__builtin_offsetof(BTreeNode, num_keys) == 144u,
              "B+Tree key-count offset changed");
static_assert(sizeof(BTreePointOutput) == 16u,
              "point output must occupy one quarter cache line");
static_assert(sizeof(BTreeRangeQuery) == 8u, "range query ABI changed");
static_assert(sizeof(BTreeRangeOutput) == 16u,
              "range output must occupy one quarter cache line");

#endif
