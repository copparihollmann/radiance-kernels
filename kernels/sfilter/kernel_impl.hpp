#pragma once

#include <stdint.h>

#include "generated/config.h"

struct SFilterArgs {
  __global const float* src;
  __global float* dst;
  uint32_t ldc;
  float mask[9];
};

static constexpr uint32_t kSFilterRowBoundaries[HPC_SM_COUNT + 1u] =
    SFILTER_SM_ROW_BOUNDARIES;

static_assert(SFILTER_SIZE >= 3u, "sfilter requires an interior");
static_assert(SFILTER_SIZE % SFILTER_FLOATS_PER_CACHE_LINE == 0u,
              "each output row must start on a cache-line boundary");
static_assert(SFILTER_FLOATS_PER_CACHE_LINE == 16u,
              "the ownership contract assumes 64-byte cache lines");

// Preserve the pinned Vortex expression: nine scalar FP32 multiplies followed
// by the same left-to-right FP32 additions. In particular, do not contract a
// multiply and add into an FMA.
#pragma clang fp contract(off)
static inline void sfilter(
    void* opaque, uint32_t tid_in_threadblock,
    uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  auto* args = reinterpret_cast<SFilterArgs*>(opaque);
  const uint32_t row_begin = kSFilterRowBoundaries[threadblock_id];
  const uint32_t row_end = kSFilterRowBoundaries[threadblock_id + 1u];
  const uint32_t interior_width = args->ldc - 2u;
  const uint32_t owned_pixels = (row_end - row_begin) * interior_width;

  for (uint32_t local = tid_in_threadblock; local < owned_pixels;
       local += threads_per_threadblock) {
    const uint32_t x = 1u + local % interior_width;
    const uint32_t y = row_begin + local / interior_width;
    const uint32_t addr = x + y * args->ldc;

    const float i0 = args->src[addr - 1u - args->ldc] * args->mask[0];
    const float i1 = args->src[addr - args->ldc] * args->mask[1];
    const float i2 = args->src[addr + 1u - args->ldc] * args->mask[2];
    const float i3 = args->src[addr - 1u] * args->mask[3];
    const float i4 = args->src[addr] * args->mask[4];
    const float i5 = args->src[addr + 1u] * args->mask[5];
    const float i6 = args->src[addr - 1u + args->ldc] * args->mask[6];
    const float i7 = args->src[addr + args->ldc] * args->mask[7];
    const float i8 = args->src[addr + 1u + args->ldc] * args->mask[8];

    args->dst[addr] =
        i0 + i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8;
  }
}
