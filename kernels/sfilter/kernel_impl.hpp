#pragma once

// Vortex spatial filter on Muon: a 3x3 convolution over the image interior.
//
// SFILTER_ILP is the unroll factor; kernel.cpp builds the ILP=1 baseline and
// kernel_ilp{2,4}.cpp the unrolled variants, the same arrangement as
// kernels/vecadd.  Each unrolled slot is an independent output pixel, so one
// pixel's nine loads overlap the next pixel's instead of waiting.
//
// Row ownership is NOT one of mu_partition.h's policies: the boundaries come
// from the generator (SFILTER_SM_ROW_BOUNDARIES), which assigns whole rows.

#include <stdint.h>

#include "generated/config.h"
#include <nightly/device.h>
#include <mu_partition.h>

#ifndef SFILTER_ILP
#define SFILTER_ILP 1
#endif

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
template <uint32_t ILP>
static inline void sfilter_impl(
    void* opaque, uint32_t tid_in_threadblock,
    uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  auto* args = reinterpret_cast<SFilterArgs*>(opaque);
  const uint32_t row_begin = kSFilterRowBoundaries[threadblock_id];
  const uint32_t row_end = kSFilterRowBoundaries[threadblock_id + 1u];
  const uint32_t interior_width = args->ldc - 2u;
  const uint32_t owned_pixels = (row_end - row_begin) * interior_width;

  mu_grid_stride<ILP>(0u, owned_pixels, tid_in_threadblock,
                      threads_per_threadblock, [&](uint32_t local) {
    NIGHTLY_LANE_SAMPLE();
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
  });
}

static inline void sfilter(
    void* opaque, uint32_t tid_in_threadblock,
    uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  sfilter_impl<SFILTER_ILP>(opaque, tid_in_threadblock,
                            threads_per_threadblock, threadblock_id);
}

// hpc.mk picks NIGHTLY_CLUSTERS from the preset; HPC_SM_COUNT is the cluster
// count the generated data was partitioned for.  A mismatch means the ELF
// would read someone else's share, so fail the build rather than the run.
static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "preset cluster count must match the generated data");

extern "C" {
extern __global const float sfilter_input[];
extern __global float sfilter_output[];
}

static SFilterArgs sfilter_args = {
    .src = sfilter_input,
    .dst = sfilter_output,
    .ldc = SFILTER_SIZE,
    .mask = {
        __builtin_bit_cast(float, SFILTER_M0_BITS),
        __builtin_bit_cast(float, SFILTER_M1_BITS),
        __builtin_bit_cast(float, SFILTER_M2_BITS),
        __builtin_bit_cast(float, SFILTER_M3_BITS),
        __builtin_bit_cast(float, SFILTER_M4_BITS),
        __builtin_bit_cast(float, SFILTER_M5_BITS),
        __builtin_bit_cast(float, SFILTER_M6_BITS),
        __builtin_bit_cast(float, SFILTER_M7_BITS),
        __builtin_bit_cast(float, SFILTER_M8_BITS),
    },
};

static void kernel_body(void* args, uint32_t tid,
                        uint32_t threads_per_threadblock,
                        uint32_t physical_threadblock_id) {
  sfilter(args, tid, threads_per_threadblock, physical_threadblock_id);
  nightly_kernel_end(tid, threads_per_threadblock);
}

int main() { return nightly_main(kernel_body, &sfilter_args); }
