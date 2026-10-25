#include <stdint.h>

#include "generated/config.h"
#include <nightly/device.h>

#include "kernel_impl.hpp"

static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "host and device cluster counts must match");

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
