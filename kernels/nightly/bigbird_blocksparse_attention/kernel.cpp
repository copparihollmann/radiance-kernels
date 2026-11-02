// Bounded BigBird forward-attention component: one batch, one head, one SM.
#ifdef BBSA_CYCLOTRON
#include <mu_intrinsics.h>
#include <mu_schedule.h>
extern "C" uint32_t __mu_num_warps = NIGHTLY_OCC;
extern "C" uint32_t __mu_cluster_from_pb = 0;
#else
#include <nightly/device.h>
#include <nightly/verify.h>
#endif

#include "bigbird_blocksparse_data.h"
#include "blocksparse_core.hpp"

#ifndef BBSA_INJECT_ERROR
#define BBSA_INJECT_ERROR 0
#endif

static_assert(NIGHTLY_CLUSTERS == 1,
              "bounded BigBird milestone is intentionally one SM");
static_assert(BBSA_BLOCKS <= 32, "structural mask is one uint32_t");

static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t tb) {
    const float *query = reinterpret_cast<const float *>(BBSA_QUERY_ADDR);
    const float *key = reinterpret_cast<const float *>(BBSA_KEY_ADDR);
    const float *value = reinterpret_cast<const float *>(BBSA_VALUE_ADDR);
    const uint8_t *random =
        reinterpret_cast<const uint8_t *>(BBSA_RANDOM_ADDR);
    const float *golden = reinterpret_cast<const float *>(BBSA_GOLD_OUTPUT_ADDR);
    const uint32_t *golden_masks =
        reinterpret_cast<const uint32_t *>(BBSA_GOLD_MASKS_ADDR);
    float *output = reinterpret_cast<float *>(BBSA_OUTPUT_ADDR);
    uint32_t *masks = reinterpret_cast<uint32_t *>(BBSA_MASKS_ADDR);
    uint32_t *status = reinterpret_cast<uint32_t *>(BBSA_STATUS_ADDR);

    const uint32_t thread = tb * tpb + tid;
    const uint32_t threads = tpb * NIGHTLY_CLUSTERS;
    for (uint32_t q = thread; q < BBSA_SEQUENCE; q += threads) {
        for (uint32_t d = 0; d < BBSA_HEAD_DIM; ++d) {
            const uint32_t element = q * BBSA_HEAD_DIM + d;
            float actual = bigbird_blocksparse::attention_element(
                q, d, query, key, value, random, BBSA_SEQUENCE,
                BBSA_BLOCK_SIZE, BBSA_HEAD_DIM, BBSA_RANDOM_BLOCKS,
                BBSA_INV_SQRT_HEAD_DIM);
#if BBSA_INJECT_ERROR
            if (element == 17U)
                actual += 1.0F;
#endif
            output[element] = actual;
            status[element] =
                bigbird_blocksparse::within_tolerance(actual, golden[element])
                    ? 0U
                    : 1U;
        }
    }
    for (uint32_t query_block = thread; query_block < BBSA_BLOCKS;
         query_block += threads) {
        const uint32_t actual = bigbird_blocksparse::block_row_bits(
            query_block, random, BBSA_BLOCKS, BBSA_RANDOM_BLOCKS);
        masks[query_block] = actual;
        status[BBSA_OUTPUT_ELEMENTS + query_block] =
            actual == golden_masks[query_block] ? 0U : 1U;
    }

    // Status construction is in the timed region. The post-timing verifier
    // checks every numerical element and every structural row exactly.
#ifdef BBSA_CYCLOTRON
    mu_fence();
    mu_barrier(1, tpb / MU_NUM_THREADS);
    if (tid == 0) {
        uint32_t bad = 0;
        for (uint32_t i = 0; i < BBSA_CHECKS; ++i)
            bad += status[i] != 0U;
        volatile uint8_t *console =
            reinterpret_cast<volatile uint8_t *>(0xFF080000U);
        const char *message =
            bad == 0 ? "bigbird_blocksparse_attention Cyclotron PASS\n"
                     : "bigbird_blocksparse_attention Cyclotron FAIL\n";
        while (*message)
            *console = static_cast<uint8_t>(*message++);
    } else {
        asm volatile("nop");
    }
#else
    nightly_kernel_end(tid, tpb);
    nightly_verify_u32_exact(BBSA_STATUS_ADDR, BBSA_GOLD_STATUS_ADDR,
                             BBSA_CHECKS, tid, tpb);
#endif
}

int main() {
#ifdef BBSA_CYCLOTRON
    mu_schedule(entry, nullptr, NIGHTLY_OCC);
    return 0;
#else
    return nightly_main(entry, nullptr);
#endif
}
