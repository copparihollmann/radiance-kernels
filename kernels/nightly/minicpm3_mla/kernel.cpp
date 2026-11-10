// Bounded MiniCPM3 MLA component: KV compression, latent RMSNorm, and K/V
// reconstruction.  This is deliberately one SM and excludes attention.
#ifdef MCM3_CYCLOTRON
#include <mu_intrinsics.h>
#include <mu_schedule.h>
extern "C" uint32_t __mu_num_warps = NIGHTLY_OCC;
extern "C" uint32_t __mu_cluster_from_pb = 0;
#else
#include <nightly/device.h>
#include <nightly/verify.h>
#endif

#include "minicpm3_mla_data.h"
#include "mla_component.hpp"

static_assert(NIGHTLY_CLUSTERS == 1,
              "the first MiniCPM3 MLA milestone is intentionally one SM");
static_assert((MCM3_OUTPUT_ELEMENTS & 1U) == 0U,
              "packed status verification requires an even output count");

static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t tb) {
    const float *hidden =
        reinterpret_cast<const float *>(MCM3_HIDDEN_INPUT_ADDR);
    const float *kv_a =
        reinterpret_cast<const float *>(MCM3_KV_A_WEIGHT_ADDR);
    const float *norm_weight =
        reinterpret_cast<const float *>(MCM3_KV_NORM_WEIGHT_ADDR);
    const float *kv_b =
        reinterpret_cast<const float *>(MCM3_KV_B_WEIGHT_ADDR);
    const float *gold = reinterpret_cast<const float *>(MCM3_GOLD_ADDR);
    volatile float *output = reinterpret_cast<volatile float *>(MCM3_OUTPUT_ADDR);
    volatile uint32_t *status =
        reinterpret_cast<volatile uint32_t *>(MCM3_STATUS_ADDR);

    const uint32_t thread = tb * tpb + tid;
    const uint32_t threads = tpb * NIGHTLY_CLUSTERS;

    // kv_a_proj_with_mqa: split its [latent | shared RoPE key] result into
    // separate contiguous output ranges.
    const uint32_t compressed_elements = MCM3_TOKENS * MCM3_COMPRESSION_WIDTH;
    for (uint32_t element = thread; element < compressed_elements;
         element += threads) {
        const uint32_t token = element / MCM3_COMPRESSION_WIDTH;
        const uint32_t component = element % MCM3_COMPRESSION_WIDTH;
        const float value = minicpm3_mla::dot(
            hidden + token * MCM3_HIDDEN,
            kv_a + component * MCM3_HIDDEN, MCM3_HIDDEN);
        const uint32_t output_index =
            component < MCM3_KV_RANK
                ? MCM3_LATENT_OFFSET + token * MCM3_KV_RANK + component
                : MCM3_ROPE_OFFSET + token * MCM3_ROPE_DIM +
                      component - MCM3_KV_RANK;
        output[output_index] = value;
    }
    mu_fence();
    mu_barrier(1, tpb / MU_NUM_THREADS);

    // Learned RMSNorm over each token's latent only; the RoPE key bypasses it.
    const uint32_t latent_elements = MCM3_TOKENS * MCM3_KV_RANK;
    for (uint32_t element = thread; element < latent_elements;
         element += threads) {
        const uint32_t token = element / MCM3_KV_RANK;
        const uint32_t column = element % MCM3_KV_RANK;
        const float *latent =
            const_cast<const float *>(output) + MCM3_LATENT_OFFSET +
            token * MCM3_KV_RANK;
        output[MCM3_NORMALIZED_OFFSET + element] =
            minicpm3_mla::normalize_element(latent, norm_weight,
                                             MCM3_KV_RANK, column);
    }
    mu_fence();
    mu_barrier(2, tpb / MU_NUM_THREADS);

    // kv_b_proj.  Each head's row layout is [non-RoPE key | value].
    const uint32_t reconstruction_elements =
        MCM3_TOKENS * MCM3_RECONSTRUCTION_WIDTH;
    for (uint32_t element = thread; element < reconstruction_elements;
         element += threads) {
        const uint32_t token = element / MCM3_RECONSTRUCTION_WIDTH;
        const uint32_t component = element % MCM3_RECONSTRUCTION_WIDTH;
        const float *normalized =
            const_cast<const float *>(output) + MCM3_NORMALIZED_OFFSET +
            token * MCM3_KV_RANK;
        output[MCM3_RECONSTRUCTED_OFFSET + element] = minicpm3_mla::dot(
            normalized, kv_b + component * MCM3_KV_RANK, MCM3_KV_RANK);
    }
    mu_fence();
    mu_barrier(3, tpb / MU_NUM_THREADS);

#ifdef MCM3_INJECT_ERROR
    // Negative-control build: corrupt exactly one public stage value after
    // the component has completed, proving that the full-output status scan
    // rejects a numerically wrong result.  Production builds omit this path.
    if (thread == 0)
        output[MCM3_RECONSTRUCTED_OFFSET] += 1.0f;
    mu_fence();
    mu_barrier(4, tpb / MU_NUM_THREADS);
#endif

    // Check every FP32 stage output against the independent oracle.  Store two
    // BF16 status codes with one aligned word store to avoid sub-word races.
    for (uint32_t pair = thread; pair < MCM3_OUTPUT_ELEMENTS / 2;
         pair += threads) {
        const uint32_t low_index = 2 * pair;
        const uint32_t high_index = low_index + 1;
        const uint32_t low =
            minicpm3_mla::within_tolerance(output[low_index], gold[low_index])
                ? 0U
                : 0x3F80U;
        const uint32_t high =
            minicpm3_mla::within_tolerance(output[high_index], gold[high_index])
                ? 0U
                : 0x3F80U;
        status[pair] = low | (high << 16);
    }

    // Status construction is included in timing.  The established nightly
    // verifier requires every code to equal zero and detects untouched poison.
#ifdef MCM3_CYCLOTRON
    mu_fence();
    mu_barrier(5, tpb / MU_NUM_THREADS);
    if (tid == 0) {
        uint32_t bad = 0;
        for (uint32_t pair = 0; pair < MCM3_OUTPUT_ELEMENTS / 2; ++pair)
            bad += status[pair] != 0U;
        volatile uint8_t *console = reinterpret_cast<volatile uint8_t *>(0xFF080000U);
        const char *message = bad == 0 ? "minicpm3_mla Cyclotron PASS\n"
                                       : "minicpm3_mla Cyclotron FAIL\n";
        while (*message) *console = static_cast<uint8_t>(*message++);
    } else {
        asm volatile("nop");
    }
#else
    nightly_kernel_end(tid, tpb);
    nightly_verify_bf16(MCM3_STATUS_ADDR, MCM3_GOLD_STATUS_ADDR,
                        MCM3_OUTPUT_ELEMENTS, tid, tpb);
#endif
}

int main() {
#ifdef MCM3_CYCLOTRON
    mu_schedule(entry, nullptr, NIGHTLY_OCC);
    return 0;
#else
    return nightly_main(entry, nullptr);
#endif
}
