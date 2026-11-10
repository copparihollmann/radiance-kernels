// One-SM Qwen3.5 Gated DeltaNet recurrent-state component.
//
// This starts at the pinned recurrent function boundary (q, k, v, beta, g).
// It does not implement projections, causal convolution, gate-logit formation,
// output normalization/projection, or any complete Qwen layer/model.
#ifdef Q35_CYCLOTRON
#include <mu_intrinsics.h>
#include <mu_schedule.h>
extern "C" uint32_t __mu_num_warps = NIGHTLY_OCC;
extern "C" uint32_t __mu_cluster_from_pb = 0;
#else
#include <nightly/device.h>
#endif

#include "deltanet_state_core.hpp"
#include "qwen35_deltanet_state_data.h"

static_assert(NIGHTLY_CLUSTERS == 1,
              "bounded Qwen3.5 state milestone supports exactly one SM");
static_assert(Q35_KEY_DIM == 128 && Q35_VALUE_DIM == 128,
              "kernel is fixed to one official-size DeltaNet head");

static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t tb) {
    const float *q = reinterpret_cast<const float *>(Q35_Q_ADDR);
    const float *k = reinterpret_cast<const float *>(Q35_K_ADDR);
    const float *v = reinterpret_cast<const float *>(Q35_V_ADDR);
    const float *beta = reinterpret_cast<const float *>(Q35_BETA_ADDR);
    const float *log_decay =
        reinterpret_cast<const float *>(Q35_LOG_DECAY_ADDR);
    const float *golden_output =
        reinterpret_cast<const float *>(Q35_GOLD_OUTPUT_ADDR);
    const float *golden_state =
        reinterpret_cast<const float *>(Q35_GOLD_STATE_ADDR);
    float *output = reinterpret_cast<float *>(Q35_OUTPUT_ADDR);
    float *final_state = reinterpret_cast<float *>(Q35_FINAL_STATE_ADDR);

    const uint32_t thread = tb * tpb + tid;
    const uint32_t threads = tpb * NIGHTLY_CLUSTERS;
    const float query_dim_scale =
        1.0F / __builtin_sqrtf(static_cast<float>(Q35_KEY_DIM));

    // Value columns are independent. State is deliberately global and laid
    // out [value][key], giving every lane an explicit disjoint ownership range.
    for (uint32_t value_index = thread; value_index < Q35_VALUE_DIM;
         value_index += threads) {
        float *state = final_state + value_index * Q35_KEY_DIM;
        for (uint32_t key_index = 0; key_index < Q35_KEY_DIM; ++key_index)
            state[key_index] = 0.0F;

        for (uint32_t time = 0; time < Q35_SEQUENCE; ++time) {
            const float *q_token = q + time * Q35_KEY_DIM;
            const float *k_token = k + time * Q35_KEY_DIM;
            const float q_inv = qwen35_deltanet_state::inverse_l2_norm(
                q_token, Q35_KEY_DIM);
            const float k_inv = qwen35_deltanet_state::inverse_l2_norm(
                k_token, Q35_KEY_DIM);
            output[time * Q35_VALUE_DIM + value_index] =
                qwen35_deltanet_state::step_column(
                    q_token, k_token, v[time * Q35_VALUE_DIM + value_index],
                    beta[time], log_decay[time], q_inv, k_inv,
                    query_dim_scale, state, Q35_KEY_DIM);
        }
    }

#ifdef Q35_INJECT_ERROR
    // Negative-control build: corrupt one checked output after the recurrence
    // to prove that the complete output/state scan fails closed.
    if (thread == 0)
        output[0] += 1.0F;
#endif

#ifdef Q35_CYCLOTRON
    mu_fence();
    mu_barrier(1, tpb / MU_NUM_THREADS);
    if (tid == 0) {
        uint32_t bad = 0;
        for (uint32_t i = 0; i < Q35_OUTPUT_ELEMENTS; ++i)
            bad += !qwen35_deltanet_state::within_tolerance(
                output[i], golden_output[i]);
        for (uint32_t i = 0; i < Q35_STATE_ELEMENTS; ++i)
            bad += !qwen35_deltanet_state::within_tolerance(
                final_state[i], golden_state[i]);
        volatile uint8_t *console =
            reinterpret_cast<volatile uint8_t *>(0xFF080000U);
        const char *message =
            bad == 0 ? "qwen35_deltanet_state Cyclotron PASS\n"
                     : "qwen35_deltanet_state Cyclotron FAIL\n";
        while (*message)
            *console = static_cast<uint8_t>(*message++);
    } else {
        asm volatile("nop");
    }
#else
    nightly_kernel_end(tid, tpb);
#endif
}

int main() {
#ifdef Q35_CYCLOTRON
    mu_schedule(entry, nullptr, NIGHTLY_OCC);
    return 0;
#else
    return nightly_main(entry, nullptr);
#endif
}
