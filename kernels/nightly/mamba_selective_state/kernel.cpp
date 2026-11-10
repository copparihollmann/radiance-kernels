// Bounded first Mamba device milestone: FP32 selective-state scan only.
//
// Inputs delta and x are already the softplus timestep and causal-convolution
// activation emitted by the existing CPU oracle.  Four single-warp cores cover
// the 64 channels.  This does not implement the convolution, timestep
// projection, input/output projections, or full Mamba.
#include <nightly/device.h>
#include <nightly/verify.h>

#include "mamba_selective_state_data.h"
#include "selective_state_core.hpp"

#ifndef MSS_DIAG
#define MSS_DIAG 0
#endif
#ifndef MSS_INJECT_ERROR
#define MSS_INJECT_ERROR 0
#endif

static_assert(NIGHTLY_CLUSTERS == 1,
              "first selective-state milestone is intentionally one cluster");
static_assert(MSS_STATE == 16, "local FP32 state allocation assumes state=16");

static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t tb) {
    const float *delta = reinterpret_cast<const float *>(MSS_DELTA_ADDR);
    const float *x = reinterpret_cast<const float *>(MSS_X_ADDR);
    const float *b = reinterpret_cast<const float *>(MSS_B_ADDR);
    const float *c = reinterpret_cast<const float *>(MSS_C_ADDR);
    const float *gate = reinterpret_cast<const float *>(MSS_GATE_ADDR);
    const float *a_log = reinterpret_cast<const float *>(MSS_A_LOG_ADDR);
    const float *d_skip = reinterpret_cast<const float *>(MSS_D_ADDR);
    const float *golden_output =
        reinterpret_cast<const float *>(MSS_GOLD_OUTPUT_ADDR);
    const float *golden_state =
        reinterpret_cast<const float *>(MSS_GOLD_STATE_ADDR);
    float *output = reinterpret_cast<float *>(MSS_OUTPUT_ADDR);
    float *final_state = reinterpret_cast<float *>(MSS_FINAL_STATE_ADDR);
    uint16_t *status = reinterpret_cast<uint16_t *>(MSS_STATUS_ADDR);

    const uint32_t thread = tb * tpb + tid;
    const uint32_t threads = tpb * NIGHTLY_CLUSTERS;
    for (uint32_t channel = thread; channel < MSS_CHANNELS;
         channel += threads) {
        // Keep every lane's recurrent state in an explicitly disjoint global
        // slice.  Muon's private-stack lowering aliases lane-local arrays in
        // Cyclotron for this kernel; an address derived from `channel` makes
        // ownership unambiguous.  This is a correctness bring-up, not the
        // eventual on-chip state-residency optimization.
        float *state = final_state + channel * MSS_STATE;
#pragma unroll
        for (uint32_t j = 0; j < MSS_STATE; ++j) state[j] = 0.0F;
        for (uint32_t time = 0; time < MSS_SEQUENCE; ++time) {
            const uint32_t element = time * MSS_CHANNELS + channel;
            float y = mamba_selective_state::step(
                delta[element], x[element], b + time * MSS_STATE,
                c + time * MSS_STATE, a_log + channel * MSS_STATE,
                d_skip[channel], gate[element], state, MSS_STATE);
#if MSS_INJECT_ERROR
            if (channel == 1 && time == 0) y += 1.0F;
#endif
            output[element] = y;
            status[element] = mamba_selective_state::within_tolerance(
                                  y, golden_output[element])
                                  ? 0U
                                  : 0x3F80U; // BF16 1.0: unambiguously far from
                                             // the zero golden
        }
        for (uint32_t j = 0; j < MSS_STATE; ++j) {
            const uint32_t state_element = channel * MSS_STATE + j;
            status[MSS_OUTPUT_ELEMENTS + state_element] =
                mamba_selective_state::within_tolerance(
                    final_state[state_element], golden_state[state_element])
                    ? 0U
                    : 0x3F80U;
        }
    }

#if MSS_DIAG
    mu_fence();
    mu_barrier(1, tpb / MU_NUM_THREADS);
    if (tid == 0) {
        for (uint32_t i = 0; i < 8; ++i) {
            nightly_pb_put(32 + 3 * i,
                           __builtin_bit_cast(uint32_t, output[i]));
            nightly_pb_put(33 + 3 * i,
                           __builtin_bit_cast(uint32_t, golden_output[i]));
            nightly_pb_put(34 + 3 * i, status[i]);
        }
    } else {
        asm volatile("nop");
    }
#endif

    // The exact status check detects numerical failures and untouched poison.
    // Status creation is included in this correctness milestone's timed region.
    nightly_kernel_end(tid, tpb);
    nightly_verify_bf16(MSS_STATUS_ADDR, MSS_GOLD_STATUS_ADDR, MSS_CHECKS, tid,
                        tpb);
}

int main() { return nightly_main(entry, nullptr); }
