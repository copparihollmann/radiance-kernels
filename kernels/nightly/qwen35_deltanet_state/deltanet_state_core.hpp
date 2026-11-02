#ifndef QWEN35_DELTANET_STATE_CORE_HPP
#define QWEN35_DELTANET_STATE_CORE_HPP

#include <stdint.h>

namespace qwen35_deltanet_state {

constexpr float kL2Epsilon = 1.0e-6F;
constexpr float kAbsoluteTolerance = 4.0e-5F;
constexpr float kRelativeTolerance = 4.0e-5F;

static inline float absf(float x) { return x < 0.0F ? -x : x; }

static inline bool finitef(float x) {
    return (__builtin_bit_cast(uint32_t, x) & 0x7F800000U) != 0x7F800000U;
}

static inline bool within_tolerance(float actual, float expected) {
    if (!finitef(actual) || !finitef(expected))
        return false;
    const float largest =
        absf(actual) > absf(expected) ? absf(actual) : absf(expected);
    return absf(actual - expected) <=
           kAbsoluteTolerance + kRelativeTolerance * largest;
}

static inline float exp_approx(float x) {
    // Degree-10 Taylor polynomial. Generated device vectors constrain the
    // official log-decay gate g to [-0.8, 0], where the approximation error is
    // much smaller than the end-to-end recurrence tolerance.
    float p = 2.7557319224e-7F;    // 1 / 10!
    p = p * x + 2.7557319224e-6F; // 1 / 9!
    p = p * x + 2.4801587302e-5F; // 1 / 8!
    p = p * x + 1.9841269841e-4F; // 1 / 7!
    p = p * x + 1.3888888889e-3F; // 1 / 6!
    p = p * x + 8.3333333333e-3F; // 1 / 5!
    p = p * x + 4.1666666667e-2F; // 1 / 4!
    p = p * x + 1.6666666667e-1F; // 1 / 3!
    p = p * x + 5.0000000000e-1F; // 1 / 2!
    p = p * x + 1.0F;
    return p * x + 1.0F;
}

static inline float inverse_l2_norm(const float *x, uint32_t count) {
    float square_sum = 0.0F;
    for (uint32_t i = 0; i < count; ++i)
        square_sum += x[i] * x[i];
    return 1.0F / __builtin_sqrtf(square_sum + kL2Epsilon);
}

// Update one value column of one expanded Qwen3.5 DeltaNet head. The caller
// owns state[key] in device-friendly [value][key] layout. q and k are raw
// post-convolution vectors; beta and log_decay are the official recurrent
// function's gate inputs.
static inline float step_column(const float *q, const float *k, float value,
                                float beta, float log_decay, float q_inv,
                                float k_inv, float query_dim_scale,
                                float *state, uint32_t key_dim) {
    const float decay = exp_approx(log_decay);
    float memory = 0.0F;
    for (uint32_t key_index = 0; key_index < key_dim; ++key_index) {
        const float next = state[key_index] * decay;
        state[key_index] = next;
        memory += next * (k[key_index] * k_inv);
    }
    const float correction = (value - memory) * beta;
    float output = 0.0F;
    for (uint32_t key_index = 0; key_index < key_dim; ++key_index) {
        const float next =
            state[key_index] + (k[key_index] * k_inv) * correction;
        state[key_index] = next;
        output += next * (q[key_index] * q_inv * query_dim_scale);
    }
    return output;
}

} // namespace qwen35_deltanet_state

#endif
