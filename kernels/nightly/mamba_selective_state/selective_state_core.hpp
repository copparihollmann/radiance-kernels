#ifndef MAMBA_SELECTIVE_STATE_CORE_HPP
#define MAMBA_SELECTIVE_STATE_CORE_HPP

#include <stdint.h>

namespace mamba_selective_state {

// This first device milestone deliberately uses a bounded software
// exponential.  Its error, and all accumulated scan error, is checked against
// the existing CPU oracle with the explicit contract below.
constexpr float kAbsoluteTolerance = 2.0e-5F;
constexpr float kRelativeTolerance = 2.0e-5F;

static inline float absf(float x) { return x < 0.0F ? -x : x; }

static inline bool finitef(float x) {
    return (__builtin_bit_cast(uint32_t, x) & 0x7F800000U) != 0x7F800000U;
}

static inline float exp_approx(float x) {
    // The generated milestone bounds every call to approximately [-1.5, 0.8]:
    // A_log is [-1.5, -0.3], delta*(-exp(A_log)) is [-0.29, -0.05], and
    // -gate is [-0.8, 0.8].  A degree-10 Taylor polynomial is accurate on that
    // bounded interval and, importantly for Muon, requires no lane-sensitive
    // float-to-integer conversion or exponent-bit construction.
    float p = 2.7557319224e-7F;        // 1 / 10!
    p = p * x + 2.7557319224e-6F;     // 1 / 9!
    p = p * x + 2.4801587302e-5F;     // 1 / 8!
    p = p * x + 1.9841269841e-4F;     // 1 / 7!
    p = p * x + 1.3888888889e-3F;     // 1 / 6!
    p = p * x + 8.3333333333e-3F;     // 1 / 5!
    p = p * x + 4.1666666667e-2F;     // 1 / 4!
    p = p * x + 1.6666666667e-1F;     // 1 / 3!
    p = p * x + 5.0000000000e-1F;     // 1 / 2!
    p = p * x + 1.0F;
    return p * x + 1.0F;
}

static inline float silu_approx(float x) { return x / (1.0F + exp_approx(-x)); }

static inline bool within_tolerance(float actual, float expected) {
    if (!finitef(actual) || !finitef(expected))
        return false;
    const float largest =
        absf(actual) > absf(expected) ? absf(actual) : absf(expected);
    return absf(actual - expected) <=
           kAbsoluteTolerance + kRelativeTolerance * largest;
}

// One channel of one selective-state step.  delta and x_activated are already
// produced by the out-of-scope timestep projection/softplus and causal
// convolution.  State remains FP32 across sequence positions.
static inline float step(float delta, float x_activated, const float *b,
                         const float *c, const float *a_log, float d_skip,
                         float gate, float *state, uint32_t d_state) {
    float y = 0.0F;
    for (uint32_t j = 0; j < d_state; ++j) {
        const float a = -exp_approx(a_log[j]);
        const float decay = exp_approx(delta * a);
        const float next = decay * state[j] + delta * b[j] * x_activated;
        state[j] = next;
        y += next * c[j];
    }
    y += d_skip * x_activated;
    return y * silu_approx(gate);
}

} // namespace mamba_selective_state

#endif
