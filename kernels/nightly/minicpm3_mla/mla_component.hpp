#ifndef MINICPM3_MLA_COMPONENT_HPP
#define MINICPM3_MLA_COMPONENT_HPP

#include <stdint.h>

namespace minicpm3_mla {

// Contract against the independent binary64/math.fsum CPU oracle.  Inputs,
// weights, stage boundaries, and device arithmetic are IEEE binary32.
constexpr float kAbsoluteTolerance = 3.0e-5F;
constexpr float kRelativeTolerance = 3.0e-5F;
constexpr float kRmsNormEpsilon = 1.0e-5F;

static inline float absf(float value) { return value < 0.0F ? -value : value; }

static inline bool finitef(float value) {
    return (__builtin_bit_cast(uint32_t, value) & 0x7F800000U) != 0x7F800000U;
}

static inline bool within_tolerance(float actual, float expected) {
    if (!finitef(actual) || !finitef(expected)) return false;
    const float scale =
        absf(actual) > absf(expected) ? absf(actual) : absf(expected);
    return absf(actual - expected) <=
           kAbsoluteTolerance + kRelativeTolerance * scale;
}

// One row of either projection.  Builds use -ffp-contract=off so this is a
// specified left-to-right sequence of FP32 multiply and add operations.
static inline float dot(const float *input, const float *weight,
                        uint32_t width) {
    float sum = 0.0F;
    for (uint32_t column = 0; column < width; ++column)
        sum += input[column] * weight[column];
    return sum;
}

// MiniCPM's learned RMSNorm over the compressed KV latent.
static inline float normalize_element(const float *latent,
                                      const float *norm_weight,
                                      uint32_t rank, uint32_t column) {
    float sum_square = 0.0F;
    for (uint32_t index = 0; index < rank; ++index)
        sum_square += latent[index] * latent[index];
    const float inverse_rms =
        1.0F / __builtin_sqrtf(sum_square / static_cast<float>(rank) +
                               kRmsNormEpsilon);
    return latent[column] * inverse_rms * norm_weight[column];
}

} // namespace minicpm3_mla

#endif
