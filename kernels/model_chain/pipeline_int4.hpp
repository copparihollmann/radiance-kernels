#pragma once

#include "pipeline_math.hpp"

namespace model_chain {

// Row-major signed W[K,N], two weights per byte; one FP32 scale per output.
inline void linear_i4(const RK_GLOBAL float* input,
                      const RK_GLOBAL uint8_t* weight,
                      const RK_GLOBAL float* scales,
                      RK_GLOBAL float* output,
                      uint32_t m, uint32_t k, uint32_t n,
                      uint32_t tid, uint32_t threads) {
  for (uint32_t out = tid; out < m * n; out += threads) {
    const uint32_t row = out / n, col = out % n;
    float sum = 0.0f;
    const float scale = scales[col];
    for (uint32_t inner = 0; inner < k; ++inner) {
      const uint32_t index = inner * n + col;
      const uint32_t nibble = (weight[index >> 1] >> ((index & 1u) * 4u)) & 15u;
      const int32_t signed_weight = nibble < 8u ? (int32_t)nibble : (int32_t)nibble - 16;
      sum += input[row * k + inner] * ((float)signed_weight * scale);
    }
    output[out] = sum;
  }
}

}  // namespace model_chain
