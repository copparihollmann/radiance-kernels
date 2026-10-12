#pragma once

#include "pipeline_math.hpp"

namespace model_chain {

// Row-major signed W[K,N]; one FP32 scale per output column.
inline void linear_i8(const RK_GLOBAL float* input,
                      const RK_GLOBAL int8_t* weight,
                      const RK_GLOBAL float* scales,
                      RK_GLOBAL float* output,
                      uint32_t m, uint32_t k, uint32_t n,
                      uint32_t tid, uint32_t threads) {
  for (uint32_t out = tid; out < m * n; out += threads) {
    const uint32_t row = out / n, col = out % n;
    float sum = 0.0f;
    const float scale = scales[col];
    for (uint32_t inner = 0; inner < k; ++inner)
      sum += input[row * k + inner] *
             ((float)weight[inner * n + col] * scale);
    output[out] = sum;
  }
}

// Row-major embedding[V,H]; one FP32 scale per vocabulary row.
inline void embedding_i8(const RK_GLOBAL int32_t* ids,
                         const RK_GLOBAL int8_t* table,
                         const RK_GLOBAL float* scales,
                         RK_GLOBAL float* output,
                         uint32_t count, uint32_t width, float output_scale,
                         uint32_t tid, uint32_t threads) {
  for (uint32_t index = tid; index < count * width; index += threads) {
    const uint32_t token = (uint32_t)ids[index / width];
    output[index] = ((float)table[token * width + index % width] *
                     scales[token]) * output_scale;
  }
}

}  // namespace model_chain
