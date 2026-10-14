#pragma once

#include "pipeline_math.hpp"

namespace model_chain {

// Consecutive signed weights occupy 5 or 6 bits each, low bit first.
inline int32_t signed_packed_weight(const RK_GLOBAL uint8_t* weights,
                                   uint32_t index, uint32_t bits) {
  const uint32_t bit_index = index * bits;
  const uint32_t byte = bit_index >> 3;
  const uint32_t shift = bit_index & 7u;
  uint32_t raw = (uint32_t)weights[byte] >> shift;
  if (shift + bits > 8u)
    raw |= (uint32_t)weights[byte + 1u] << (8u - shift);
  raw &= (1u << bits) - 1u;
  return raw < (1u << (bits - 1u)) ? (int32_t)raw :
         (int32_t)raw - (int32_t)(1u << bits);
}

// W[K,N], with one FP32 scale per output column.
inline void linear_i6(const RK_GLOBAL float* input,
                      const RK_GLOBAL uint8_t* weights,
                      const RK_GLOBAL float* scales,
                      RK_GLOBAL float* output,
                      uint32_t m, uint32_t k, uint32_t n,
                      uint32_t tid, uint32_t threads) {
  for (uint32_t out = tid; out < m * n; out += threads) {
    const uint32_t row = out / n, col = out % n;
    const float scale = scales[col];
    float sum = 0.0f;
    for (uint32_t inner = 0; inner < k; ++inner)
      sum += input[row * k + inner] *
             ((float)signed_packed_weight(weights, inner * n + col, 6) * scale);
    output[out] = sum;
  }
}

// One scale covers 64 hidden values within each vocabulary row.
inline void embedding_i5_group64(const RK_GLOBAL int32_t* ids,
                                 const RK_GLOBAL uint8_t* table,
                                 const RK_GLOBAL float* scales,
                                 RK_GLOBAL float* output,
                                 uint32_t count, uint32_t width,
                                 float output_scale,
                                 uint32_t tid, uint32_t threads) {
  const uint32_t groups = width / 64u;
  for (uint32_t index = tid; index < count * width; index += threads) {
    const uint32_t token = (uint32_t)ids[index / width];
    const uint32_t hidden = index % width;
    output[index] =
        ((float)signed_packed_weight(table, token * width + hidden, 5) *
         scales[token * groups + hidden / 64u]) * output_scale;
  }
}

// The output head reads the same packed table transposed.
inline void linear_i5_tied(const RK_GLOBAL float* input,
                           const RK_GLOBAL uint8_t* table,
                           const RK_GLOBAL float* scales,
                           RK_GLOBAL float* output,
                           uint32_t m, uint32_t hidden, uint32_t vocab,
                           uint32_t tid, uint32_t threads) {
  const uint32_t groups = hidden / 64u;
  for (uint32_t out = tid; out < m * vocab; out += threads) {
    const uint32_t row = out / vocab, token = out % vocab;
    float sum = 0.0f;
    for (uint32_t inner = 0; inner < hidden; ++inner)
      sum += input[row * hidden + inner] *
             ((float)signed_packed_weight(table, token * hidden + inner, 5) *
              scales[token * groups + inner / 64u]);
    output[out] = sum;
  }
}

}  // namespace model_chain
