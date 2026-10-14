#pragma once

#include "pipeline_math.hpp"

namespace model_chain {

// FP32 [vocab, hidden] embedding table reused by the output projection.
inline void linear_tied(const RK_GLOBAL float* input,
                        const RK_GLOBAL float* embedding,
                        RK_GLOBAL float* output,
                        uint32_t m, uint32_t hidden, uint32_t vocab,
                        uint32_t tid, uint32_t threads) {
  for (uint32_t out = tid; out < m * vocab; out += threads) {
    const uint32_t row = out / vocab, token = out % vocab;
    float sum = 0.0f;
    for (uint32_t inner = 0; inner < hidden; ++inner)
      sum += input[row * hidden + inner] * embedding[token * hidden + inner];
    output[out] = sum;
  }
}

}  // namespace model_chain
