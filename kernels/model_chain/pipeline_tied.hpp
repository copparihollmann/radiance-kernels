#pragma once

#include "pipeline_math.hpp"

namespace model_chain {

// W is stored as [vocab, hidden] for input embedding. The tied LM head uses
// its transpose without duplicating the checkpoint tensor in GMEM.
inline void linear_f16_tied(const RK_GLOBAL float* input,
                            const RK_GLOBAL uint16_t* embedding,
                            RK_GLOBAL float* output,
                            uint32_t m, uint32_t hidden, uint32_t vocab,
                            uint32_t tid, uint32_t threads) {
  for (uint32_t out = tid; out < m * vocab; out += threads) {
    const uint32_t row = out / vocab, token = out % vocab;
    float sum = 0.0f;
    for (uint32_t inner = 0; inner < hidden; ++inner)
      sum += input[row * hidden + inner] *
             fp16_to_fp32(embedding[token * hidden + inner]);
    output[out] = sum;
  }
}

}  // namespace model_chain
