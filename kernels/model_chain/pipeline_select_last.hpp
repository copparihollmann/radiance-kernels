#pragma once

#include "pipeline_math.hpp"

namespace model_chain {

// Copy only the final time position of a [batch, tokens, width] tensor.
inline void select_last_token(const RK_GLOBAL float* input,
                              RK_GLOBAL float* output,
                              uint32_t batch, uint32_t tokens, uint32_t width,
                              uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < batch * width; i += threads) {
    const uint32_t b = i / width, feature = i % width;
    output[i] = input[(b * tokens + tokens - 1) * width + feature];
  }
}

}  // namespace model_chain
