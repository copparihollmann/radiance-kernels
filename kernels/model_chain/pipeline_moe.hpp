#pragma once

#include "pipeline_math.hpp"

namespace model_chain {

// Each row belongs to one lane; the maximum is subtracted before exponentiation.
inline void moe_softmax(const RK_GLOBAL float* logits, RK_GLOBAL float* probabilities,
                        uint32_t rows, uint32_t experts,
                        uint32_t tid, uint32_t threads) {
  for (uint32_t row = tid; row < rows; row += threads) {
    float maximum = -3.402823466e38f;
    for (uint32_t expert = 0; expert < experts; ++expert) {
      const float value = logits[row * experts + expert];
      if (value > maximum) maximum = value;
    }
    float denominator = 0.0f;
    for (uint32_t expert = 0; expert < experts; ++expert)
      denominator += exponential(logits[row * experts + expert] - maximum);
    for (uint32_t expert = 0; expert < experts; ++expert)
      probabilities[row * experts + expert] =
          exponential(logits[row * experts + expert] - maximum) / denominator;
  }
}

inline void moe_topk_indices(const RK_GLOBAL float* probabilities,
                             RK_GLOBAL int32_t* indices,
                             uint32_t rows, uint32_t experts, uint32_t selected,
                             uint32_t tid, uint32_t threads) {
  for (uint32_t row = tid; row < rows; row += threads) {
    for (uint32_t slot = 0; slot < selected; ++slot) {
      uint32_t best = experts;
      float best_value = -1.0f;
      for (uint32_t expert = 0; expert < experts; ++expert) {
        bool used = false;
        for (uint32_t prior = 0; prior < slot; ++prior)
          if (indices[row * selected + prior] == (int32_t)expert) used = true;
        const float value = probabilities[row * experts + expert];
        if (!used && value > best_value) {
          best_value = value;
          best = expert;
        }
      }
      indices[row * selected + slot] = (int32_t)best;
    }
  }
}

inline void moe_topk_scores(const RK_GLOBAL float* probabilities,
                            const RK_GLOBAL int32_t* indices,
                            RK_GLOBAL float* scores,
                            uint32_t rows, uint32_t experts, uint32_t selected,
                            bool renormalize, uint32_t tid, uint32_t threads) {
  for (uint32_t row = tid; row < rows; row += threads) {
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < selected; ++slot) {
      const uint32_t expert = (uint32_t)indices[row * selected + slot];
      const float value = probabilities[row * experts + expert];
      scores[row * selected + slot] = value;
      sum += value;
    }
    if (renormalize)
      for (uint32_t slot = 0; slot < selected; ++slot)
        scores[row * selected + slot] /= sum;
  }
}

inline void moe_expert_group(const RK_GLOBAL float* hidden,
                             const RK_GLOBAL int32_t* indices,
                             RK_GLOBAL float* grouped,
                             uint32_t rows, uint32_t selected, uint32_t width,
                             uint32_t experts, uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < rows * selected * width; i += threads) {
    const uint32_t row = i / (selected * width);
    const uint32_t slot = (i / width) % selected;
    grouped[i] = (uint32_t)indices[row * selected + slot] < experts
                     ? hidden[row * width + i % width] : 0.0f;
  }
}

// Expert gate/up weights are [experts, 2*intermediate, hidden].
inline void moe_expert_gate_up(const RK_GLOBAL float* grouped,
                               const RK_GLOBAL int32_t* indices,
                               const RK_GLOBAL float* weight,
                               RK_GLOBAL float* projected,
                               uint32_t rows, uint32_t selected,
                               uint32_t hidden, uint32_t intermediate,
                               uint32_t tid, uint32_t threads) {
  const uint32_t out_width = 2 * intermediate;
  for (uint32_t i = tid; i < rows * selected * out_width; i += threads) {
    const uint32_t slot = i / out_width;
    const uint32_t expert = (uint32_t)indices[slot];
    const uint32_t output = i % out_width;
    float sum = 0.0f;
    for (uint32_t h = 0; h < hidden; ++h)
      sum += grouped[slot * hidden + h]
             * weight[(expert * out_width + output) * hidden + h];
    projected[i] = sum;
  }
}

inline void moe_expert_swiglu(const RK_GLOBAL float* gate_up,
                              RK_GLOBAL float* activated,
                              uint32_t elements, uint32_t intermediate,
                              uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < elements; i += threads) {
    const uint32_t slot = i / intermediate;
    const uint32_t k = i % intermediate;
    const float gate = gate_up[slot * 2 * intermediate + k];
    const float up = gate_up[slot * 2 * intermediate + intermediate + k];
    activated[i] = (gate / (1.0f + exponential(-gate))) * up;
  }
}

// Expert down weights are [experts, hidden, intermediate].
inline void moe_expert_down(const RK_GLOBAL float* activated,
                            const RK_GLOBAL int32_t* indices,
                            const RK_GLOBAL float* weight,
                            RK_GLOBAL float* projected,
                            uint32_t rows, uint32_t selected,
                            uint32_t intermediate, uint32_t hidden,
                            uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < rows * selected * hidden; i += threads) {
    const uint32_t slot = i / hidden;
    const uint32_t expert = (uint32_t)indices[slot];
    const uint32_t output = i % hidden;
    float sum = 0.0f;
    for (uint32_t k = 0; k < intermediate; ++k)
      sum += activated[slot * intermediate + k]
             * weight[(expert * hidden + output) * intermediate + k];
    projected[i] = sum;
  }
}

inline void moe_expert_reduce(const RK_GLOBAL float* expert_outputs,
                              const RK_GLOBAL float* scores,
                              RK_GLOBAL float* reduced,
                              uint32_t rows, uint32_t selected, uint32_t hidden,
                              uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < rows * hidden; i += threads) {
    const uint32_t row = i / hidden, h = i % hidden;
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < selected; ++slot)
      sum += scores[row * selected + slot]
             * expert_outputs[(row * selected + slot) * hidden + h];
    reduced[i] = sum;
  }
}

}  // namespace model_chain
