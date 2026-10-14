// Reduced Qwen3 MoE FFN: router softmax, top-k, selected-score normalization,
// expert gate/up, SiLU, expert down, and weighted reduction in one device ELF.
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
#include "data"

extern "C" uint32_t __mu_num_warps = 1;
__global uint32_t indices_raw[TOKENS * SELECTED] = {0};
__global float scores_raw[TOKENS * SELECTED] = {0};
__global float output_raw[TOKENS * HIDDEN] = {0};
__global float probabilities_raw[TOKENS * EXPERTS] = {0};
__global float activated_raw[TOKENS * INTERMEDIATE] = {0};

static float exp_neg(float z) {
  if (z <= -80.0f) return 0.0f;
  const float x = z * (1.0f / 256.0f);
  float term = 1.0f, sum = 1.0f;
  for (uint32_t n = 1; n <= 8; ++n) {
    term = term * x / static_cast<float>(n);
    sum += term;
  }
  for (uint32_t n = 0; n < 8; ++n) sum *= sum;
  return sum;
}

static float silu(float z) {
  const float e = exp_neg(z < 0.0f ? z : -z);
  return z * (z < 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e));
}

static void moe_ffn(void*, uint32_t tid, uint32_t tpb, uint32_t) {
  for (uint32_t token = tid; token < TOKENS; token += tpb) {
    float maximum = -3.4e38f;
    for (uint32_t expert = 0; expert < EXPERTS; ++expert) {
      float sum = 0.0f;
      for (uint32_t h = 0; h < HIDDEN; ++h)
        sum += input_raw[token * HIDDEN + h]
               * router_weight_raw[expert * HIDDEN + h];
      probabilities_raw[token * EXPERTS + expert] = sum;
      if (sum > maximum) maximum = sum;
    }
    float denominator = 0.0f;
    for (uint32_t expert = 0; expert < EXPERTS; ++expert) {
      const uint32_t offset = token * EXPERTS + expert;
      const float value = exp_neg(probabilities_raw[offset] - maximum);
      probabilities_raw[offset] = value;
      denominator += value;
    }
    uint32_t chosen_mask = 0;
    float selected_sum = 0.0f;
    for (uint32_t slot = 0; slot < SELECTED; ++slot) {
      uint32_t best = EXPERTS;
      float best_probability = -1.0f;
      for (uint32_t expert = 0; expert < EXPERTS; ++expert) {
        const float probability = probabilities_raw[token * EXPERTS + expert];
        if (!(chosen_mask & (1u << expert)) && probability > best_probability) {
          best_probability = probability;
          best = expert;
        }
      }
      chosen_mask |= 1u << best;
      indices_raw[token * SELECTED + slot] = best;
      const float probability = best_probability / denominator;
      scores_raw[token * SELECTED + slot] = probability;
      selected_sum += probability;
    }
    for (uint32_t slot = 0; slot < SELECTED; ++slot)
      scores_raw[token * SELECTED + slot] /= selected_sum;

    for (uint32_t h = 0; h < HIDDEN; ++h)
      output_raw[token * HIDDEN + h] = 0.0f;
    for (uint32_t slot = 0; slot < SELECTED; ++slot) {
      const uint32_t expert = indices_raw[token * SELECTED + slot];
      for (uint32_t k = 0; k < INTERMEDIATE; ++k) {
        float gate = 0.0f, up = 0.0f;
        const uint32_t gate_base = (expert * 2 * INTERMEDIATE + k) * HIDDEN;
        const uint32_t up_base = gate_base + INTERMEDIATE * HIDDEN;
        for (uint32_t h = 0; h < HIDDEN; ++h) {
          const float x = input_raw[token * HIDDEN + h];
          gate += expert_gate_up_raw[gate_base + h] * x;
          up += expert_gate_up_raw[up_base + h] * x;
        }
        activated_raw[token * INTERMEDIATE + k] = silu(gate) * up;
      }
      const float score = scores_raw[token * SELECTED + slot];
      for (uint32_t h = 0; h < HIDDEN; ++h) {
        float value = 0.0f;
        for (uint32_t k = 0; k < INTERMEDIATE; ++k)
          value += expert_down_raw[(expert * HIDDEN + h) * INTERMEDIATE + k]
                   * activated_raw[token * INTERMEDIATE + k];
        output_raw[token * HIDDEN + h] += score * value;
      }
    }
  }
}

int main() {
  mu_schedule(moe_ffn, nullptr, 1);
  mu_barrier(0, MU_NUM_CORES);
  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  if (mu_hart_id() != 0) { for (;;) {} }
  uint32_t errors = 0;
  for (uint32_t i = 0; i < TOKENS * SELECTED; ++i) {
    if (indices_raw[i] != gold_indices_raw[i]) ++errors;
    if (!mu_close(scores_raw[i], gold_scores_raw[i], 3e-5f, 3e-6f)) ++errors;
  }
  for (uint32_t i = 0; i < TOKENS * HIDDEN; ++i)
    if (!mu_close(output_raw[i], gold_output_raw[i], 3e-5f, 3e-6f)) ++errors;
  mu_tohost(errors ? ((errors << 1) | 1u) : 0u);
  return 0;
}
