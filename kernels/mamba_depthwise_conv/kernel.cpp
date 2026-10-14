// Causal Mamba2 depthwise convolution. Each channel keeps K raw projected
// samples, from oldest to newest, across the prefill and decode launches.
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
#include "data"

extern "C" uint32_t __mu_num_warps = 1;

struct ConvArgs { uint32_t start, length; };
static ConvArgs args;
__global float y_raw[TOKENS * CHANNELS] = {0};

// The Muon expf library path gave lane-dependent values in a 16-lane device
// check. For z <= 0, range reduction keeps the Taylor input >= -0.3125.
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
  const float sigmoid = z < 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e);
  return z * sigmoid;
}

static void conv(void* raw, uint32_t tid, uint32_t tpb, uint32_t) {
  const ConvArgs* a = reinterpret_cast<const ConvArgs*>(raw);
  for (uint32_t channel = tid; channel < CHANNELS; channel += tpb) {
    float state[KERNEL_WIDTH];
    for (uint32_t k = 0; k < KERNEL_WIDTH; ++k)
      state[k] = state_raw[channel * KERNEL_WIDTH + k];
    for (uint32_t t = a->start; t < a->start + a->length; ++t) {
      for (uint32_t k = 0; k + 1 < KERNEL_WIDTH; ++k)
        state[k] = state[k + 1];
      state[KERNEL_WIDTH - 1] = x_raw[t * CHANNELS + channel];
      float value = bias_raw[channel];
      for (uint32_t k = 0; k < KERNEL_WIDTH; ++k)
        value += state[k] * weight_raw[channel * KERNEL_WIDTH + k];
      y_raw[t * CHANNELS + channel] = silu(value);
    }
    for (uint32_t k = 0; k < KERNEL_WIDTH; ++k)
      state_raw[channel * KERNEL_WIDTH + k] = state[k];
  }
}

int main() {
  args = {0, PREFILL_TOKENS};
  mu_schedule(conv, &args, 1);
  mu_barrier(0, MU_NUM_CORES);
  args = {PREFILL_TOKENS, TOKENS - PREFILL_TOKENS};
  mu_schedule(conv, &args, 1);
  mu_barrier(0, MU_NUM_CORES);
  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  if (mu_hart_id() != 0) { for (;;) {} }
  uint32_t errors = 0;
  for (uint32_t i = 0; i < TOKENS * CHANNELS; ++i)
    if (!mu_close(y_raw[i], gold_y_raw[i], 3e-5f, 3e-6f)) ++errors;
  for (uint32_t i = 0; i < CHANNELS * KERNEL_WIDTH; ++i)
    if (!mu_close(state_raw[i], gold_state_raw[i], 3e-5f, 3e-6f)) ++errors;
  mu_tohost(errors ? ((errors << 1) | 1u) : 0u);
  return 0;
}
