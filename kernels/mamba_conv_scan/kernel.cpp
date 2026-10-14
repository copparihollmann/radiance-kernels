// Reduced Mamba2 conv -> selective-scan chain. The two callbacks exchange
// their values in device memory and preserve both states across decode tokens.
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
#include "data"

extern "C" uint32_t __mu_num_warps = 1;

struct Phase { uint32_t start, length; };
static Phase phase;
__global float conv_y_raw[TOKENS * CHANNELS] = {0};
__global float scan_y_raw[TOKENS * HEADS * HEAD_DIM] = {0};

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

static float prepare_dt(float raw, float bias) {
  const float z = raw + bias;
  const float u = exp_neg(z < 0.0f ? z : -z);
  const float q = u / (2.0f + u), q2 = q * q;
  float power = q, log1p = 0.0f;
  for (uint32_t n = 1; n <= 15; n += 2) {
    log1p += power / static_cast<float>(n);
    power *= q2;
  }
  float dt = (z > 0.0f ? z : 0.0f) + 2.0f * log1p;
  if (dt < DT_MIN) dt = DT_MIN;
  if (dt > DT_MAX) dt = DT_MAX;
  return dt;
}

static void conv(void* raw, uint32_t tid, uint32_t tpb, uint32_t) {
  const Phase* p = reinterpret_cast<const Phase*>(raw);
  for (uint32_t channel = tid; channel < CHANNELS; channel += tpb) {
    float state[KERNEL_WIDTH];
    for (uint32_t k = 0; k < KERNEL_WIDTH; ++k)
      state[k] = conv_state_raw[channel * KERNEL_WIDTH + k];
    for (uint32_t t = p->start; t < p->start + p->length; ++t) {
      for (uint32_t k = 0; k + 1 < KERNEL_WIDTH; ++k)
        state[k] = state[k + 1];
      state[KERNEL_WIDTH - 1] = conv_x_raw[t * CHANNELS + channel];
      float value = conv_bias_raw[channel];
      for (uint32_t k = 0; k < KERNEL_WIDTH; ++k)
        value += state[k] * conv_weight_raw[channel * KERNEL_WIDTH + k];
      conv_y_raw[t * CHANNELS + channel] = silu(value);
    }
    for (uint32_t k = 0; k < KERNEL_WIDTH; ++k)
      conv_state_raw[channel * KERNEL_WIDTH + k] = state[k];
  }
}

static void scan(void* raw, uint32_t tid, uint32_t tpb, uint32_t) {
  const Phase* p = reinterpret_cast<const Phase*>(raw);
  constexpr uint32_t X_CHANNELS = HEADS * HEAD_DIM;
  constexpr uint32_t GROUP_SIZE = HEADS / GROUPS;
  for (uint32_t channel = tid; channel < X_CHANNELS; channel += tpb) {
    const uint32_t head = channel / HEAD_DIM;
    const uint32_t group = head / GROUP_SIZE;
    float state[STATE_SIZE];
    for (uint32_t n = 0; n < STATE_SIZE; ++n)
      state[n] = scan_state_raw[channel * STATE_SIZE + n];
    for (uint32_t t = p->start; t < p->start + p->length; ++t) {
      const float x = conv_y_raw[t * CHANNELS + channel];
      const float dt = prepare_dt(dt_raw[t * HEADS + head], dt_bias_raw[head]);
      const float decay = exp_neg(dt * a_raw[head]);
      float out = d_raw[head] * x;
      for (uint32_t n = 0; n < STATE_SIZE; ++n) {
        const uint32_t b_offset = t * CHANNELS + X_CHANNELS + group * STATE_SIZE + n;
        const uint32_t c_offset = b_offset + GROUPS * STATE_SIZE;
        state[n] = state[n] * decay + dt * conv_y_raw[b_offset] * x;
        out += state[n] * conv_y_raw[c_offset];
      }
      scan_y_raw[t * X_CHANNELS + channel] = out;
    }
    for (uint32_t n = 0; n < STATE_SIZE; ++n)
      scan_state_raw[channel * STATE_SIZE + n] = state[n];
  }
}

static void run_phase(uint32_t start, uint32_t length) {
  phase = {start, length};
  mu_schedule(conv, &phase, 1);
  mu_barrier(0, MU_NUM_CORES);
  mu_schedule(scan, &phase, 1);
  mu_barrier(0, MU_NUM_CORES);
}

int main() {
  run_phase(0, PREFILL_TOKENS);
  for (uint32_t t = PREFILL_TOKENS; t < TOKENS; ++t) run_phase(t, 1);
  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  if (mu_hart_id() != 0) { for (;;) {} }
  uint32_t errors = 0;
  for (uint32_t i = 0; i < TOKENS * CHANNELS; ++i)
    if (!mu_close(conv_y_raw[i], gold_conv_y_raw[i], 3e-5f, 3e-6f)) ++errors;
  for (uint32_t i = 0; i < CHANNELS * KERNEL_WIDTH; ++i)
    if (!mu_close(conv_state_raw[i], gold_conv_state_raw[i], 3e-5f, 3e-6f)) ++errors;
  for (uint32_t i = 0; i < TOKENS * HEADS * HEAD_DIM; ++i)
    if (!mu_close(scan_y_raw[i], gold_scan_y_raw[i], 3e-5f, 3e-6f)) ++errors;
  for (uint32_t i = 0; i < HEADS * HEAD_DIM * STATE_SIZE; ++i)
    if (!mu_close(scan_state_raw[i], gold_scan_state_raw[i], 3e-5f, 3e-6f)) ++errors;
  mu_tohost(errors ? ((errors << 1) | 1u) : 0u);
  return 0;
}
