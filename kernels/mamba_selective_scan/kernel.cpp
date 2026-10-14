// One-token-at-a-time Mamba2 SSM recurrence. Inputs are post-convolution X/B/C,
// positive dt after softplus/clamp, and pretransformed A = -exp(A_log).
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
#include "data"

extern "C" uint32_t __mu_num_warps = 1;

struct ScanArgs { uint32_t start, length; };
static ScanArgs args;
__global float y_raw[TOKENS * HEADS * HEAD_DIM] = {0};

// dt*A is nonpositive. Range reduction keeps the degree-8 Taylor polynomial
// within [-0.3125, 0] even at the FP32 underflow edge. Avoid the Muon expf
// library path: it gave lane-dependent results in the 16-lane Cyclotron check.
static float decay_exp(float z) {
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

static void scan(void* raw, uint32_t tid, uint32_t tpb, uint32_t) {
  const ScanArgs* a = reinterpret_cast<const ScanArgs*>(raw);
  constexpr uint32_t CHANNELS = HEADS * HEAD_DIM;
  constexpr uint32_t HEADS_PER_GROUP = HEADS / GROUPS;
  for (uint32_t channel = tid; channel < CHANNELS; channel += tpb) {
    const uint32_t head = channel / HEAD_DIM;
    const uint32_t group = head / HEADS_PER_GROUP;
    float state[STATE_SIZE];
    for (uint32_t n = 0; n < STATE_SIZE; ++n)
      state[n] = state_raw[channel * STATE_SIZE + n];
    for (uint32_t t = a->start; t < a->start + a->length; ++t) {
      const float x = x_raw[t * CHANNELS + channel];
      const float dt = dt_raw[t * HEADS + head];
      const float decay = decay_exp(dt * a_raw[head]);
      float out = d_raw[head] * x;
      for (uint32_t n = 0; n < STATE_SIZE; ++n) {
        const uint32_t offset = (t * GROUPS + group) * STATE_SIZE + n;
        state[n] = state[n] * decay + dt * b_raw[offset] * x;
        out += state[n] * c_raw[offset];
      }
      y_raw[t * CHANNELS + channel] = out;
    }
    for (uint32_t n = 0; n < STATE_SIZE; ++n)
      state_raw[channel * STATE_SIZE + n] = state[n];
  }
}

int main() {
  args = {0, PREFILL_TOKENS};
  mu_schedule(scan, &args, 1);
  mu_barrier(0, MU_NUM_CORES);
  args = {PREFILL_TOKENS, TOKENS - PREFILL_TOKENS};
  mu_schedule(scan, &args, 1);
  mu_barrier(0, MU_NUM_CORES);
  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  if (mu_hart_id() != 0) { for (;;) {} }
  uint32_t errors = 0;
  for (uint32_t i = 0; i < TOKENS * HEADS * HEAD_DIM; ++i)
    if (!mu_close(y_raw[i], gold_y_raw[i], 3e-5f, 3e-6f)) ++errors;
  for (uint32_t i = 0; i < HEADS * HEAD_DIM * STATE_SIZE; ++i)
    if (!mu_close(state_raw[i], gold_state_raw[i], 3e-5f, 3e-6f)) ++errors;
  mu_tohost(errors ? ((errors << 1) | 1u) : 0u);
  return 0;
}
