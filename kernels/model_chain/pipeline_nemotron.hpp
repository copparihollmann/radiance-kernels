#pragma once

#include "pipeline_math.hpp"

namespace model_chain {

inline float mamba_silu(float z) {
  const float e = exponential(z < 0.0f ? z : -z);
  return z * (z < 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e));
}

inline float mamba_dt(float raw, float bias, float minimum, float maximum) {
  const float z = raw + bias;
  const float u = exponential(z < 0.0f ? z : -z);
  const float q = u / (2.0f + u), q2 = q * q;
  float power = q, log1p = 0.0f;
  for (uint32_t n = 1; n <= 15; n += 2) {
    log1p += power / (float)n;
    power *= q2;
  }
  float dt = (z > 0.0f ? z : 0.0f) + 2.0f * log1p;
  if (dt < minimum) dt = minimum;
  if (dt > maximum) dt = maximum;
  return dt;
}

inline void mamba_slice(const RK_GLOBAL float* in, RK_GLOBAL float* out,
                        uint32_t batch, uint32_t tokens, uint32_t in_width,
                        uint32_t start, uint32_t out_width,
                        uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < batch * tokens * out_width; i += threads)
    out[i] = in[(i / out_width) * in_width + start + i % out_width];
}

inline void mamba_depthwise_conv(const RK_GLOBAL float* raw,
                                 const RK_GLOBAL float* old_state,
                                 const RK_GLOBAL float* weight,
                                 const RK_GLOBAL float* bias,
                                 RK_GLOBAL float* out, uint32_t batch,
                                 uint32_t tokens, uint32_t channels,
                                 uint32_t kernel_width,
                                 uint32_t tid, uint32_t threads) {
  for (uint32_t row = tid; row < batch * channels; row += threads) {
    const uint32_t b = row / channels, channel = row % channels;
    for (uint32_t t = 0; t < tokens; ++t) {
      float sum = bias[channel];
      for (uint32_t k = 0; k < kernel_width; ++k) {
        const int32_t source = (int32_t)t + (int32_t)k
                               - (int32_t)kernel_width + 1;
        const float sample = source >= 0
            ? raw[(b * tokens + (uint32_t)source) * channels + channel]
            : old_state ? old_state[row * kernel_width
                                    + kernel_width + source] : 0.0f;
        sum += sample * weight[channel * kernel_width + k];
      }
      out[(b * tokens + t) * channels + channel] = mamba_silu(sum);
    }
  }
}

inline void mamba_conv_state(const RK_GLOBAL float* raw,
                             const RK_GLOBAL float* old_state,
                             RK_GLOBAL float* out, uint32_t batch,
                             uint32_t tokens, uint32_t channels,
                             uint32_t kernel_width,
                             uint32_t tid, uint32_t threads) {
  for (uint32_t row = tid; row < batch * channels; row += threads) {
    const uint32_t b = row / channels, channel = row % channels;
    for (uint32_t k = 0; k < kernel_width; ++k) {
      const int32_t source = (int32_t)tokens + (int32_t)k
                             - (int32_t)kernel_width;
      out[row * kernel_width + k] = source >= 0
          ? raw[(b * tokens + (uint32_t)source) * channels + channel]
          : old_state ? old_state[row * kernel_width
                                  + kernel_width + source] : 0.0f;
    }
  }
}

inline void mamba_scan(const RK_GLOBAL float* x, const RK_GLOBAL float* b_value,
                       const RK_GLOBAL float* c_value, const RK_GLOBAL float* raw_dt,
                       const RK_GLOBAL float* old_state,
                       const RK_GLOBAL float* a_log, const RK_GLOBAL float* d,
                       const RK_GLOBAL float* dt_bias,
                       RK_GLOBAL float* output, RK_GLOBAL float* final_state,
                       uint32_t batch, uint32_t tokens, uint32_t heads,
                       uint32_t head_width, uint32_t groups, uint32_t state_width,
                       float dt_min, float dt_max,
                       uint32_t tid, uint32_t threads) {
  const uint32_t channels = heads * head_width;
  for (uint32_t row = tid; row < batch * channels; row += threads) {
    const uint32_t batch_index = row / channels;
    const uint32_t head = row / head_width % heads;
    const uint32_t channel = row % channels;
    const uint32_t group = head / (heads / groups);
    float state[128] = {0};
    for (uint32_t n = 0; n < state_width; ++n)
      state[n] = old_state ? old_state[row * state_width + n] : 0.0f;
    const float a = exponential(a_log[head]);
    for (uint32_t t = 0; t < tokens; ++t) {
      const float value = x[(batch_index * tokens + t) * channels + channel];
      const float dt = mamba_dt(raw_dt[(batch_index * tokens + t) * heads + head],
                                dt_bias[head], dt_min, dt_max);
      const float decay = exponential(-dt * a);
      float result = d[head] * value;
      for (uint32_t n = 0; n < state_width; ++n) {
        const uint32_t offset = ((batch_index * tokens + t) * groups + group)
                              * state_width + n;
        state[n] = state[n] * decay + dt * b_value[offset] * value;
        result += state[n] * c_value[offset];
      }
      if (output) output[(batch_index * tokens + t) * channels + channel] = result;
    }
    if (final_state)
      for (uint32_t n = 0; n < state_width; ++n)
        final_state[row * state_width + n] = state[n];
  }
}

inline void mamba_gated_norm(const RK_GLOBAL float* x,
                             const RK_GLOBAL float* gate,
                             const RK_GLOBAL float* gamma,
                             RK_GLOBAL float* out, uint32_t rows,
                             uint32_t width, uint32_t group_width,
                             float epsilon, uint32_t tid, uint32_t threads) {
  const uint32_t groups = width / group_width;
  for (uint32_t row = tid; row < rows * groups; row += threads) {
    const uint32_t offset = (row / groups) * width + row % groups * group_width;
    float squares = 0.0f;
    for (uint32_t i = 0; i < group_width; ++i) {
      const float value = x[offset + i] * mamba_silu(gate[offset + i]);
      squares += value * value;
    }
    const float scale = 1.0f / __builtin_sqrtf(squares / (float)group_width + epsilon);
    for (uint32_t i = 0; i < group_width; ++i)
      out[offset + i] = x[offset + i] * mamba_silu(gate[offset + i])
                        * scale * gamma[row % groups * group_width + i];
  }
}

inline void relu_squared(const RK_GLOBAL float* in, RK_GLOBAL float* out,
                         uint32_t count, uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < count; i += threads) {
    const float positive = in[i] > 0.0f ? in[i] : 0.0f;
    out[i] = positive * positive;
  }
}

}  // namespace model_chain
