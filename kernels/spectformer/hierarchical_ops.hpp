#pragma once

#include <stdint.h>
#include "spectral.hpp"
#include "pipeline_math.hpp"

namespace spectformer {

inline void conv2d(const SF_GLOBAL float* input, const SF_GLOBAL float* weight,
                   const SF_GLOBAL float* bias, SF_GLOBAL float* output,
                   uint32_t height, uint32_t width, uint32_t channels_in,
                   uint32_t channels_out, uint32_t kernel, uint32_t stride,
                   bool input_nchw, uint32_t tid, uint32_t threads) {
  const int32_t pad = (int32_t)(kernel / 2);
  const uint32_t out_h = (height + 2 * pad - kernel) / stride + 1;
  const uint32_t out_w = (width + 2 * pad - kernel) / stride + 1;
  for (uint32_t index = tid; index < out_h * out_w * channels_out; index += threads) {
    const uint32_t oc = index % channels_out;
    const uint32_t ox = (index / channels_out) % out_w;
    const uint32_t oy = index / (channels_out * out_w);
    float sum = bias ? bias[oc] : 0.0f;
    for (uint32_t ky = 0; ky < kernel; ++ky) {
      const int32_t iy = (int32_t)(oy * stride + ky) - pad;
      if (iy < 0 || iy >= (int32_t)height) continue;
      for (uint32_t kx = 0; kx < kernel; ++kx) {
        const int32_t ix = (int32_t)(ox * stride + kx) - pad;
        if (ix < 0 || ix >= (int32_t)width) continue;
        for (uint32_t ic = 0; ic < channels_in; ++ic) {
          const uint32_t source = input_nchw ?
              (ic * height + (uint32_t)iy) * width + (uint32_t)ix :
              ((uint32_t)iy * width + (uint32_t)ix) * channels_in + ic;
          const uint32_t filter =
              ((oc * channels_in + ic) * kernel + ky) * kernel + kx;
          sum += input[source] * weight[filter];
        }
      }
    }
    output[index] = sum;
  }
}

inline void batchnorm_relu(SF_GLOBAL float* input, const SF_GLOBAL float* gamma,
                           const SF_GLOBAL float* beta,
                           const SF_GLOBAL float* mean,
                           const SF_GLOBAL float* variance,
                           uint32_t count, uint32_t channels,
                           uint32_t tid, uint32_t threads) {
  for (uint32_t index = tid; index < count; index += threads) {
    const uint32_t c = index % channels;
    const float value = ((input[index] - mean[c]) *
                         (1.0f / __builtin_sqrtf(variance[c] + 1.0e-5f))) *
                        gamma[c] + beta[c];
    input[index] = value > 0.0f ? value : 0.0f;
  }
}

inline void depthwise3(const SF_GLOBAL float* input, const SF_GLOBAL float* weight,
                       const SF_GLOBAL float* bias, SF_GLOBAL float* output,
                       uint32_t height, uint32_t width, uint32_t channels,
                       uint32_t tid, uint32_t threads) {
  for (uint32_t index = tid; index < height * width * channels; index += threads) {
    const uint32_t c = index % channels;
    const uint32_t x = (index / channels) % width;
    const uint32_t y = index / (channels * width);
    float sum = bias[c];
    for (uint32_t ky = 0; ky < 3; ++ky) {
      const int32_t iy = (int32_t)y + (int32_t)ky - 1;
      if (iy < 0 || iy >= (int32_t)height) continue;
      for (uint32_t kx = 0; kx < 3; ++kx) {
        const int32_t ix = (int32_t)x + (int32_t)kx - 1;
        if (ix < 0 || ix >= (int32_t)width) continue;
        sum += input[((uint32_t)iy * width + (uint32_t)ix) * channels + c] *
               weight[c * 9 + ky * 3 + kx];
      }
    }
    output[index] = sum;
  }
}

inline void split_attention(const SF_GLOBAL float* q, const SF_GLOBAL float* kv,
                            SF_GLOBAL float* output, SF_GLOBAL float* scratch,
                            uint32_t tokens, uint32_t channels, uint32_t heads,
                            uint32_t queries, uint32_t tid, uint32_t threads) {
  const uint32_t head_width = channels / heads;
  const float scale = 1.0f / __builtin_sqrtf((float)head_width);
  for (uint32_t work = tid; work < queries * heads; work += threads) {
    const uint32_t query = work / heads, head = work % heads;
    SF_GLOBAL float* scores = scratch + tid * tokens;
    float maximum = -3.402823466e38f;
    for (uint32_t key = 0; key < tokens; ++key) {
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_width; ++d)
        dot += q[query * channels + head * head_width + d] *
               kv[key * 2 * channels + head * head_width + d];
      scores[key] = dot * scale;
      if (scores[key] > maximum) maximum = scores[key];
    }
    float denominator = 0.0f;
    for (uint32_t key = 0; key < tokens; ++key) {
      scores[key] = model_chain::exponential(scores[key] - maximum);
      denominator += scores[key];
    }
    for (uint32_t d = 0; d < head_width; ++d) {
      float sum = 0.0f;
      for (uint32_t key = 0; key < tokens; ++key)
        sum += scores[key] *
               kv[key * 2 * channels + channels + head * head_width + d];
      output[query * channels + head * head_width + d] = sum / denominator;
    }
  }
}

inline void init_class_token(const SF_GLOBAL float* tokens,
                             SF_GLOBAL float* output, uint32_t count,
                             uint32_t channels, uint32_t tid,
                             uint32_t threads) {
  for (uint32_t c = tid; c < channels; c += threads) {
    float sum = 0.0f;
    for (uint32_t t = 0; t < count; ++t)
      sum += tokens[t * channels + c];
    output[c] = sum / (float)count;
  }
  for (uint32_t i = tid; i < count * channels; i += threads)
    output[channels + i] = tokens[i];
}

inline void aux_max(const SF_GLOBAL float* input, SF_GLOBAL float* output,
                    uint32_t tokens, uint32_t classes, uint32_t tid,
                    uint32_t threads) {
  for (uint32_t c = tid; c < classes; c += threads) {
    float maximum = -3.402823466e38f;
    for (uint32_t t = 0; t < tokens; ++t) {
      const float value = input[t * classes + c];
      if (value > maximum) maximum = value;
    }
    output[c] = maximum;
  }
}

inline void class_combine(const SF_GLOBAL float* head,
                          const SF_GLOBAL float* aux,
                          SF_GLOBAL float* output, uint32_t classes,
                          uint32_t tid, uint32_t threads) {
  for (uint32_t c = tid; c < classes; c += threads)
    output[c] = head[c] + 0.5f * aux[c];
}

}  // namespace spectformer
