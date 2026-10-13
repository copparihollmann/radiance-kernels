#pragma once

#include <stdint.h>

#ifdef RADIANCE_DEVICE
#define SF_GLOBAL __global
#else
#define SF_GLOBAL
#endif

namespace spectformer {

// All spatial tensors are HWC. A spectrum is [H,W/2+1,C,2], with the
// real and imaginary FP32 components interleaved. Twiddles are precomputed
// FP32 cos/sin tables [length,length] by the host, so no device trig or
// complex ABI is needed. Each one-dimensional transform has ortho scaling.
inline void real_dft_rows(const SF_GLOBAL float* input, SF_GLOBAL float* output,
                          const SF_GLOBAL float* cosine,
                          const SF_GLOBAL float* sine, uint32_t height,
                          uint32_t width, uint32_t channels,
                          uint32_t tid, uint32_t threads) {
  const uint32_t bins = width / 2 + 1;
  const float scale = 1.0f / __builtin_sqrtf((float)width);
  for (uint32_t work = tid; work < height * bins * channels; work += threads) {
    const uint32_t channel = work % channels;
    const uint32_t frequency = (work / channels) % bins;
    const uint32_t row = work / (channels * bins);
    float real = 0.0f, imaginary = 0.0f;
    for (uint32_t x = 0; x < width; ++x) {
      const float value = input[(row * width + x) * channels + channel];
      real += value * cosine[frequency * width + x];
      imaginary -= value * sine[frequency * width + x];
    }
    output[2 * work] = real * scale;
    output[2 * work + 1] = imaginary * scale;
  }
}

inline void complex_dft_columns(const SF_GLOBAL float* input,
                                SF_GLOBAL float* output,
                                const SF_GLOBAL float* cosine,
                                const SF_GLOBAL float* sine, uint32_t height,
                                uint32_t bins, uint32_t channels, bool inverse,
                                uint32_t tid, uint32_t threads) {
  const float scale = 1.0f / __builtin_sqrtf((float)height);
  for (uint32_t work = tid; work < height * bins * channels; work += threads) {
    const uint32_t channel = work % channels;
    const uint32_t frequency_x = (work / channels) % bins;
    const uint32_t frequency_y = work / (channels * bins);
    float real = 0.0f, imaginary = 0.0f;
    for (uint32_t y = 0; y < height; ++y) {
      const uint32_t source = 2 * ((y * bins + frequency_x) * channels + channel);
      const float a = input[source], b = input[source + 1];
      const float c = cosine[frequency_y * height + y];
      const float s = sine[frequency_y * height + y];
      real += inverse ? a * c - b * s : a * c + b * s;
      imaginary += inverse ? a * s + b * c : b * c - a * s;
    }
    output[2 * work] = real * scale;
    output[2 * work + 1] = imaginary * scale;
  }
}

inline void complex_filter(const SF_GLOBAL float* input,
                           const SF_GLOBAL float* filter,
                           SF_GLOBAL float* output, uint32_t complex_count,
                           uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < complex_count; i += threads) {
    const float a = input[2 * i], b = input[2 * i + 1];
    const float c = filter[2 * i], d = filter[2 * i + 1];
    output[2 * i] = a * c - b * d;
    output[2 * i + 1] = a * d + b * c;
  }
}

inline void inverse_real_dft_rows(const SF_GLOBAL float* input,
                                  SF_GLOBAL float* output,
                                  const SF_GLOBAL float* cosine,
                                  const SF_GLOBAL float* sine, uint32_t height,
                                  uint32_t width, uint32_t channels,
                                  uint32_t tid, uint32_t threads) {
  const uint32_t bins = width / 2 + 1;
  const float scale = 1.0f / __builtin_sqrtf((float)width);
  for (uint32_t work = tid; work < height * width * channels; work += threads) {
    const uint32_t channel = work % channels;
    const uint32_t x = (work / channels) % width;
    const uint32_t row = work / (channels * width);
    float sum = 0.0f;
    for (uint32_t k = 0; k < bins; ++k) {
      const uint32_t source = 2 * ((row * bins + k) * channels + channel);
      const float a = input[source], b = input[source + 1];
      const float term = a * cosine[k * width + x] -
                         b * sine[k * width + x];
      const bool paired = k != 0 && !(width % 2 == 0 && k == width / 2);
      sum += paired ? 2.0f * term : term;
    }
    output[work] = sum * scale;
  }
}

inline void token_mean(const SF_GLOBAL float* input, SF_GLOBAL float* output,
                       uint32_t tokens, uint32_t channels,
                       uint32_t tid, uint32_t threads) {
  for (uint32_t c = tid; c < channels; c += threads) {
    float sum = 0.0f;
    for (uint32_t n = 0; n < tokens; ++n)
      sum += input[n * channels + c];
    output[c] = sum / (float)tokens;
  }
}

inline void gelu_exact(const SF_GLOBAL float* input, SF_GLOBAL float* output,
                       uint32_t count, uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < count; i += threads) {
    const float x = input[i];
    output[i] = 0.5f * x * (1.0f + __builtin_erff(x * 0.7071067811865475f));
  }
}

// QKV is row-major [N,3,C] as emitted by upstream nn.Linear. This is the
// noncausal FP32 attention used by the vanilla SpectFormer blocks.
inline void attention_qkv(const SF_GLOBAL float* qkv, SF_GLOBAL float* output,
                          SF_GLOBAL float* scratch, uint32_t tokens,
                          uint32_t channels, uint32_t heads,
                          uint32_t tid, uint32_t threads) {
  const uint32_t head_width = channels / heads;
  const float scale = 1.0f / __builtin_sqrtf((float)head_width);
  for (uint32_t work = tid; work < tokens * heads; work += threads) {
    const uint32_t query = work / heads, head = work % heads;
    SF_GLOBAL float* scores = scratch + tid * tokens;
    float maximum = -3.402823466e38f;
    for (uint32_t key = 0; key < tokens; ++key) {
      float dot = 0.0f;
      for (uint32_t d = 0; d < head_width; ++d)
        dot += qkv[query * 3 * channels + head * head_width + d] *
               qkv[key * 3 * channels + channels + head * head_width + d];
      scores[key] = dot * scale;
      if (scores[key] > maximum) maximum = scores[key];
    }
    float denominator = 0.0f;
    for (uint32_t key = 0; key < tokens; ++key) {
      scores[key] = __builtin_expf(scores[key] - maximum);
      denominator += scores[key];
    }
    for (uint32_t d = 0; d < head_width; ++d) {
      float sum = 0.0f;
      for (uint32_t key = 0; key < tokens; ++key)
        sum += scores[key] *
               qkv[key * 3 * channels + 2 * channels + head * head_width + d];
      output[query * channels + head * head_width + d] = sum / denominator;
    }
  }
}

}  // namespace spectformer
