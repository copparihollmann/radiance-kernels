#pragma once

#include <stdint.h>
#include "pipeline_math.hpp"

#ifdef RADIANCE_DEVICE
#define SF_GLOBAL __global
#else
#define SF_GLOBAL
#endif

namespace spectformer {

// Factor N=7*M (M=2,4,8) into M short transforms followed by seven-way
// combinations. This retains FP32 real/imaginary storage and orthonormal
// scaling while avoiding the N-by-N direct transform at model resolutions.
inline void mixed_fft(const float* input_real, const float* input_imag,
                      float* output_real, float* output_imag,
                      const SF_GLOBAL float* cosine,
                      const SF_GLOBAL float* sine, uint32_t n, bool inverse) {
  const uint32_t m = n / 7;
  float partial_real[7][8], partial_imag[7][8];
  for (uint32_t q = 0; q < 7; ++q) {
    for (uint32_t k1 = 0; k1 < m; ++k1) {
      float ar = 0.0f, ai = 0.0f;
      for (uint32_t j = 0; j < m; ++j) {
        const float c = cosine[(7 * k1) * n + j];
        const float s = sine[(7 * k1) * n + j];
        const float xr = input_real[7 * j + q];
        const float xi = input_imag[7 * j + q];
        ar += xr * c + (inverse ? -xi * s : xi * s);
        ai += xi * c + (inverse ? xr * s : -xr * s);
      }
      partial_real[q][k1] = ar;
      partial_imag[q][k1] = ai;
    }
  }
  const float scale = 1.0f / __builtin_sqrtf((float)n);
  for (uint32_t k = 0; k < n; ++k) {
    const uint32_t k1 = k % m;
    float ar = 0.0f, ai = 0.0f;
    for (uint32_t q = 0; q < 7; ++q) {
      const float c = cosine[k * n + q];
      const float s = sine[k * n + q];
      const float xr = partial_real[q][k1];
      const float xi = partial_imag[q][k1];
      ar += xr * c + (inverse ? -xi * s : xi * s);
      ai += xi * c + (inverse ? xr * s : -xr * s);
    }
    output_real[k] = ar * scale;
    output_imag[k] = ai * scale;
  }
}

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
#ifndef RADIANCE_DEVICE
  // The factored transform fails the Radiance device check at this grid size.
  // Use the checked direct transform on device until its cause is resolved.
  if (width % 7 == 0 && width <= 56) {
    for (uint32_t work = tid; work < height * channels; work += threads) {
      const uint32_t row = work / channels, channel = work % channels;
      float real[56], imag[56], out_real[56], out_imag[56];
      for (uint32_t x = 0; x < width; ++x) {
        real[x] = input[(row * width + x) * channels + channel];
        imag[x] = 0.0f;
      }
      mixed_fft(real, imag, out_real, out_imag, cosine, sine, width, false);
      for (uint32_t k = 0; k < bins; ++k) {
        const uint32_t dest = 2 * ((row * bins + k) * channels + channel);
        output[dest] = out_real[k]; output[dest + 1] = out_imag[k];
      }
    }
    return;
  }
#endif
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
#ifndef RADIANCE_DEVICE
  if (height % 7 == 0 && height <= 56) {
    for (uint32_t work = tid; work < bins * channels; work += threads) {
      const uint32_t frequency_x = work / channels, channel = work % channels;
      float real[56], imag[56], out_real[56], out_imag[56];
      for (uint32_t y = 0; y < height; ++y) {
        const uint32_t source = 2 * ((y * bins + frequency_x) * channels + channel);
        real[y] = input[source]; imag[y] = input[source + 1];
      }
      mixed_fft(real, imag, out_real, out_imag, cosine, sine, height, inverse);
      for (uint32_t y = 0; y < height; ++y) {
        const uint32_t dest = 2 * ((y * bins + frequency_x) * channels + channel);
        output[dest] = out_real[y]; output[dest + 1] = out_imag[y];
      }
    }
    return;
  }
#endif
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
#ifndef RADIANCE_DEVICE
  if (width % 7 == 0 && width <= 56) {
    for (uint32_t work = tid; work < height * channels; work += threads) {
      const uint32_t row = work / channels, channel = work % channels;
      float real[56], imag[56], out_real[56], out_imag[56];
      for (uint32_t k = 0; k < bins; ++k) {
        const uint32_t source = 2 * ((row * bins + k) * channels + channel);
        real[k] = input[source]; imag[k] = input[source + 1];
      }
      for (uint32_t k = bins; k < width; ++k) {
        real[k] = real[width - k]; imag[k] = -imag[width - k];
      }
      mixed_fft(real, imag, out_real, out_imag, cosine, sine, width, true);
      for (uint32_t x = 0; x < width; ++x)
        output[(row * width + x) * channels + channel] = out_real[x];
    }
    return;
  }
#endif
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
    // Muon/Cyclotron's erff and expf paths disagree with the FP32 reference.
    // Approximate exact GELU as x/2 plus an even Chebyshev series in x^2.
    // The coefficients fit x*erf(x/sqrt(2))/2 over |x| <= 6; saturation
    // beyond 6 has absolute error below 6e-9 in the exact function.
    float y;
    if (x >= 6.0f) y = x;
    else if (x <= -6.0f) y = 0.0f;
    else {
      const float t = (x * x) / 18.0f - 1.0f;
      const float b17 = 0.0f, b18 = 0.0f;
      const float b16 = 2.0f * t * b17 - b18 - 4.31628767476e-07f;
      const float b15 = 2.0f * t * b16 - b17 + 1.6450031348e-06f;
      const float b14 = 2.0f * t * b15 - b16 - 6.15045559768e-06f;
      const float b13 = 2.0f * t * b14 - b15 + 2.09947945046e-05f;
      const float b12 = 2.0f * t * b13 - b14 - 6.76077111068e-05f;
      const float b11 = 2.0f * t * b12 - b13 + 0.000203883281238f;
      const float b10 = 2.0f * t * b11 - b12 - 0.000575447086017f;
      const float b9 = 2.0f * t * b10 - b11 + 0.00151531745933f;
      const float b8 = 2.0f * t * b9 - b10 - 0.00371531702853f;
      const float b7 = 2.0f * t * b8 - b9 + 0.00847203374628f;
      const float b6 = 2.0f * t * b7 - b8 - 0.0179937643155f;
      const float b5 = 2.0f * t * b6 - b7 + 0.0358462197919f;
      const float b4 = 2.0f * t * b5 - b6 - 0.0683166083358f;
      const float b3 = 2.0f * t * b4 - b5 + 0.131107787939f;
      const float b2 = 2.0f * t * b3 - b4 - 0.292061322112f;
      const float b1 = 2.0f * t * b2 - b3 + 1.32283169892f;
      y = 0.5f * x + t * b1 - b2 + 1.88273674627f;
    }
    output[i] = y;
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
      scores[key] = model_chain::exponential(scores[key] - maximum);
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
