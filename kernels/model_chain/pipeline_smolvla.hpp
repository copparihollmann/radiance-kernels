#pragma once

#include "pipeline_math.hpp"

namespace model_chain {

inline void copy_values(const RK_GLOBAL float* input, RK_GLOBAL float* output,
                        uint32_t count, uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < count; i += threads) output[i] = input[i];
}

inline void scale_values(const RK_GLOBAL float* input, RK_GLOBAL float* output,
                         uint32_t count, float factor,
                         uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < count; i += threads) output[i] = input[i] * factor;
}

inline void layernorm(const RK_GLOBAL float* input,
                      const RK_GLOBAL float* gamma,
                      const RK_GLOBAL float* bias,
                      RK_GLOBAL float* output,
                      uint32_t rows, uint32_t width, float epsilon,
                      uint32_t tid, uint32_t threads) {
  for (uint32_t row = tid; row < rows; row += threads) {
    float mean = 0.0f;
    for (uint32_t col = 0; col < width; ++col)
      mean += input[row * width + col];
    mean /= (float)width;
    float variance = 0.0f;
    for (uint32_t col = 0; col < width; ++col) {
      const float delta = input[row * width + col] - mean;
      variance += delta * delta;
    }
    const float inv = 1.0f / __builtin_sqrtf(variance / (float)width + epsilon);
    for (uint32_t col = 0; col < width; ++col)
      output[row * width + col] =
          (input[row * width + col] - mean) * inv * gamma[col] + bias[col];
  }
}

inline void patch_embed(const RK_GLOBAL float* image,
                        const RK_GLOBAL float* weight,
                        const RK_GLOBAL float* bias,
                        const RK_GLOBAL float* position,
                        RK_GLOBAL float* output,
                        uint32_t image_size, uint32_t patch,
                        uint32_t channels, uint32_t width,
                        uint32_t tid, uint32_t threads) {
  const uint32_t patches_per_axis = image_size / patch;
  const uint32_t patches = patches_per_axis * patches_per_axis;
  for (uint32_t work = tid; work < patches * width; work += threads) {
    const uint32_t token = work / width, output_channel = work % width;
    const uint32_t patch_row = token / patches_per_axis;
    const uint32_t patch_col = token % patches_per_axis;
    float sum = bias[output_channel] + position[work];
    for (uint32_t channel = 0; channel < channels; ++channel)
      for (uint32_t y = 0; y < patch; ++y)
        for (uint32_t x = 0; x < patch; ++x) {
          const uint32_t image_index = channel * image_size * image_size +
              (patch_row * patch + y) * image_size + patch_col * patch + x;
          const uint32_t weight_index =
              ((output_channel * channels + channel) * patch + y) * patch + x;
          sum += image[image_index] * weight[weight_index];
        }
    output[work] = sum;
  }
}

inline void pixel_shuffle(const RK_GLOBAL float* input, RK_GLOBAL float* output,
                          uint32_t source_axis, uint32_t width, uint32_t factor,
                          uint32_t tid, uint32_t threads) {
  const uint32_t count = source_axis * source_axis * width;
  const uint32_t target_axis = source_axis / factor;
  for (uint32_t index = tid; index < count; index += threads) {
    const uint32_t feature = index % width;
    const uint32_t token = index / width;
    const uint32_t row = token / source_axis, column = token % source_axis;
    const uint32_t target_token = (row / factor) * target_axis + column / factor;
    const uint32_t target_feature =
        ((row % factor) * factor + column % factor) * width + feature;
    output[target_token * factor * factor * width + target_feature] = input[index];
  }
}

inline void gelu_tanh_values(const RK_GLOBAL float* input,
                             RK_GLOBAL float* output, uint32_t count,
                             uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < count; i += threads) {
    const float x = input[i];
    const float argument = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    output[i] = 0.5f * x * (1.0f + hyperbolic_tangent(argument));
  }
}

inline void silu_values(const RK_GLOBAL float* input, RK_GLOBAL float* output,
                        uint32_t count, uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < count; i += threads)
    output[i] = input[i] / (1.0f + exponential(-input[i]));
}

inline void bidirectional_mha(const RK_GLOBAL float* q,
                              const RK_GLOBAL float* k,
                              const RK_GLOBAL float* v,
                              RK_GLOBAL float* output,
                              RK_GLOBAL float* scratch,
                              uint32_t tokens, uint32_t heads, uint32_t width,
                              uint32_t tid, uint32_t threads) {
  for (uint32_t work = tid; work < tokens * heads; work += threads) {
    const uint32_t query = work / heads, head = work % heads;
    RK_GLOBAL float* scores = scratch + tid * tokens;
    float maximum = -1.0e30f;
    for (uint32_t key = 0; key < tokens; ++key) {
      float dot = 0.0f;
      for (uint32_t part = 0; part < width; ++part)
        dot += q[(query * heads + head) * width + part] *
               k[(key * heads + head) * width + part];
      scores[key] = dot / __builtin_sqrtf((float)width);
      if (scores[key] > maximum) maximum = scores[key];
    }
    float denominator = 0.0f;
    for (uint32_t key = 0; key < tokens; ++key) {
      scores[key] = exponential(scores[key] - maximum);
      denominator += scores[key];
    }
    for (uint32_t part = 0; part < width; ++part) {
      float sum = 0.0f;
      for (uint32_t key = 0; key < tokens; ++key)
        sum += scores[key] * v[(key * heads + head) * width + part];
      output[(query * heads + head) * width + part] = sum / denominator;
    }
  }
}

inline void rope_positions(const RK_GLOBAL float* input,
                           const RK_GLOBAL int32_t* positions,
                           const RK_GLOBAL float* cosines,
                           const RK_GLOBAL float* sines,
                           RK_GLOBAL float* output,
                           uint32_t tokens, uint32_t heads, uint32_t width,
                           uint32_t tid, uint32_t threads) {
  const uint32_t half = width / 2;
  for (uint32_t index = tid; index < tokens * heads * half; index += threads) {
    const uint32_t pair = index % half;
    const uint32_t token = index / (half * heads);
    const uint32_t base = (index / half) * width + pair;
    const uint32_t trig = (uint32_t)positions[token] * half + pair;
    const float c = cosines[trig], s = sines[trig];
    const float left = input[base], right = input[base + half];
    output[base] = left * c - right * s;
    output[base + half] = right * c + left * s;
  }
}

inline void masked_gqa(const RK_GLOBAL float* q,
                       const RK_GLOBAL float* k,
                       const RK_GLOBAL float* v,
                       const RK_GLOBAL uint32_t* mask,
                       RK_GLOBAL float* output,
                       RK_GLOBAL float* scratch,
                       uint32_t queries, uint32_t keys,
                       uint32_t q_heads, uint32_t kv_heads, uint32_t width,
                       uint32_t tid, uint32_t threads) {
  for (uint32_t work = tid; work < queries * q_heads; work += threads) {
    const uint32_t query = work / q_heads, head = work % q_heads;
    const uint32_t kv_head = head / (q_heads / kv_heads);
    RK_GLOBAL float* scores = scratch + tid * keys;
    float maximum = -1.0e30f;
    for (uint32_t key = 0; key < keys; ++key) {
      if (!mask[query * keys + key]) {
        scores[key] = -1.0e30f;
        continue;
      }
      float dot = 0.0f;
      for (uint32_t part = 0; part < width; ++part)
        dot += q[(query * q_heads + head) * width + part] *
               k[(key * kv_heads + kv_head) * width + part];
      scores[key] = dot / __builtin_sqrtf((float)width);
      if (scores[key] > maximum) maximum = scores[key];
    }
    float denominator = 0.0f;
    for (uint32_t key = 0; key < keys; ++key) {
      scores[key] = mask[query * keys + key]
          ? exponential(scores[key] - maximum) : 0.0f;
      denominator += scores[key];
    }
    for (uint32_t part = 0; part < width; ++part) {
      float sum = 0.0f;
      for (uint32_t key = 0; key < keys; ++key)
        sum += scores[key] * v[(key * kv_heads + kv_head) * width + part];
      output[(query * q_heads + head) * width + part] =
          denominator ? sum / denominator : 0.0f;
    }
  }
}

inline void concat_values(const RK_GLOBAL float* first,
                          const RK_GLOBAL float* second,
                          RK_GLOBAL float* output,
                          uint32_t first_count, uint32_t second_count,
                          uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < first_count + second_count; i += threads)
    output[i] = i < first_count ? first[i] : second[i - first_count];
}

inline void concat_features(const RK_GLOBAL float* first,
                            const RK_GLOBAL float* second,
                            RK_GLOBAL float* output,
                            uint32_t rows, uint32_t first_width,
                            uint32_t second_width,
                            uint32_t tid, uint32_t threads) {
  const uint32_t width = first_width + second_width;
  for (uint32_t i = tid; i < rows * width; i += threads) {
    const uint32_t row = i / width, col = i % width;
    output[i] = col < first_width ? first[row * first_width + col]
                                  : second[row * second_width + col - first_width];
  }
}

inline void repeat_rows(const RK_GLOBAL float* input, RK_GLOBAL float* output,
                        uint32_t rows, uint32_t width,
                        uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < rows * width; i += threads)
    output[i] = input[i % width];
}

inline void append_tokens(const RK_GLOBAL float* prefix,
                          const RK_GLOBAL float* suffix,
                          RK_GLOBAL float* output,
                          uint32_t prefix_count, uint32_t suffix_count,
                          uint32_t tid, uint32_t threads) {
  concat_values(prefix, suffix, output, prefix_count, suffix_count, tid, threads);
}

inline void euler_step(const RK_GLOBAL float* action,
                       const RK_GLOBAL float* velocity,
                       RK_GLOBAL float* output, uint32_t count, float step,
                       uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < count; i += threads)
    output[i] = action[i] + step * velocity[i];
}

}  // namespace model_chain
