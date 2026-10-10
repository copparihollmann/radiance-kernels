#pragma once

#include <stdint.h>

#ifdef RADIANCE_DEVICE
#define RK_GLOBAL __global
#else
#define RK_GLOBAL
#endif

namespace model_chain {

inline float exponential(float value) {
  // Range reduction plus degree-7 Taylor on |r| <= ln(2)/2. This avoids
  // target libm and the half-precision fexp instruction in FP32 softmax.
  if (value < -80.0f) return 0.0f;
  if (value > 80.0f) value = 80.0f;
  const float z = value * 1.4426950408889634f;
  const int n = (int)(z + (z >= 0.0f ? 0.5f : -0.5f));
  const float r = value - (float)n * 0.6931471805599453f;
  float p = 1.0f / 5040.0f;
  p = 1.0f / 720.0f + r * p;
  p = 1.0f / 120.0f + r * p;
  p = 1.0f / 24.0f + r * p;
  p = 1.0f / 6.0f + r * p;
  p = 0.5f + r * p;
  p = 1.0f + r * p;
  p = 1.0f + r * p;
  union { uint32_t bits; float value; } scale;
  scale.bits = (uint32_t)(n + 127) << 23;
  return p * scale.value;
}

inline float hyperbolic_tangent(float value) {
  const float magnitude = value >= 0.0f ? value : -value;
  const float e = exponential(-2.0f * magnitude);
  const float result = (1.0f - e) / (1.0f + e);
  return value >= 0.0f ? result : -result;
}

// Row-major X[M,K], gamma[K], and normalized[M,K]. Each row has one writer.
inline void rmsnorm(const RK_GLOBAL float* x, const RK_GLOBAL float* gamma,
                    RK_GLOBAL float* normalized, uint32_t m, uint32_t k,
                    float epsilon, uint32_t tid, uint32_t threads) {
  for (uint32_t row = tid; row < m; row += threads) {
    float square_sum = 0.0f;
    for (uint32_t col = 0; col < k; ++col) {
      const float v = x[row * k + col];
      square_sum += v * v;
    }
    const float scale = 1.0f / __builtin_sqrtf(square_sum / (float)k + epsilon);
    for (uint32_t col = 0; col < k; ++col)
      normalized[row * k + col] = x[row * k + col] * scale * gamma[col];
  }
}

// Row-major W[K,N]. The activation comes from the previous stage's GMEM
// output; it is never replaced with a generated fixture in this operation.
inline void linear(const RK_GLOBAL float* normalized, const RK_GLOBAL float* weight,
                   const RK_GLOBAL float* bias, RK_GLOBAL float* projected,
                   uint32_t m, uint32_t k, uint32_t n,
                   uint32_t tid, uint32_t threads) {
  for (uint32_t out = tid; out < m * n; out += threads) {
    const uint32_t row = out / n, col = out % n;
    float sum = bias ? bias[col] : 0.0f;
    for (uint32_t inner = 0; inner < k; ++inner)
      sum += normalized[row * k + inner] * weight[inner * n + col];
    projected[out] = sum;
  }
}

inline float fp16_to_fp32(uint16_t half) {
  const uint32_t sign = ((uint32_t)half & 0x8000u) << 16;
  const uint32_t exponent = ((uint32_t)half >> 10) & 0x1fu;
  const uint32_t fraction = (uint32_t)half & 0x3ffu;
  if (!exponent && fraction) {
    const float magnitude = (float)fraction * 5.9604644775390625e-8f;
    return sign ? -magnitude : magnitude;
  }
  union { uint32_t bits; float value; } converted;
  converted.bits = sign | ((exponent == 0x1fu ? 0xffu :
                            exponent ? exponent + 112u : 0u) << 23)
                   | (fraction << 13);
  return converted.value;
}

inline void linear_f16(const RK_GLOBAL float* normalized,
                       const RK_GLOBAL uint16_t* weight,
                       RK_GLOBAL float* projected,
                       uint32_t m, uint32_t k, uint32_t n,
                       uint32_t tid, uint32_t threads) {
  for (uint32_t out = tid; out < m * n; out += threads) {
    const uint32_t row = out / n, col = out % n;
    float sum = 0.0f;
    for (uint32_t inner = 0; inner < k; ++inner)
      sum += normalized[row * k + inner] *
             fp16_to_fp32(weight[inner * n + col]);
    projected[out] = sum;
  }
}

inline void residual(const RK_GLOBAL float* projected, const RK_GLOBAL float* skip,
                     RK_GLOBAL float* output, uint32_t count,
                     uint32_t tid, uint32_t threads) {
  for (uint32_t index = tid; index < count; index += threads)
    output[index] = projected[index] + skip[index];
}

inline void embedding(const RK_GLOBAL int32_t* ids, const RK_GLOBAL float* table,
                      RK_GLOBAL float* out, uint32_t count, uint32_t width,
                      float scale, uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < count * width; i += threads)
    out[i] = table[(uint32_t)ids[i / width] * width + i % width] * scale;
}

inline void embedding_f16(const RK_GLOBAL int32_t* ids,
                          const RK_GLOBAL uint16_t* table,
                          RK_GLOBAL float* out, uint32_t count, uint32_t width,
                          float scale, uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < count * width; i += threads)
    out[i] = fp16_to_fp32(table[(uint32_t)ids[i / width] * width + i % width])
           * scale;
}

inline void bias_add(const RK_GLOBAL float* in, const RK_GLOBAL float* bias,
                     RK_GLOBAL float* out, uint32_t count, uint32_t width,
                     uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < count; i += threads)
    out[i] = in[i] + bias[i % width];
}

// Hugging Face's split-half rotary layout: [first half, second half].
inline void rope(const RK_GLOBAL float* in, const RK_GLOBAL float* cosine,
                 const RK_GLOBAL float* sine,
                 RK_GLOBAL float* out, uint32_t batch, uint32_t tokens,
                 uint32_t heads, uint32_t width, uint32_t position_start,
                 uint32_t tid, uint32_t threads) {
  const uint32_t half = width / 2;
  for (uint32_t i = tid; i < batch * tokens * heads * half; i += threads) {
    const uint32_t pair = i % half;
    const uint32_t token = (i / (half * heads)) % tokens;
    const uint32_t base = (i / half) * width + pair;
    const uint32_t trig = (position_start + token) * half + pair;
    const float c = cosine[trig], s = sine[trig];
    const float left = in[base], right = in[base + half];
    out[base] = left * c - right * s;
    out[base + half] = right * c + left * s;
  }
}

// [batch,time,heads,width] cache. A distinct output buffer makes dependency
// and lifetime explicit, including when the old cache is absent for prefill.
inline void kv_append(const RK_GLOBAL float* old_cache,
                      const RK_GLOBAL float* current, RK_GLOBAL float* out,
                      uint32_t batch, uint32_t past, uint32_t tokens,
                      uint32_t heads, uint32_t width,
                      uint32_t tid, uint32_t threads) {
  const uint32_t row = heads * width;
  for (uint32_t i = tid; i < batch * (past + tokens) * row; i += threads) {
    const uint32_t b = i / ((past + tokens) * row);
    const uint32_t offset = i % ((past + tokens) * row);
    out[i] = offset < past * row
        ? old_cache[b * past * row + offset]
        : current[b * tokens * row + offset - past * row];
  }
}

inline float attention_score(const RK_GLOBAL float* q, const RK_GLOBAL float* k,
                             uint32_t batch, uint32_t query, uint32_t qhead,
                             uint32_t key, uint32_t q_tokens, uint32_t kv_tokens,
                             uint32_t qheads, uint32_t kvheads, uint32_t width,
                             float softcap) {
  const uint32_t kvhead = qhead / (qheads / kvheads);
  const uint32_t qo = ((batch * q_tokens + query) * qheads + qhead) * width;
  const uint32_t ko = ((batch * kv_tokens + key) * kvheads + kvhead) * width;
  float score = 0.0f;
  for (uint32_t d = 0; d < width; ++d) score += q[qo + d] * k[ko + d];
  score /= __builtin_sqrtf((float)width);
  return softcap > 0.0f ? softcap * hyperbolic_tangent(score / softcap) : score;
}

// Each lane owns an entire query/head row, so softmax has no cross-lane state.
inline void causal_gqa(const RK_GLOBAL float* q, const RK_GLOBAL float* k,
                       const RK_GLOBAL float* v, RK_GLOBAL float* out,
                       uint32_t batch, uint32_t q_tokens, uint32_t kv_tokens,
                       uint32_t qheads, uint32_t kvheads, uint32_t width,
                       uint32_t query_start, uint32_t window, float softcap,
                       uint32_t tid, uint32_t threads) {
  for (uint32_t row = tid; row < batch * q_tokens * qheads; row += threads) {
    const uint32_t b = row / (q_tokens * qheads);
    const uint32_t query = (row / qheads) % q_tokens;
    const uint32_t head = row % qheads;
    const uint32_t position = query_start + query;
    const uint32_t first = window && position + 1 > window ? position + 1 - window : 0;
    const uint32_t last = position < kv_tokens ? position : kv_tokens - 1;
    float maximum = -3.402823466e38f;
    for (uint32_t key = first; key <= last; ++key) {
      const float s = attention_score(q, k, b, query, head, key, q_tokens,
                                      kv_tokens, qheads, kvheads, width, softcap);
      if (s > maximum) maximum = s;
    }
    float denominator = 0.0f;
    for (uint32_t key = first; key <= last; ++key)
      denominator += exponential(attention_score(q, k, b, query, head, key,
          q_tokens, kv_tokens, qheads, kvheads, width, softcap) - maximum);
    const uint32_t kvhead = head / (qheads / kvheads);
    for (uint32_t d = 0; d < width; ++d) {
      float sum = 0.0f;
      for (uint32_t key = first; key <= last; ++key) {
        const float weight = exponential(attention_score(q, k, b, query,
            head, key, q_tokens, kv_tokens, qheads, kvheads, width, softcap)
            - maximum) / denominator;
        sum += weight * v[((b * kv_tokens + key) * kvheads + kvhead) * width + d];
      }
      out[row * width + d] = sum;
    }
  }
}

inline void gated_activation(const RK_GLOBAL float* gate,
                             const RK_GLOBAL float* up, RK_GLOBAL float* out,
                             uint32_t count, bool gelu_tanh,
                             uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < count; i += threads) {
    const float x = gate[i];
    const float activation = gelu_tanh
        ? 0.5f * x * (1.0f + hyperbolic_tangent(0.7978845608028654f *
            (x + 0.044715f * x * x * x)))
        : x / (1.0f + exponential(-x));
    out[i] = activation * up[i];
  }
}

inline void softcap(const RK_GLOBAL float* in, RK_GLOBAL float* out,
                    uint32_t count, float cap, uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < count; i += threads)
    out[i] = cap * hyperbolic_tangent(in[i] / cap);
}

inline void token_select(const RK_GLOBAL float* logits, RK_GLOBAL int32_t* out,
                         uint32_t batch, uint32_t tokens, uint32_t vocab,
                         uint32_t tid, uint32_t threads) {
  for (uint32_t b = tid; b < batch; b += threads) {
    const uint32_t base = (b * tokens + tokens - 1) * vocab;
    float best = logits[base];
    uint32_t index = 0;
    for (uint32_t v = 1; v < vocab; ++v) {
      if (logits[base + v] > best) { best = logits[base + v]; index = v; }
    }
    out[b] = (int32_t)index;
  }
}

}  // namespace model_chain
