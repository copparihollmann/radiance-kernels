#pragma once
#ifdef RADIANCE_DEVICE
#include <mu_intrinsics.h>
#endif
#include "../pipeline_math.hpp"

namespace model_chain {
namespace muon_bf16 {
inline uint16_t encode(float value) {
  union { float f; uint32_t u; } v{value};
  if ((v.u & 0x7fffffffu) > 0x7f800000u) return uint16_t((v.u >> 16) | 0x40u);
  return uint16_t((v.u + 0x7fffu + ((v.u >> 16) & 1u)) >> 16);
}
inline float decode(uint16_t value) {
  union { uint32_t u; float f; } v{uint32_t(value) << 16};
  return v.f;
}
}

// Preserve the original VPU helper's lane ownership and BF16 operand/result
// rounding. Keeping lane 0 as the writer also supports partitioned call sites,
// which invoke the selected-column helper only from that lane.
inline void mx_muon_bf16_add(const RK_GLOBAL float* left,
                             const RK_GLOBAL float* right,
                             RK_GLOBAL float* output, uint32_t count,
                             uint32_t right_width, uint32_t tid, uint32_t) {
  if (tid != 0) return;
  for (uint32_t i = 0; i < count; ++i) {
    const float a = muon_bf16::decode(muon_bf16::encode(left[i]));
    const float b = muon_bf16::decode(muon_bf16::encode(right[i % right_width]));
    output[i] = muon_bf16::decode(muon_bf16::encode(a + b));
  }
}
}
