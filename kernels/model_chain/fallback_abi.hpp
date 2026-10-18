#pragma once

#include <stdint.h>
#ifdef RADIANCE_DEVICE
#include <shared_mem.h>
#endif
#include "pipeline_smolvla.hpp"
#include "pipeline_int8.hpp"
#include "pipeline_int4.hpp"

namespace model_chain {

// GPU words are32bits; native graph builds use uintptr_t without truncating
// their pointers. The simulator bridge explicitly widens the32GPUwords.
struct FallbackDescriptor { uintptr_t word[32]; };
static constexpr uintptr_t FALLBACK_MAGIC = 0x524b4631u;

enum FallbackOperation {
  LinearF32=1, LinearI8=2, LinearI4=3, LinearF16=4,
  LayerNorm=5, RmsNorm=6, PatchEmbed=7, BidirectionalAttention=8,
  MaskedAttention=9, Gelu=10, Silu=11, Residual=12, Bias=13,
  Copy=14, Scale=15, GatedActivation=16, Euler=17, LinearExceptColumns=18
};

// The FP32 remainder of the mixed MX projection, with the original lane and
// reduction order. Selected columns are untouched and remain the MX's work.
inline void linear_except_columns(const RK_GLOBAL float* input,
                                  const RK_GLOBAL float* weights,
                                  RK_GLOBAL float* output,
                                  uint32_t m, uint32_t k, uint32_t n,
                                  uint32_t first, uint32_t columns,
                                  uint32_t tid, uint32_t threads) {
  if (first > n || columns > n-first) return;
  for (uint32_t index = tid; index < m*n; index += threads) {
    const uint32_t row = index/n, col = index%n;
    if (col >= first && col < first+columns) continue;
    float sum = 0;
    for (uint32_t inner = 0; inner < k; ++inner)
      sum += input[row*k+inner] * weights[inner*n+col];
    output[index] = sum;
  }
}

inline uintptr_t fallback_address(const RK_GLOBAL FallbackDescriptor* d,
                                 unsigned index, uintptr_t context, uintptr_t ram) {
  uintptr_t address = d->word[index];
  if (d->word[2] & (uintptr_t(1) << index))
    address += (d->word[2] & 0x80000000u) ? context*d->word[index+16] : context;
  return ram + address;
}

inline float fallback_float(uintptr_t bits) {
  union { uint32_t bits; float value; } result;
  result.bits = (uint32_t)bits;
  return result.value;
}

inline bool fallback_dispatch(const RK_GLOBAL FallbackDescriptor* d,
                              uintptr_t context, uint32_t tid, uint32_t threads,
                              uintptr_t ram = 0) {
  if (d->word[0] != FALLBACK_MAGIC || !threads || tid >= threads) return false;
#define RK_P(i) ((const RK_GLOBAL float*)fallback_address(d,i,context,ram))
#define RK_O(i) ((RK_GLOBAL float*)fallback_address(d,i,context,ram))
#define RK_W(i) ((uint32_t)d->word[i])
  switch (RK_W(1)) {
    case LinearF32:
      linear(RK_P(3),RK_P(4),(d->word[5] || (d->word[2] & (uintptr_t(1)<<5))) ? RK_P(5) : nullptr,RK_O(6),
             RK_W(7),RK_W(8),RK_W(9),tid,threads); break;
    case LinearI8:
      linear_i8(RK_P(3),(const RK_GLOBAL int8_t*)fallback_address(d,4,context,ram),
                RK_P(5),RK_O(6),RK_W(7),RK_W(8),RK_W(9),tid,threads); break;
    case LinearI4:
      linear_i4(RK_P(3),(const RK_GLOBAL uint8_t*)fallback_address(d,4,context,ram),
                RK_P(5),RK_O(6),RK_W(7),RK_W(8),RK_W(9),tid,threads); break;
    case LinearF16:
      linear_f16(RK_P(3),(const RK_GLOBAL uint16_t*)fallback_address(d,4,context,ram),
                 RK_O(5),RK_W(6),RK_W(7),RK_W(8),tid,threads); break;
    case LayerNorm:
      layernorm(RK_P(3),RK_P(4),RK_P(5),RK_O(6),RK_W(7),RK_W(8),
                fallback_float(d->word[9]),tid,threads); break;
    case RmsNorm:
      rmsnorm(RK_P(3),RK_P(4),RK_O(5),RK_W(6),RK_W(7),
              fallback_float(d->word[8]),tid,threads); break;
    case PatchEmbed:
      patch_embed(RK_P(3),RK_P(4),RK_P(5),RK_P(6),RK_O(7),
                  RK_W(8),RK_W(9),RK_W(10),RK_W(11),tid,threads); break;
    case BidirectionalAttention:
      bidirectional_mha(RK_P(3),RK_P(4),RK_P(5),RK_O(6),RK_O(7),
                        RK_W(8),RK_W(9),RK_W(10),tid,threads); break;
    case MaskedAttention:
      masked_gqa(RK_P(3),RK_P(4),RK_P(5),
                 (const RK_GLOBAL uint32_t*)fallback_address(d,6,context,ram),
                 RK_O(7),RK_O(8),RK_W(9),RK_W(10),RK_W(11),RK_W(12),RK_W(13),
                 tid,threads); break;
    case Gelu: gelu_tanh_values(RK_P(3),RK_O(4),RK_W(5),tid,threads); break;
    case Silu: silu_values(RK_P(3),RK_O(4),RK_W(5),tid,threads); break;
    case Residual: residual(RK_P(3),RK_P(4),RK_O(5),RK_W(6),tid,threads); break;
    case Bias: bias_add(RK_P(3),RK_P(4),RK_O(5),RK_W(6),RK_W(7),tid,threads); break;
    case Copy: copy_values(RK_P(3),RK_O(4),RK_W(5),tid,threads); break;
    case Scale:
      scale_values(RK_P(3),RK_O(4),RK_W(5),fallback_float(d->word[6]),tid,threads); break;
    case GatedActivation:
      gated_activation(RK_P(3),RK_P(4),RK_O(5),RK_W(6),RK_W(7)!=0,tid,threads); break;
    case Euler:
      euler_step(RK_P(3),RK_P(4),RK_O(5),RK_W(6),fallback_float(d->word[7]),tid,threads); break;
    case LinearExceptColumns:
      linear_except_columns(RK_P(3),RK_P(4),RK_O(5),RK_W(6),RK_W(7),RK_W(8),
                            RK_W(9),RK_W(10),tid,threads); break;
    default: return false;
  }
#undef RK_P
#undef RK_O
#undef RK_W
  return true;
}

extern "C" void rk_fallback_call(const RK_GLOBAL FallbackDescriptor*,void*,uint32_t,uint32_t);
}
