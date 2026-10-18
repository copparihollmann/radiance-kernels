#pragma once

#ifdef RADIANCE_DEVICE
#include <mu_intrinsics.h>
#endif
#include "../pipeline_math.hpp"

// Model target: Rakanic/cyclotron + its pinned, unmodified libgemmini.
// One issuing lane owns MMIO; its warp cooperates on scratchpad tile packing.
// This does not claim support in the currently incompatible Radiance VPU RTL.
namespace model_chain {
namespace mx_model {

enum Storage { F32, F16, I8, I4, I6, F32_TIED, F16_TIED, I8_TIED, I5_TIED };
constexpr uint32_t A_ROW = 0, B_ROW = 2048, C_ROW = 4096;
// Keep scale exponents in the otherwise-unused fourth scratchpad bank. This
// also avoids lane-interleaved private-array indexing in the Muon toolchain.
constexpr uint32_t A_EXP = 0x18000, B_EXP = 0x18100;
extern RK_GLOBAL const uint8_t unit_scales[64];

#ifdef RADIANCE_DEVICE
inline void shared_store(uint32_t address, uint32_t value) {
  asm volatile("sw.shared %0, 0(%1)" :: "r"(value), "r"(address) : "memory");
}
inline uint32_t shared_load(uint32_t address) {
  uint32_t value;
  asm volatile("lw.shared %0, 0(%1)" : "=r"(value) : "r"(address) : "memory");
  return value;
}
inline void command(uint32_t funct, uint64_t rs1, uint64_t rs2) {
  shared_store(0x84010, uint32_t(rs1));
  shared_store(0x84014, uint32_t(rs1 >> 32));
  shared_store(0x84018, uint32_t(rs2));
  shared_store(0x8401c, uint32_t(rs2 >> 32));
  shared_store(0x84000, 0x7bu | (3u << 12) | (1u << 15) | (2u << 20) | (funct << 25));
}
inline void fence() { while (shared_load(0x84020)) asm volatile("nop"); }
#else
void shared_store(uint32_t address, uint32_t value);
uint32_t shared_load(uint32_t address);
void command(uint32_t funct, uint64_t rs1, uint64_t rs2);
void fence();
#endif

inline uint16_t bf16(float value) {
  union { float f; uint32_t u; } v{value};
  if ((v.u & 0x7fffffffu) > 0x7f800000u) return uint16_t((v.u >> 16) | 0x40u);
  return uint16_t((v.u + 0x7fffu + ((v.u >> 16) & 1u)) >> 16);
}
inline float from_bf16(uint16_t value) {
  union { uint32_t u; float f; } v{uint32_t(value) << 16};
  return v.f;
}
inline float power2(int exponent) {
  // Block scales are clamped to finite, normal FP32 powers of two.
  union { uint32_t u; float f; } v{uint32_t(exponent + 127) << 23};
  return v.f;
}
inline int block_exponent(float maximum) {
  union { float f; uint32_t u; } v{maximum};
  int e = int((v.u >> 23) & 255u) - 127;
  return e < -126 ? -126 : e > 126 ? 126 : e;
}
inline uint8_t fp8(float value) {
  union { float f; uint32_t u; } v{value};
  const uint32_t sign = (v.u >> 24) & 0x80u, magnitude = v.u & 0x7fffffffu;
  if (magnitude > 0x7f800000u) return uint8_t(sign | 127u);
  if (magnitude >= 0x43e00000u) return uint8_t(sign | 126u); // saturate at 448
  const uint32_t exponent = magnitude >> 23;
  if (exponent < 117) return uint8_t(sign); // below half the smallest subnormal
  const uint32_t significand = (magnitude & 0x7fffffu) | 0x800000u;
  // Round the integer significand directly, preserving ties to even. E4M3
  // subnormals use a fixed 2^-9 quantum; normal values retain three fraction
  // bits. A rounded carry naturally advances the encoded exponent.
  const uint32_t shift = exponent < 121 ? 141 - exponent : 20;
  const uint32_t rounded = (significand + ((1u << (shift-1)) - 1)
                           + ((significand >> shift) & 1u)) >> shift;
  return uint8_t(sign | (exponent < 121 ? rounded : ((exponent-121) << 3) + rounded));
}
inline int32_t packed(const RK_GLOBAL uint8_t* weight, uint32_t index, uint32_t bits) {
  uint32_t bit = index * bits, shift = bit & 7u;
  uint32_t raw = uint32_t(weight[bit >> 3]) >> shift;
  if (shift + bits > 8) raw |= uint32_t(weight[(bit >> 3) + 1]) << (8 - shift);
  raw &= (1u << bits) - 1;
  return raw < (1u << (bits - 1)) ? int32_t(raw) : int32_t(raw) - int32_t(1u << bits);
}
inline float weight_at(const RK_GLOBAL void* data, const RK_GLOBAL float* scales,
                       Storage storage, uint32_t k, uint32_t n,
                       uint32_t inner, uint32_t column) {
  bool tied = storage >= F32_TIED;
  uint32_t index = tied ? column * k + inner : inner * n + column;
  if (storage == F32 || storage == F32_TIED) return ((const RK_GLOBAL float*)data)[index];
  if (storage == F16 || storage == F16_TIED)
    return fp16_to_fp32(((const RK_GLOBAL uint16_t*)data)[index]);
  if (storage == I8 || storage == I8_TIED)
    return float(((const RK_GLOBAL int8_t*)data)[index]) * scales[column];
  uint32_t bits = storage == I4 ? 4 : storage == I6 ? 6 : 5;
  uint32_t scale = storage == I5_TIED ? column * (k / 64) + inner / 64 : column;
  return float(packed((const RK_GLOBAL uint8_t*)data, index, bits)) * scales[scale];
}
inline void initialize() {
  command(7, 0, 0);
  command(30, 0, 0); // LUT disabled; direct E4M3 codes
  command(0, (1ull << 16) | (3ull << 14) | (1ull << 2), 1ull << 48);
#ifdef RADIANCE_DEVICE
  uint64_t address = 0x100000000ull | uint32_t(uintptr_t(unit_scales));
#else
  uint64_t address = 0x100001000ull; // native callback's read-only scale mapping
#endif
  command(27, address, 16);
  command(27, address, (1ull << 32) | 16);
  fence();
}

} // namespace mx_model

// Block-scaled E4M3: 16x16 output tiles, K blocks of 32. Normalized operands
// have magnitude < 2. Unit E8M0 scales feed MX; Muon restores block powers of
// two and accumulates K-block results in FP32. No full FP8 checkpoint is stored.
inline void mx_vpu_linear(const RK_GLOBAL float* input,
                          const RK_GLOBAL void* weights,
                          const RK_GLOBAL float* scales,
                          RK_GLOBAL float* output, mx_model::Storage storage,
                          uint32_t m, uint32_t k, uint32_t n,
                          uint32_t tid, uint32_t threads) {
#ifdef RADIANCE_DEVICE
  // Keep all participants in one warp: SIMT reconvergence orders the owner
  // command/fence and the other lanes' packing without a cross-warp barrier.
  const uint32_t lanes = threads < MU_NUM_THREADS ? threads : MU_NUM_THREADS;
#else
  const uint32_t lanes = 1; // host stage runner calls lanes sequentially
#endif
  if (tid >= lanes) return;
  using namespace mx_model;
  if (tid == 0) initialize();
  for (uint32_t i = tid; i < m * n; i += lanes) output[i] = 0;
  for (uint32_t row = 0; row < m; row += 16)
    for (uint32_t col = 0; col < n; col += 16)
      for (uint32_t block = 0; block < k; block += 32) {
        for (uint32_t r = tid; r < 16; r += lanes) {
          float maximum = 0;
          for (uint32_t t = 0; t < 32 && row+r < m && block+t < k; ++t) {
            float v = __builtin_fabsf(input[(row+r)*k+block+t]);
            if (v > maximum) maximum = v;
          }
          shared_store(A_EXP+4*r, uint32_t(maximum ? block_exponent(maximum) : 0));
        }
        for (uint32_t c = tid; c < 16; c += lanes) {
          float maximum = 0;
          for (uint32_t t = 0; t < 32 && col+c < n && block+t < k; ++t) {
            float v = __builtin_fabsf(weight_at(weights, scales, storage, k, n, block+t, col+c));
            if (v > maximum) maximum = v;
          }
          shared_store(B_EXP+4*c, uint32_t(maximum ? block_exponent(maximum) : 0));
        }
        for (uint32_t t = 0; t < 32; t += 4)
          for (uint32_t r = tid; r < 16; r += lanes) {
            // A's two 16x16 tiles are consecutive scratchpad row groups.
            uint32_t word = 0;
            for (uint32_t b = 0; b < 4; ++b) {
              float v = row+r < m && block+t+b < k ? input[(row+r)*k+block+t+b] : 0;
              word |= uint32_t(fp8(v * power2(-int32_t(shared_load(A_EXP+4*r))))) << (8*b);
            }
            shared_store(16*((t/16)*16+r) + t%16, word);
          }
        for (uint32_t packed_word = tid; packed_word < 128; packed_word += lanes) {
            const uint32_t t = packed_word / 4, c = 4 * (packed_word % 4);
            uint32_t word = 0;
            for (uint32_t b = 0; b < 4; ++b) {
              float v = block+t < k && col+c+b < n ?
                weight_at(weights, scales, storage, k, n, block+t, col+c+b) : 0;
              word |= uint32_t(fp8(v * power2(-int32_t(shared_load(B_EXP+4*(c+b)))))) << (8*b);
            }
            shared_store(16*(B_ROW+t)+c, word);
        }
#ifdef RADIANCE_DEVICE
        mu_fence_smem(); // all packed words must retire before MMIO consumes them
#endif
        // Reset the scale address cursors for this independent K block.
        if (tid == 0) {
          command(26, (1ull << 62) | (2ull << 51) | (1ull << 42) | (1ull << 33),
                  (1ull << 16) | 1);
          command(10, 0, 0);
          command(9, 0, (2ull << 32) | (1ull << 16) | 1);
          command(24, A_ROW, B_ROW+32);
          command(8, 0, (uint64_t(C_ROW) << 32) | (1ull << 9) | (7ull << 3));
          fence();
        }
        for (uint32_t r = 0; r < 16 && row+r < m; ++r)
          for (uint32_t c = tid; c < 16 && col+c < n; c += lanes) {
            uint32_t word = shared_load(16*C_ROW + r*32 + (c/2)*4);
            float partial = from_bf16(uint16_t(word >> (16*(c%2))));
            // Sequential multiplies avoid constructing an out-of-range FP32 power.
            output[(row+r)*n+col+c] +=
                (partial * power2(int32_t(shared_load(A_EXP+4*r)))) *
                power2(int32_t(shared_load(B_EXP+4*c)));
          }
      }
}

inline void mx_vpu_add(const RK_GLOBAL float* left, const RK_GLOBAL float* right,
                       RK_GLOBAL float* output, uint32_t count, uint32_t right_width,
                       uint32_t tid, uint32_t) {
  if (tid != 0) return;
  using namespace mx_model;
  for (uint32_t start = 0; start < count; start += 256) {
    uint32_t size = count-start < 256 ? count-start : 256;
    uint32_t rows = (size+7)/8;
    for (uint32_t p = 0; p < rows*8; p += 2) {
      uint32_t a = 0, b = 0;
      for (uint32_t lane = 0; lane < 2; ++lane) if (p+lane < size) {
        a |= uint32_t(bf16(left[start+p+lane])) << (16*lane);
        b |= uint32_t(bf16(right[(start+p+lane)%right_width])) << (16*lane);
      }
      shared_store(16*A_ROW+2*p, a);
      shared_store(16*B_ROW+2*p, b);
    }
    command(33, uint64_t(A_ROW) | (uint64_t(B_ROW) << 14) |
            (uint64_t(C_ROW) << 28) | (uint64_t(rows) << 42), 1ull << 5);
    fence();
    for (uint32_t p = 0; p < size; ++p) {
      uint32_t word = shared_load(16*C_ROW+4*(p/2));
      output[start+p] = from_bf16(uint16_t(word >> (16*(p%2))));
    }
  }
}

} // namespace model_chain
