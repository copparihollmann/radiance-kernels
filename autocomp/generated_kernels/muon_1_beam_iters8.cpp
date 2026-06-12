#include <stdint.h>

// Note: KernelArgs is defined by the harness. Do not redefine it here.

static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  
  // Define compile-time constants for fixed ViT patch-embedding shapes
  constexpr uint32_t C = 3;
  constexpr uint32_t KH = 16;
  constexpr uint32_t KW = 16;
  constexpr uint32_t OC = 16;
  constexpr uint32_t OH = 4;
  constexpr uint32_t OW = 4;
  constexpr uint32_t stride = 16;
  constexpr uint32_t CKK = C * KH * KW; // 768
  constexpr uint32_t P = OH * OW;       // 16
  
  // Pad SMEM patch row by +16 floats to make the stride 49 * 64B lines.
  // Since 49 is odd, consecutive rows cycle banks cleanly, avoiding bank conflicts 
  // while preserving natural 64-byte cache line alignment.
  constexpr uint32_t PADDED_CKK = CKK + 16; // 784
  constexpr uint32_t smem_w = P * PADDED_CKK * 4; // 16 * 784 * 4 = 50176 bytes

  // Assign address-space qualified pointers (Rule 23)
  __global float* in_ptr = a->in;
  __global float* w_ptr = a->w;
  __global float* out_ptr = a->out;

  // (1) Stage weights (W) flat into SMEM starting after patch buffers.
  {
    constexpr uint32_t total_w = OC * CKK; // 16 * 768 = 12288
    for (uint32_t i = tid_in_threadblock; i < total_w; i += threads_per_threadblock) {
      store_shared(smem_w + i * 4, 0, __builtin_bit_cast(uint32_t, w_ptr[i]));
    }
  }

  // (2) Gather im2col patches into SMEM using fast division multipliers and bitwise shifts.
  {
    const uint32_t Wd = a->W;
    const uint32_t HWd = a->H * Wd;
    constexpr uint32_t total_patches_size = P * CKK; // 16 * 768 = 12288

    for (uint32_t i = tid_in_threadblock; i < total_patches_size; i += threads_per_threadblock) {
      // Fast division/modulo by 768:
      // Since CKK is 768, we can do (i >> 8) / 3.
      // Multiplying by 342 and shifting right by 10 divides by 3 exactly for all values < 48.
      const uint32_t p = ((i >> 8) * 342) >> 10;
      const uint32_t ckk = i - p * 768;
      
      const uint32_t c = ckk >> 8;               // ckk / 256
      const uint32_t kh = (ckk >> 4) & 15;        // (ckk / 16) % 16
      const uint32_t kw = ckk & 15;               // ckk % 16
      
      const uint32_t p_h = p >> 2;                // p / 4
      const uint32_t p_w = p & 3;                 // p % 4
      
      const uint32_t in_idx = c * HWd
                            + (p_h * stride + kh) * Wd
                            + p_w * stride + kw;
      
      // Store into padded SMEM structure to prevent bank conflicts
      const uint32_t store_addr = (p * PADDED_CKK + ckk) * 4;
      store_shared(store_addr, 0, __builtin_bit_cast(uint32_t, in_ptr[in_idx]));
    }
  }

  // Synchronize across both cores (ID 0 = cross-core cluster barrier)
  // No mu_fence_smem() is called here to ensure RTL timing simulation terminates safely (Rule 26)
  mu_barrier(0, threads_per_threadblock / 16); 

  // (3) Matmul with 2D (2 rows x 2 cols) Register Tiling (Strategy 4)
  {
    const uint32_t tile_id = tid_in_threadblock;
    if (tile_id < 64) {
      const uint32_t tile_row = tile_id >> 3; // tile_id / 8
      const uint32_t tile_col = tile_id & 7;  // tile_id % 8

      const uint32_t oc0 = tile_row * 2;
      const uint32_t oc1 = oc0 + 1;
      const uint32_t p0 = tile_col * 2;
      const uint32_t p1 = p0 + 1;

      uint32_t wa0 = smem_w + oc0 * 3072;     // oc0 * 768 * 4
      uint32_t pa0 = p0 * (PADDED_CKK * 4);   // p0 * 784 * 4

      float acc00 = 0.0f;
      float acc01 = 0.0f;
      float acc10 = 0.0f;
      float acc11 = 0.0f;

      // Keep loops strictly rolled to adhere to register constraints (Rules 9 & 11)
      #pragma GCC unroll 1
      for (uint32_t k = 0; k < CKK; k++) {
        // Load patch values (pa0 + 3136 is exactly pa1)
        const float p0_val = __builtin_bit_cast(float, load32_shared(pa0));
        const float p1_val = __builtin_bit_cast(float, load32_shared(pa0 + 3136));

        // Load weight values (wa0 + 3072 is exactly wa1)
        const float w0_val = __builtin_bit_cast(float, load32_shared(wa0));
        const float w1_val = __builtin_bit_cast(float, load32_shared(wa0 + 3072));

        acc00 += w0_val * p0_val;
        acc01 += w0_val * p1_val;
        acc10 += w1_val * p0_val;
        acc11 += w1_val * p1_val;

        wa0 += 4;
        pa0 += 4;
      }

      out_ptr[oc0 * 16 + p0] = acc00;
      out_ptr[oc0 * 16 + p1] = acc01;
      out_ptr[oc1 * 16 + p0] = acc10;
      out_ptr[oc1 * 16 + p1] = acc11;
    }
  }
}
