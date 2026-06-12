static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);

  // Specialize for vision transformer patch embedding:
  // OUT[16,4,4] = IN[3,64,64] * W[16,3,16,16], stride 16
  if (__builtin_expect(a->OC == 16 && a->C == 3 && a->KH == 16 && a->KW == 16 &&
                       a->OH == 4 && a->OW == 4 && a->stride == 16, 1)) {
    constexpr uint32_t OC = 16, C = 3, KH = 16, KW = 16, OH = 4, OW = 4, stride = 16;
    constexpr uint32_t CKK = C * KH * KW; // 768
    constexpr uint32_t P = OH * OW;       // 16
    constexpr uint32_t smem_w = P * CKK * 4; // SMEM offset for weights: 49152 bytes

    // (1) Stage W flat into SMEM[smem_w..]
    for (uint32_t i = tid_in_threadblock; i < OC * CKK; i += threads_per_threadblock) {
      store_shared(smem_w + i * 4, 0, __builtin_bit_cast(uint32_t, a->w[i]));
    }

    // (2) Gather im2col patches into SMEM[0..]. Explicit loop over P avoids heavy div/mod.
    const uint32_t Wd = a->W;
    const uint32_t HWd = a->H * Wd;
    
    // Loop over patch index `p` in [0, P)
    for (uint32_t p = 0; p < P; ++p) {
      // Compiler replaces these with mask/shift due to constexpr powers of two
      const uint32_t pr = p >> 2;  // p / OW
      const uint32_t pc = p & 3;   // p % OW
      const uint32_t out_p_base = p * CKK;

      // Each thread gathers one element per patch
      for (uint32_t ckk = tid_in_threadblock; ckk < CKK; ckk += threads_per_threadblock) {
        const uint32_t c = ckk >> 8;            // ckk / (KH * KW)
        const uint32_t kh = (ckk >> 4) & 15;    // (ckk / KW) % KH
        const uint32_t kw = ckk & 15;           // ckk % KW

        const uint32_t in_idx = c * HWd
                              + (pr * stride + kh) * Wd
                              + pc * stride + kw;
        
        // Store at patch base + ckk offset
        store_shared((out_p_base + ckk) * 4, 0, __builtin_bit_cast(uint32_t, a->in[in_idx]));
      }
    }
    
    // Cross-core synchronization after SMEM write
    mu_barrier(0, threads_per_threadblock / MU_NUM_THREADS);

    // (3) Matmul: OUT[oc][p] = sum_ckk W[oc][ckk] * patch[p][ckk]
    for (uint32_t idx = tid_in_threadblock; idx < OC * P; idx += threads_per_threadblock) {
      const uint32_t oc = idx >> 4; // idx / P
      const uint32_t p  = idx & 15; // idx % P
      
      const uint32_t wa_base = smem_w + oc * CKK * 4;
      const uint32_t pa_base = p * CKK * 4;

      float acc = 0.0f;
      uint32_t wa = wa_base;
      uint32_t pa = pa_base;
      
#pragma GCC unroll 1
      for (uint32_t k = 0; k < CKK; k++) {
        acc += __builtin_bit_cast(float, load32_shared(wa)) *
               __builtin_bit_cast(float, load32_shared(pa));
        wa += 4;
        pa += 4;
      }
      a->out[idx] = acc;
    }
  } else {
    // --- Generic fallback path ---
    const uint32_t CKK = a->C * a->KH * a->KW;
    const uint32_t P = a->OH * a->OW;
    const uint32_t smem_w = P * CKK * 4;

    // (1) Stage W flat into SMEM
    for (uint32_t i = tid_in_threadblock; i < a->OC * CKK; i += threads_per_threadblock) {
      store_shared(smem_w + i * 4, 0, __builtin_bit_cast(uint32_t, a->w[i]));
    }

    // (2) Gather patches
    const uint32_t Wd = a->W;
    const uint32_t HWd = a->H * Wd;
    const uint32_t stride = a->stride;
    
    for (uint32_t p = 0; p < P; ++p) {
      const uint32_t pr = p / a->OW;
      const uint32_t pc = p % a->OW;
      const uint32_t out_p_base = p * CKK;
      
      for (uint32_t ckk = tid_in_threadblock; ckk < CKK; ckk += threads_per_threadblock) {
        const uint32_t c = ckk / (a->KH * a->KW);
        const uint32_t kh = (ckk / a->KW) % a->KH;
        const uint32_t kw = ckk % a->KW;
        
        const uint32_t in_idx = c * HWd
                              + (pr * stride + kh) * Wd
                              + pc * stride + kw;
        store_shared((out_p_base + ckk) * 4, 0, __builtin_bit_cast(uint32_t, a->in[in_idx]));
      }
    }

    // Cross-core barrier
    mu_barrier(0, threads_per_threadblock / MU_NUM_THREADS);

    // (3) Matmul
    for (uint32_t idx = tid_in_threadblock; idx < a->OC * P; idx += threads_per_threadblock) {
      const uint32_t oc = idx / P;
      const uint32_t p = idx % P;
      
      const uint32_t wa_base = smem_w + oc * CKK * 4;
      const uint32_t pa_base = p * CKK * 4;

      float acc = 0.0f;
      uint32_t wa = wa_base;
      uint32_t pa = pa_base;
      
#pragma GCC unroll 1
      for (uint32_t k = 0; k < CKK; k++) {
        acc += __builtin_bit_cast(float, load32_shared(wa)) *
               __builtin_bit_cast(float, load32_shared(pa));
        wa += 4;
        pa += 4;
      }
      a->out[idx] = acc;
    }
  }
}
