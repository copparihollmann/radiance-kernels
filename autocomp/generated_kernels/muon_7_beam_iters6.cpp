static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);

  const uint32_t seq = a->seq;
  const uint32_t d   = a->d;
  const float scale  = 0.125f;

  // SMEM layout: K_s[seq*d floats] | V_s[seq*d floats] | O_s[threads_per_threadblock * d floats]
  const uint32_t smem_k = 0;
  const uint32_t smem_v = smem_k + seq * d * 4;
  const uint32_t smem_o = smem_v + seq * d * 4;

  const uint32_t total_kv = seq * d;

  // Stage K into SMEM
  for (uint32_t i = tid_in_threadblock; i < total_kv; i += threads_per_threadblock) {
    store_shared(smem_k + i * 4, 0, __builtin_bit_cast(uint32_t, a->K[i]));
  }
  // Stage V into SMEM
  for (uint32_t i = tid_in_threadblock; i < total_kv; i += threads_per_threadblock) {
    store_shared(smem_v + i * 4, 0, __builtin_bit_cast(uint32_t, a->V[i]));
  }

  const uint32_t num_warps = threads_per_threadblock / MU_NUM_THREADS;
  mu_barrier(1, num_warps);

  // Each thread handles one or more query rows (strided by threads_per_threadblock)
  for (uint32_t i = tid_in_threadblock; i < seq; i += threads_per_threadblock) {

    // O accumulator in SMEM for this thread
    const uint32_t obase = smem_o + tid_in_threadblock * d * 4;

    // Zero out O accumulator
    for (uint32_t k = 0; k < d; k++) {
      store_shared(obase + k * 4, 0, __builtin_bit_cast(uint32_t, 0.0f));
    }

    float m = -3.0e38f;
    float l  = 0.0f;

    const __global float* q = a->Q + i * d;

    for (uint32_t j = 0; j < seq; j++) {
      // Compute dot product q . K[j]
      const uint32_t kbase = smem_k + j * d * 4;
      float s = 0.0f;
      for (uint32_t k = 0; k < d; k++) {
        float qk  = q[k];
        float kk  = __builtin_bit_cast(float, load32_shared(kbase + k * 4));
        s += qk * kk;
      }
      s *= scale;

      // Online softmax update
      float mnew = (s > m) ? s : m;
      float corr = (m < -1.0e37f) ? 0.0f : mu_exp(m - mnew);
      float p    = mu_exp(s - mnew);
      l = l * corr + p;

      // Update O accumulator
      const uint32_t vbase = smem_v + j * d * 4;
      for (uint32_t k = 0; k < d; k++) {
        uint32_t oaddr = obase + k * 4;
        float ok = __builtin_bit_cast(float, load32_shared(oaddr));
        float vk = __builtin_bit_cast(float, load32_shared(vbase + k * 4));
        ok = ok * corr + p * vk;
        store_shared(oaddr, 0, __builtin_bit_cast(uint32_t, ok));
      }

      m = mnew;
    }

    // Write normalized output
    float inv = 1.0f / l;
    __global float* out = a->O + i * d;
    for (uint32_t k = 0; k < d; k++) {
      float ok = __builtin_bit_cast(float, load32_shared(obase + k * 4));
      out[k] = ok * inv;
    }
  }
}
