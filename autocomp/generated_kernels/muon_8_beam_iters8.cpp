static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t cols = a->cols;
  const float eps = 1.0e-5f;

  // SMEM layout:
  //   smem_gamma: offset 0,         size = cols * 4 bytes
  //   smem_beta:  offset cols*4,    size = cols * 4 bytes
  const uint32_t smem_gamma_base = 0x00000000u;
  const uint32_t smem_beta_base  = smem_gamma_base + cols * 4u;

  // Cooperatively load gamma and beta into SMEM.
  {
    const __global float* g_gamma = a->gamma;
    const __global float* g_beta  = a->beta;
    uint32_t idx = tid_in_threadblock;
    while (idx < cols) {
      uint32_t gval, bval;
      uint32_t g_addr = reinterpret_cast<uint32_t>(g_gamma + idx);
      uint32_t b_addr = reinterpret_cast<uint32_t>(g_beta  + idx);
      asm volatile("lw.global %0, 0(%1)" : "=r"(gval) : "r"(g_addr) : "memory");
      asm volatile("lw.global %0, 0(%1)" : "=r"(bval) : "r"(b_addr) : "memory");
      uint32_t sg_addr = smem_gamma_base + idx * 4u;
      uint32_t sb_addr = smem_beta_base  + idx * 4u;
      store_shared(sg_addr, 0, gval);
      store_shared(sb_addr, 0, bval);
      idx += threads_per_threadblock;
    }
  }

  mu_barrier(0, 8);

  const float rcols = 1.0f / (float)cols;

  for (uint32_t row = tid_in_threadblock; row < a->rows; row += threads_per_threadblock) {
    const uint32_t base = row * cols;
    const __global float* in_row = a->in + base;

    // Pass 1: compute sum and sum_sq
    float sum    = 0.0f;
    float sum_sq = 0.0f;
    for (uint32_t j = 0; j < cols; j++) {
      uint32_t addr = reinterpret_cast<uint32_t>(in_row + j);
      uint32_t bits;
      asm volatile("lw.global %0, 0(%1)" : "=r"(bits) : "r"(addr) : "memory");
      float x = __builtin_bit_cast(float, bits);
      sum    += x;
      sum_sq += x * x;
    }

    const float mean    = sum * rcols;
    const float var     = sum_sq * rcols - mean * mean;
    const float inv_std = mu_rsqrt(var + eps);

    // Pass 2: normalize and write output
    __global float* out_row = a->out + base;
    for (uint32_t j = 0; j < cols; j++) {
      uint32_t in_addr = reinterpret_cast<uint32_t>(in_row + j);
      uint32_t in_bits;
      asm volatile("lw.global %0, 0(%1)" : "=r"(in_bits) : "r"(in_addr) : "memory");
      float x = __builtin_bit_cast(float, in_bits);

      uint32_t gamma_bits = load32_shared(smem_gamma_base + j * 4u);
      uint32_t beta_bits  = load32_shared(smem_beta_base  + j * 4u);
      float g = __builtin_bit_cast(float, gamma_bits);
      float b = __builtin_bit_cast(float, beta_bits);

      float result = (x - mean) * inv_std * g + b;
      uint32_t out_addr = reinterpret_cast<uint32_t>(out_row + j);
      uint32_t out_bits = __builtin_bit_cast(uint32_t, result);
      asm volatile("sw.global %1, 0(%0)" :: "r"(out_addr), "r"(out_bits) : "memory");
    }
  }
}
