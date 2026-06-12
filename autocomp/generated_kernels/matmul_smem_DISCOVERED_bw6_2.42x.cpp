static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* args = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t M = args->M;
  const uint32_t N = args->N;
  const uint32_t K = args->K;

  const uint32_t num_warps = threads_per_threadblock / 16;
  const uint32_t smem_a = 0;
  const uint32_t smem_b = M * K * 4;
  const uint32_t PAD = 16;

  // Load A into shared memory (row-major)
  for (uint32_t idx = tid_in_threadblock; idx < M * K; idx += threads_per_threadblock) {
    const uint32_t addr = smem_a + idx * 4;
    const float val = args->A[idx];
    union { uint32_t u; float f; } c;
    c.f = val;
    store_shared(addr, 0, c.u);
  }

  // Load B into shared memory (column-major with padding)
  for (uint32_t idx = tid_in_threadblock; idx < K * N; idx += threads_per_threadblock) {
    const uint32_t k = idx / N;
    const uint32_t n = idx % N;
    const uint32_t smem_idx = n * (K + PAD) + k;
    const uint32_t addr = smem_b + smem_idx * 4;
    const float val = args->B[idx];
    union { uint32_t u; float f; } c;
    c.f = val;
    store_shared(addr, 0, c.u);
  }

  // Cross-core barrier: ensure all SMEM writes are visible
  mu_barrier(0, num_warps);

  // Compute outputs
  for (uint32_t idx = tid_in_threadblock; idx < M * N; idx += threads_per_threadblock) {
    const uint32_t m = idx / N;
    const uint32_t n = idx % N;

    float acc = 0.0f;
    const uint32_t a_row_base = smem_a + m * K * 4;
    const uint32_t b_col_base = smem_b + n * (K + PAD) * 4;

#pragma GCC unroll 1
    for (uint32_t k = 0; k < K; ++k) {
      const uint32_t a_addr = a_row_base + k * 4;
      const uint32_t b_addr = b_col_base + k * 4;
      union { uint32_t u; float f; } a_union, b_union;
      a_union.u = load32_shared(a_addr);
      b_union.u = load32_shared(b_addr);
      acc += a_union.f * b_union.f;
    }

    args->C[idx] = acc;
  }
}
