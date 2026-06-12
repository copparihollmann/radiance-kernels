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
  const uint32_t num_warps = threads_per_threadblock / MU_NUM_THREADS;
  const uint32_t PAD = 16;

  // SMEM layout:
  //   smem_a: A in row-major, M*K floats
  //   smem_b: B in column-major with padding, N*(K+PAD) floats
  const uint32_t smem_a = 0;
  const uint32_t smem_b = M * K * 4;

  // Stage A into SMEM (row-major)
  {
    const __global float* A = args->A;
    uint32_t i = tid_in_threadblock;
    const uint32_t total_a = M * K;
    while (i < total_a) {
      store_shared(smem_a + i * 4, 0, __builtin_bit_cast(uint32_t, A[i]));
      i += threads_per_threadblock;
    }
  }

  // Stage B into SMEM (column-major with padding)
  // B[k][col] = B_global[k * N + col], stored at smem_b + col*(K+PAD)*4 + k*4
  {
    const __global float* B = args->B;
    const uint32_t col = tid_in_threadblock % N;
    uint32_t k_cnt = tid_in_threadblock / N;
    const uint32_t k_stride = threads_per_threadblock / N;
    const uint32_t b_col_base = smem_b + col * (K + PAD) * 4;
    const uint32_t kp_stride = k_stride * 4;
    uint32_t b_smem_addr = b_col_base + k_cnt * 4;
    while (k_cnt < K) {
      store_shared(b_smem_addr, 0, __builtin_bit_cast(uint32_t, B[k_cnt * N + col]));
      b_smem_addr += kp_stride;
      k_cnt += k_stride;
    }
  }

  mu_barrier(1, num_warps);

  // Compute C = A * B
  {
    __global float* C = args->C;
    const uint32_t total_c = M * N;
    uint32_t idx = tid_in_threadblock;
    while (idx < total_c) {
      const uint32_t row = idx / N;
      const uint32_t col = idx % N;
      uint32_t a_addr = smem_a + row * K * 4;
      uint32_t b_addr = smem_b + col * (K + PAD) * 4;
      float acc = 0.0f;
      uint32_t k = 0;
      while (k < K) {
        float a_val = __builtin_bit_cast(float, load32_shared(a_addr));
        float b_val = __builtin_bit_cast(float, load32_shared(b_addr));
        acc += a_val * b_val;
        a_addr += 4;
        b_addr += 4;
        k++;
      }
      C[idx] = acc;
      idx += threads_per_threadblock;
    }
  }
}
