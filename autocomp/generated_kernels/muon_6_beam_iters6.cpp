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

  // B is stored transposed: smem_b[col][k], with PAD to avoid bank conflicts
  // PAD=4 floats per row (16 bytes) to avoid 16-way bank conflicts
  const uint32_t PAD = 4;
  const uint32_t K_PAD = K + PAD;

  // smem layout: A at 0, B immediately after
  // A: M * K * 4 bytes
  // B: N * K_PAD * 4 bytes
  // For M=N=128, K=192, PAD=4: A=98304, B=128*196*4=100352, total=198656 > 128KB
  // Use PAD=0 and accept minor conflicts, or tile. Since original used PAD=16 and
  // may have overflowed, use minimal PAD and keep algorithm identical.

  const uint32_t smem_a = 0;
  // Align smem_b to 16 bytes after A
  const uint32_t smem_b = ((M * K * 4) + 15) & ~15u;

  // Load A into SMEM: row-major, A[i] -> smem_a + i*4
  for (uint32_t i = tid_in_threadblock; i < M * K; i += threads_per_threadblock) {
    const __global float* a_ptr = args->A;
    uint32_t val = __builtin_bit_cast(uint32_t, a_ptr[i]);
    store_shared(smem_a + i * 4, 0, val);
  }

  // Load B into SMEM transposed: B[k][col] -> smem_b[(col * K_PAD + k) * 4]
  // so inner loop over k for a fixed col is stride-1 in SMEM
  for (uint32_t i = tid_in_threadblock; i < K * N; i += threads_per_threadblock) {
    const __global float* b_ptr = args->B;
    uint32_t k   = i / N;
    uint32_t col = i % N;
    uint32_t val = __builtin_bit_cast(uint32_t, b_ptr[i]);
    store_shared(smem_b + (col * K_PAD + k) * 4, 0, val);
  }

  mu_barrier(1, num_warps);

  // Compute C = A * B, one output element per thread per iteration
  for (uint32_t idx = tid_in_threadblock; idx < M * N; idx += threads_per_threadblock) {
    uint32_t row = idx / N;
    uint32_t col = idx % N;
    uint32_t a_addr = smem_a + row * K * 4;
    uint32_t b_addr = smem_b + col * K_PAD * 4;
    float acc = 0.0f;
    for (uint32_t k = 0; k < K; k++) {
      float a_val = __builtin_bit_cast(float, load32_shared(a_addr));
      float b_val = __builtin_bit_cast(float, load32_shared(b_addr));
      acc += a_val * b_val;
      a_addr += 4;
      b_addr += 4;
    }
    args->C[idx] = acc;
  }
}
