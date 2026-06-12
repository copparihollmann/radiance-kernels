// Manual SMEM-tiled matmul: stage A (row-major) + B (transposed, +16 padded) into SMEM,
// cross-core sync, then 1 output/thread from SMEM. Fixes the autocomp candidate's barrier bug
// (it used mu_barrier(1,..) = per-core; cross-core SMEM staging needs ID 0 + a fence).
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
  const uint32_t nw = threads_per_threadblock / MU_NUM_THREADS;

  const uint32_t PAD = 16;
  const uint32_t smem_a = 0;
  const uint32_t smem_b = M * K * 4;
  const uint32_t kpad = K + PAD;

  // stage A row-major
  const __global float* ga = args->A;
  for (uint32_t i = tid_in_threadblock; i < M * K; i += threads_per_threadblock) {
    store_shared(smem_a + i * 4, 0, __builtin_bit_cast(uint32_t, ga[i]));
  }
  // stage B transposed + padded: B[k][col] -> smem_b + (col*kpad + k)*4
  const __global float* gb = args->B;
  for (uint32_t i = tid_in_threadblock; i < K * N; i += threads_per_threadblock) {
    uint32_t k = i / N, col = i % N;
    store_shared(smem_b + (col * kpad + k) * 4, 0, __builtin_bit_cast(uint32_t, gb[i]));
  }

  mu_barrier(0, nw);  // ID 0 = cross-core (both cores' stores visible); no fence (cyclotron has none)

  const uint32_t stride_b = kpad * 4;
  __global float* gc = args->C;
  for (uint32_t idx = tid_in_threadblock; idx < M * N; idx += threads_per_threadblock) {
    uint32_t row = idx / N, col = idx % N;
    uint32_t a_base = smem_a + row * K * 4;
    uint32_t b_base = smem_b + col * stride_b;
    float acc = 0.0f;
    for (uint32_t k = 0; k < K; k++) {
      float a_val = __builtin_bit_cast(float, load32_shared(a_base + k * 4));
      float b_val = __builtin_bit_cast(float, load32_shared(b_base + k * 4));
      acc += a_val * b_val;
    }
    gc[idx] = acc;
  }
}
