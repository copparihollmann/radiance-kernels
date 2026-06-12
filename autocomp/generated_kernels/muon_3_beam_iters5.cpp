static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t seq = a->seq;
  const uint32_t d = a->d;
  const uint32_t num_warps = threads_per_threadblock / MU_NUM_THREADS;
  const float scale = 0.125f;

  // Phase 1: scores
  for (uint32_t idx = tid_in_threadblock; idx < seq * seq; idx += threads_per_threadblock) {
    const uint32_t i = idx / seq;
    const uint32_t j = idx % seq;
    float acc = 0.0f;
    for (uint32_t k = 0; k < d; k++) {
      acc += a->Q[i * d + k] * a->K[j * d + k];
    }
    a->scratch[idx] = acc * scale;
  }
  mu_barrier(0, num_warps); // Cross-core barrier for scratch visibility

  // Phase 2: softmax per row
  for (uint32_t row = tid_in_threadblock; row < seq; row += threads_per_threadblock) {
    float m = a->scratch[row * seq];
    for (uint32_t j = 1; j < seq; j++) {
      const float v = a->scratch[row * seq + j];
      m = (v > m) ? v : m; // Branchless max reduction
    }

    float sum = 0.0f;
    for (uint32_t j = 0; j < seq; j++) {
      const float e = mu_exp(a->scratch[row * seq + j] - m);
      a->scratch[row * seq + j] = e;
      sum += e;
    }

    const float inv = 1.0f / sum;
    for (uint32_t j = 0; j < seq; j++) {
      a->scratch[row * seq + j] *= inv;
    }
  }
  mu_barrier(0, num_warps); // Cross-core barrier for softmax output visibility

  // Phase 3: O = P V
  for (uint32_t idx = tid_in_threadblock; idx < seq * d; idx += threads_per_threadblock) {
    const uint32_t i = idx / d;
    const uint32_t k = idx % d;
    float acc = 0.0f;
    for (uint32_t j = 0; j < seq; j++) {
      acc += a->scratch[i * seq + j] * a->V[j * d + k];
    }
    a->O[idx] = acc;
  }
}
