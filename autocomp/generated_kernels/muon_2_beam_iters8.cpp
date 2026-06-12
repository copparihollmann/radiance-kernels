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
  const uint32_t num_warps = threads_per_threadblock / MU_NUM_THREADS;
  const float scale = 0.125f;

  const uint32_t total = seq * seq;

  // Phase 1: scratch[i*seq+j] = scale * dot(Q[i,:], K[j,:])
  for (uint32_t idx = tid_in_threadblock; idx < total; idx += threads_per_threadblock) {
    uint32_t i = idx / seq;
    uint32_t j = idx % seq;
    __global float* Qrow = a->Q + i * d;
    __global float* Krow = a->K + j * d;
    float acc = 0.0f;
    for (uint32_t k = 0; k < d; k++) {
      acc += Qrow[k] * Krow[k];
    }
    a->scratch[idx] = acc * scale;
  }
  mu_barrier(0, num_warps);

  // Phase 2: softmax each row of scratch in-place
  for (uint32_t row = tid_in_threadblock; row < seq; row += threads_per_threadblock) {
    __global float* srow = a->scratch + row * seq;

    // find max
    float m = srow[0];
    for (uint32_t j = 1; j < seq; j++) {
      float v = srow[j];
      if (v > m) m = v;
    }

    // exp and sum
    float sum = 0.0f;
    for (uint32_t j = 0; j < seq; j++) {
      float e = mu_exp(srow[j] - m);
      srow[j] = e;
      sum += e;
    }

    // normalize
    float inv = 1.0f / sum;
    for (uint32_t j = 0; j < seq; j++) {
      srow[j] *= inv;
    }
  }
  mu_barrier(1, num_warps);

  // Phase 3: O[i*d+k] = sum_j scratch[i*seq+j] * V[j*d+k]
  const uint32_t out_total = seq * d;
  for (uint32_t idx = tid_in_threadblock; idx < out_total; idx += threads_per_threadblock) {
    uint32_t i = idx / d;
    uint32_t k = idx % d;
    __global float* Vk = a->V + k;
    __global float* Prow = a->scratch + i * seq;
    float acc = 0.0f;
    for (uint32_t j = 0; j < seq; j++) {
      acc += Prow[j] * Vk[j * d];
    }
    a->O[idx] = acc;
  }
}
