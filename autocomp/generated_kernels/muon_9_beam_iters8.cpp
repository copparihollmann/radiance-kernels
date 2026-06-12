// Baseline GELU (sigmoid-approx): one element per thread, C = x * sigmoid(1.702*x).
static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t total = a->M * a->N;
  for (uint32_t i = tid_in_threadblock; i < total; i += threads_per_threadblock) {
    const float x = a->A[i];
    const float sig = 1.0f / (1.0f + mu_exp(-1.702f * x));
    a->C[i] = x * sig;
  }
}
