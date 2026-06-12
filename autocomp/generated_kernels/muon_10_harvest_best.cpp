static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  auto* args = reinterpret_cast<KernelArgs*>(raw_arg);

  // Precompute initial row/col using division once
  uint32_t row = tid_in_threadblock / args->N;
  uint32_t col = tid_in_threadblock % args->N;

  // Precompute strides to avoid repeated division
  const uint32_t row_stride = threads_per_threadblock / args->N;
  const uint32_t col_stride = threads_per_threadblock % args->N;

  const uint32_t total_elements = args->M * args->N;
  const uint32_t K = args->K;

  // Assign pointers with correct address space qualifier
  const __global float* A = args->A;
  const __global float* B = args->B;
  __global float* C = args->C;

  // Loop over output matrix elements assigned to this thread
  for (uint32_t idx = tid_in_threadblock; idx < total_elements; idx += threads_per_threadblock) {
    float acc = 0.0f;

    // Iterate over reduction dimension
    for (uint32_t k = 0; k < K; ++k) {
      acc += A[row * K + k] * B[k * args->N + col];
    }

    // Write result
    C[idx] = acc;

    // Update (row, col) incrementally instead of recomputing from idx
    col += col_stride;
    row += row_stride;

    // Handle carry-over when column wraps around
    if (col >= args->N) {
      col -= args->N;
      row += 1;
    }
  }
}
