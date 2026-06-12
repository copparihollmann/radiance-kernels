#include <stdint.h>
#include <type_traits>

struct KernelArgs {
  __global float* Q;
  __global float* K;
  __global float* V;
  __global float* O;
  uint32_t seq;
  uint32_t d;
};

static inline void kernel_body(
  void* raw_arg,
  uint32_t tid_in_threadblock,
  uint32_t threads_per_threadblock,
  uint32_t threadblock_id
) {
  (void)threadblock_id;
  (void)threads_per_threadblock;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t seq = a->seq;

  // Early exit if tid >= seq
  if (tid_in_threadblock >= seq) return;

  const uint32_t tid = tid_in_threadblock;

  // Shared memory layout: Total 128 KiB maximum
  // Phase 1 & 2:
  // Q: [128][64] = 32 KiB at 0x0000_0000
  // K: [128][64] = 32 KiB at 0x0000_8000
  // S: [128][128] = 64 KiB at 0x0001_0000
  const uint32_t smem_Q_base = 0x00000000;
  const uint32_t smem_K_base = 0x00008000;
  const uint32_t smem_S_base = 0x00010000;

  // Phase 1: Parallel coalesced load of Q and K into SMEM
  for (uint32_t step = 0; step < 64; step++) {
    uint32_t idx = step * 128 + tid;
    store_shared(smem_Q_base + idx * 4, 0, __builtin_bit_cast(uint32_t, a->Q[idx]));
    store_shared(smem_K_base + idx * 4, 0, __builtin_bit_cast(uint32_t, a->K[idx]));
  }

  // Cross-core synchronization to ensure Q and K are fully visible
  mu_barrier(0, 8);

  // Phase 2: Compute S_ij = 0.125 * dot(Q_i, K_j)
  for (uint32_t j_outer = 0; j_outer < 128; j_outer += 4) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint32_t k = 0; k < 64; k++) {
      float q_val = __builtin_bit_cast(float, load32_shared(smem_Q_base + (tid * 64 + k) * 4));
      float k0 = __builtin_bit_cast(float, load32_shared(smem_K_base + ((j_outer + 0) * 64 + k) * 4));
      float k1 = __builtin_bit_cast(float, load32_shared(smem_K_base + ((j_outer + 1) * 64 + k) * 4));
      float k2 = __builtin_bit_cast(float, load32_shared(smem_K_base + ((j_outer + 2) * 64 + k) * 4));
      float k3 = __builtin_bit_cast(float, load32_shared(smem_K_base + ((j_outer + 3) * 64 + k) * 4));
      acc0 += q_val * k0;
      acc1 += q_val * k1;
      acc2 += q_val * k2;
      acc3 += q_val * k3;
    }
    store_shared(smem_S_base + (tid * 128 + j_outer + 0) * 4, 0, __builtin_bit_cast(uint32_t, acc0 * 0.125f));
    store_shared(smem_S_base + (tid * 128 + j_outer + 1) * 4, 0, __builtin_bit_cast(uint32_t, acc1 * 0.125f));
    store_shared(smem_S_base + (tid * 128 + j_outer + 2) * 4, 0, __builtin_bit_cast(uint32_t, acc2 * 0.125f));
    store_shared(smem_S_base + (tid * 128 + j_outer + 3) * 4, 0, __builtin_bit_cast(uint32_t, acc3 * 0.125f));
  }

  // Sync to ensure all threads are done reading Q and K
  mu_barrier(0, 8);

  // Phase 3: Overwrite Q and K SMEM spaces with V
  // V: [128][64] = 32 KiB at 0x0000_0000
  const uint32_t smem_V_base = 0x00000000;
  for (uint32_t step = 0; step < 64; step++) {
    uint32_t idx = step * 128 + tid;
    store_shared(smem_V_base + idx * 4, 0, __builtin_bit_cast(uint32_t, a->V[idx]));
  }

  // Sync to ensure V is fully staged
  mu_barrier(0, 8);

  // Phase 4: Row-local Softmax statistics
  float m = -10000.0f;
  for (uint32_t j = 0; j < 128; j++) {
    float s = __builtin_bit_cast(float, load32_shared(smem_S_base + (tid * 128 + j) * 4));
    if (s > m) m = s;
  }

  float d = 0.0f;
  for (uint32_t j = 0; j < 128; j++) {
    float s = __builtin_bit_cast(float, load32_shared(smem_S_base + (tid * 128 + j) * 4));
    float e = mu_exp(s - m);
    store_shared(smem_S_base + (tid * 128 + j) * 4, 0, __builtin_bit_cast(uint32_t, e));
    d += e;
  }
  float inv_d = 1.0f / d;

  // Phase 5: Compute final O_ik = sum_j (A_ij * V_jk)
  __global float* Oi = a->O + tid * 64;
  for (uint32_t k_outer = 0; k_outer < 64; k_outer += 4) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint32_t j = 0; j < 128; j++) {
      float a_val = __builtin_bit_cast(float, load32_shared(smem_S_base + (tid * 128 + j) * 4));
      float v0 = __builtin_bit_cast(float, load32_shared(smem_V_base + (j * 64 + k_outer + 0) * 4));
      float v1 = __builtin_bit_cast(float, load32_shared(smem_V_base + (j * 64 + k_outer + 1) * 4));
      float v2 = __builtin_bit_cast(float, load32_shared(smem_V_base + (j * 64 + k_outer + 2) * 4));
      float v3 = __builtin_bit_cast(float, load32_shared(smem_V_base + (j * 64 + k_outer + 3) * 4));
      acc0 += a_val * v0;
      acc1 += a_val * v1;
      acc2 += a_val * v2;
      acc3 += a_val * v3;
    }
    Oi[k_outer + 0] = acc0 * inv_d;
    Oi[k_outer + 1] = acc1 * inv_d;
    Oi[k_outer + 2] = acc2 * inv_d;
    Oi[k_outer + 3] = acc3 * inv_d;
  }
}
