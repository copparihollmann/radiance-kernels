#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
#include "pipeline_math.hpp"
#include "data"

extern "C" uint32_t __mu_num_warps = 1;

struct Args {
  __global float *x, *gamma, *weight, *bias, *skip;
  __global float *normalized, *projected, *output;
};

alignas(64) __global float normalized_raw[CHAIN_M * CHAIN_K] = {0};
alignas(64) __global float projected_raw[CHAIN_M * CHAIN_N] = {0};
alignas(64) __global float output_raw[CHAIN_M * CHAIN_N] = {0};
// Both cores initialize this struct; keep it off the output's cache lines.
alignas(64) static Args args;

// Muon L0d is private to each core. Keep each stage and its consumer on core 0
// until a cross-core buffer flush/ownership protocol is available.
static bool owned_lane(uint32_t tid, uint32_t tpb,
                       uint32_t* local_tid, uint32_t* local_tpb) {
  if ((tid / MU_NUM_THREADS) % MU_NUM_CORES != 0) return false;
  *local_tid = (tid / (MU_NUM_THREADS * MU_NUM_CORES)) * MU_NUM_THREADS
             + tid % MU_NUM_THREADS;
  *local_tpb = tpb / MU_NUM_CORES;
  return true;
}

static void norm_body(void* raw, uint32_t tid, uint32_t tpb, uint32_t) {
  uint32_t local_tid, local_tpb;
  if (!owned_lane(tid, tpb, &local_tid, &local_tpb)) return;
  auto* a = reinterpret_cast<Args*>(raw);
  model_chain::rmsnorm(a->x, a->gamma, a->normalized,
                       CHAIN_M, CHAIN_K, CHAIN_EPS, local_tid, local_tpb);
}

static void linear_body(void* raw, uint32_t tid, uint32_t tpb, uint32_t) {
  uint32_t local_tid, local_tpb;
  if (!owned_lane(tid, tpb, &local_tid, &local_tpb)) return;
  auto* a = reinterpret_cast<Args*>(raw);
  model_chain::linear(a->normalized, a->weight, a->bias, a->projected,
                      CHAIN_M, CHAIN_K, CHAIN_N, local_tid, local_tpb);
}

static void residual_body(void* raw, uint32_t tid, uint32_t tpb, uint32_t) {
  uint32_t local_tid, local_tpb;
  if (!owned_lane(tid, tpb, &local_tid, &local_tpb)) return;
  auto* a = reinterpret_cast<Args*>(raw);
  model_chain::residual(a->projected, a->skip, a->output,
                        CHAIN_M * CHAIN_N, local_tid, local_tpb);
}

int main() {
  args = {input_raw, gamma_raw, weight_raw, bias_raw, skip_raw,
          normalized_raw, projected_raw, output_raw};
  mu_schedule(norm_body, &args, 1);
  mu_barrier(0, MU_NUM_CORES);
  mu_fence();
  mu_schedule(linear_body, &args, 1);
  mu_barrier(0, MU_NUM_CORES);
  mu_fence();
  mu_schedule(residual_body, &args, 1);
  mu_barrier(0, MU_NUM_CORES);
  mu_fence();
  asm volatile("vx_tmc %0" :: "r"(1) : "memory");
  if (mu_hart_id() != 0) { for (;;) {} }
  uint32_t errors = 0;
  for (uint32_t i = 0; i < CHAIN_M * CHAIN_K; ++i)
    if (!mu_close(normalized_raw[i], gold_norm_raw[i], 2e-4f, 2e-5f)) ++errors;
  if (errors) { mu_tohost((1u << 16) | (errors << 1) | 1u); return 0; }
  for (uint32_t i = 0; i < CHAIN_M * CHAIN_N; ++i)
    if (!mu_close(projected_raw[i], gold_projected_raw[i], 2e-4f, 2e-5f)) ++errors;
  if (errors) { mu_tohost((2u << 16) | (errors << 1) | 1u); return 0; }
  for (uint32_t i = 0; i < CHAIN_M * CHAIN_N; ++i)
    if (!mu_close(output_raw[i], gold_raw[i], 2e-4f, 2e-5f)) ++errors;
  mu_tohost(errors ? ((3u << 16) | (errors << 1) | 1u) : 0u);
  return 0;
}
