// SigLIP vision projection / MLP bias-add epilogue (SmolVLA vision tower).
//   out[t, c] = proj[t, c] + bias[c]      (bias broadcast over token rows)
// Every vision linear (q/k/v/out_proj, fc1, fc2) carries a bias; the text tower
// does not. One token-row per thread, grid-stride. fp32. tohost 0 = pass.

#ifndef NUM_WARPS
#define NUM_WARPS 4
#endif

#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

extern "C" uint32_t __mu_num_warps = NUM_WARPS;

#ifndef TOLERANCE_REL
#define TOLERANCE_REL 1.0e-3f
#endif
#ifndef TOLERANCE_ABS
#define TOLERANCE_ABS 2.0e-4f
#endif

#include "data"

struct KernelArgs { __global float *proj, *bias, *out; uint32_t rows, cols; };

// SUBSTITUTE HERE
static inline void kernel_body(void* raw_arg, uint32_t tid_in_threadblock,
                               uint32_t threads_per_threadblock, uint32_t threadblock_id) {
  (void)threadblock_id;
  auto* a = reinterpret_cast<KernelArgs*>(raw_arg);
  const uint32_t cols = a->cols;
  for (uint32_t row = tid_in_threadblock; row < a->rows; row += threads_per_threadblock) {
    const uint32_t base = row * cols;
    #pragma clang loop unroll(disable)
    for (uint32_t c = 0; c < cols; c++) a->out[base + c] = a->proj[base + c] + a->bias[c];
  }
}
// SUBSTITUTE END

static KernelArgs kernel_args;

static inline float fabsf_(float x) { return x < 0.0f ? -x : x; }
static inline bool close_enough(float c, float g) {
  return fabsf_(c - g) <= TOLERANCE_REL * fabsf_(g) + TOLERANCE_ABS;
}
static inline uint32_t hart_id() {
  uint32_t id; asm volatile("csrr %0, mhartid" : "=r"(id)::"memory"); return id;
}

int main() {
  kernel_args = {proj_raw, bias_raw, out_raw, BA_M, BA_N};
  mu_schedule(kernel_body, &kernel_args, NUM_WARPS);
  mu_barrier(0, MU_NUM_CORES);
  asm volatile("vx_tmc %0" ::"r"(1) : "memory");
  if (hart_id() != 0) { for (;;) {} }
  uint32_t errors = 0;
  for (uint32_t i = 0; i < VERIFY_COUNT; i++)
    if (!close_enough(out_raw[i], gold_raw[i])) errors++;
  uint32_t code = errors ? ((errors << 1) | 1u) : 0u;
  asm volatile(".insn i 0x73, 0, x0, %0, 0" ::"r"(code) : "memory");
  return 0;
}
