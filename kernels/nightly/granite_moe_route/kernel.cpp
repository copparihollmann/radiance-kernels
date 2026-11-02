// Granite 3.1 MoE routing bring-up: exact top-k and stable expert-major scatter.
//
// Scope is intentionally structural.  This kernel consumes FP32 router logits and emits all
// 1,024 token/expert assignments, but does not yet compute selected-softmax weights, expert
// GEMMs or the combined model output.  Every emitted uint32_t is checked exactly after timing.
#include <nightly/device.h>
#include <nightly/verify.h>
#include "granite_moe_route_data.h"

#ifndef GMR_DIAG
#define GMR_DIAG 0
#endif

static_assert(NIGHTLY_CLUSTERS == 1,
              "first Granite route milestone is a one-cluster correctness bring-up");
static_assert(GMR_TOP_K == 8 && GMR_EXPERTS == 32,
              "the fixed-size insertion network assumes Granite 3.1 routing dimensions");

static inline bool route_before(float score, uint32_t expert, float old_score,
                                uint32_t old_expert) {
  return score > old_score || (score == old_score && expert < old_expert);
}

static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t) {
  const float *logits = (const float *)GMR_LOGITS_ADDR;
  volatile uint32_t *output = (volatile uint32_t *)GMR_OUTPUT_ADDR;
  volatile uint32_t *top = output + GMR_TOP_WORD;
  volatile uint32_t *counts = output + GMR_COUNTS_WORD;
  volatile uint32_t *offsets = output + GMR_OFFSETS_WORD;
  volatile uint32_t *packed = output + GMR_PACKED_WORD;
  volatile uint32_t *inverse = output + GMR_INVERSE_WORD;

  // Each token is independent.  Select one rank per pass, excluding experts selected by prior
  // passes.  This is deliberately more explicit than a dynamically indexed insertion network:
  // the Muon SIMT compiler has miscompiled divergent array-shift loops, while this formulation
  // maps directly onto Granite's deterministic ranking rule.
  for (uint32_t token = tid; token < GMR_TOKENS; token += tpb) {
    uint32_t selected[GMR_TOP_K];
#pragma unroll
    for (uint32_t rank = 0; rank < GMR_TOP_K; rank++) selected[rank] = 0xFFFFFFFFu;
#pragma unroll
    for (uint32_t rank = 0; rank < GMR_TOP_K; rank++) {
      float best_score = -3.402823466e+38F;
      uint32_t best_expert = 0xFFFFFFFFu;
      for (uint32_t expert = 0; expert < GMR_EXPERTS; expert++) {
        bool used = false;
#pragma unroll
        for (uint32_t prior = 0; prior < GMR_TOP_K; prior++) {
          if (prior < rank && selected[prior] == expert) used = true;
        }
        const float score = logits[token * GMR_EXPERTS + expert];
        if (!used && route_before(score, expert, best_score, best_expert)) {
          best_score = score;
          best_expert = expert;
        }
      }
      selected[rank] = best_expert;
    }
#pragma unroll
    for (uint32_t rank = 0; rank < GMR_TOP_K; rank++)
      top[token * GMR_TOP_K + rank] = selected[rank];
  }
  mu_fence();
  mu_barrier(1, tpb / MU_NUM_THREADS);

  // One thread per expert builds the histogram.  Offset entries are independent prefix sums;
  // this small 32-expert table favors transparent correctness over a complicated scan.
  for (uint32_t expert = tid; expert < GMR_EXPERTS; expert += tpb) {
    uint32_t count = 0;
    for (uint32_t assignment = 0; assignment < GMR_ASSIGNMENTS; assignment++)
      count += top[assignment] == expert;
    counts[expert] = count;
  }
  mu_fence();
  mu_barrier(2, tpb / MU_NUM_THREADS);
  for (uint32_t boundary = tid; boundary <= GMR_EXPERTS; boundary += tpb) {
    uint32_t offset = 0;
    for (uint32_t expert = 0; expert < boundary; expert++) offset += counts[expert];
    offsets[boundary] = offset;
  }
  mu_fence();
  mu_barrier(3, tpb / MU_NUM_THREADS);

  // Assignments are canonical token-major IDs.  Counting earlier IDs with the same expert
  // computes a stable local rank without atomics, so expert groups preserve token/rank order.
  for (uint32_t assignment = tid; assignment < GMR_ASSIGNMENTS; assignment += tpb) {
    const uint32_t expert = top[assignment];
    uint32_t local_rank = 0;
    for (uint32_t earlier = 0; earlier < assignment; earlier++)
      local_rank += top[earlier] == expert;
    const uint32_t packed_index = offsets[expert] + local_rank;
    packed[packed_index] = assignment;
    inverse[assignment] = packed_index;
  }
#if GMR_DIAG
  mu_fence();
  mu_barrier(4, tpb / MU_NUM_THREADS);
  if (tid == 0) {
    for (uint32_t i = 0; i < 16; i++) nightly_pb_put(32 + i, top[i]);
  } else {
    asm volatile("nop");
  }
#endif
  nightly_kernel_end(tid, tpb);
  nightly_verify_u32_exact(GMR_OUTPUT_ADDR, GMR_GOLD_ADDR, GMR_OUTPUT_WORDS,
                           tid, tpb);
}

int main() { return nightly_main(entry, nullptr); }
