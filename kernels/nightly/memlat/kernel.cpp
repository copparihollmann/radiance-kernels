// memlat device side: a dependent-load latency probe plus two background bandwidth generators.
// See memlat.h for the method, the geometry and the correctness gate.
//
// Role assignment inside a cluster's threadblock (mu_schedule numbers threadblock warp w onto
// core w % MU_NUM_CORES, so warps 0 and 1 are one warp on each core):
//
//   warp 0        the latency probe.  Lane 0 alone runs the chase, so exactly one lane issues
//                 each load and exactly one request is outstanding at a time.
//   warp 1        the gemmini DMA command issuer (lane 0 only: RS1/RS2 are one register pair per
//                 cluster, so only one thread per cluster may issue commands).
//   warps 2..     the muon background generator, one disjoint slice of the source region each.
//
// A warp whose generator is not active at this level does nothing and waits at the level's
// closing barrier, so level 0 really is idle: no spin loop is running beside the probe.
#include <nightly/device.h>
#include <nightly/mx.h>
#include "memlat.h"

#define MEMLAT_BAR_LEVEL_OPEN  1u
#define MEMLAT_BAR_LEVEL_CLOSE 2u

static_assert(MEMLAT_LOADS % MEMLAT_CHASE_UNROLL == 0, "chase unroll must divide the lap");
static_assert((MEMLAT_BG_SLICE_WORDS & MEMLAT_BG_SLICE_MASK) == 0,
              "the background slice wraps with a mask, so it must be a power of two");
static_assert(MEMLAT_SPAD_ROW0 + MEMLAT_SPAD_SLOTS * MEMLAT_DMA_ROWS <= mx::SPAD_ROWS,
              "the DMA landing zone must fit in the scratchpad");
static_assert(MEMLAT_SPAD_ROW0 * 16u > MEMLAT_SM_FLAGS + 64u,
              "the DMA landing zone must not overlap the kernel's own shared words");
static_assert(MU_NUM_CORES * NIGHTLY_OCC <= MEMLAT_BG_SLICES,
              "every background warp needs its own disjoint slice of the source region");

struct MemlatChase {
  uint32_t cycles;     // mcycle delta across the whole lap
  uint32_t terminal;   // node index the lap ended on
  uint32_t sum;        // sum of every visited node index (wrapping)
  uint32_t hash;       // order-sensitive rolling hash of the visit sequence
};

// One lap of the chase.  `base` is the first word of node 0; node i is at base + i * STRIDE_WORDS.
// The load is volatile and the value is laundered through an empty asm, so the compiler can
// neither drop a load, merge two, nor start the next one before this one has returned: the
// address of step k+1 is literally the data of step k.
__attribute__((noinline))
static MemlatChase memlat_chase(volatile const uint32_t *base) {
  uint32_t idx = 0u, sum = 0u, hash = 0u;
  const uint32_t t0 = nightly_mcycle();
  for (uint32_t k = 0; k < MEMLAT_LOADS; k += MEMLAT_CHASE_UNROLL) {
#pragma unroll
    for (uint32_t u = 0; u < MEMLAT_CHASE_UNROLL; u++) {
      idx = base[idx * MEMLAT_STRIDE_WORDS];
      asm volatile("" : "+r"(idx));
      sum += idx;
      hash = hash * MEMLAT_HASH_MUL + idx;
    }
  }
  const uint32_t t1 = nightly_mcycle();
  MemlatChase r;
  r.cycles = t1 - t0;
  r.terminal = idx;
  r.sum = sum;
  r.hash = hash;
  return r;
}

// Background generator 1: Muon L1/L0d load path.  Fully coalesced 16-lane loads walking this
// warp's own slice of the source region.  Every byte counted here is a byte this warp requested
// from the memory system; the slices are disjoint so no two warps share a line.
static uint32_t memlat_muon_bg(volatile __shared uint32_t *stop, uint32_t lane, uint32_t slice) {
  volatile const uint32_t *src =
      (volatile const uint32_t *)MEMLAT_MUON_ADDR + slice * MEMLAT_BG_SLICE_WORDS;
  uint32_t off = lane, iters = 0u, acc = 0u;
  while (*stop == 0u) {
#pragma unroll
    for (uint32_t u = 0; u < MEMLAT_BG_UNROLL; u++)
      acc += src[(off + u * 16u) & MEMLAT_BG_SLICE_MASK];
    off = (off + MEMLAT_BG_UNROLL * 16u) & MEMLAT_BG_SLICE_MASK;
    iters++;
  }
  asm volatile("" :: "r"(acc));   // keep every load live
  return iters * MEMLAT_BG_ITER_BYTES;
}

// Bounded poll of gemmini's READY register.  Returns false when the command queue has not freed
// up within the bound, which stops the generator instead of letting the blocking INST store wedge
// the core -- a timing kernel must not be able to hang a card.
static bool memlat_mx_ready(void) {
  for (uint32_t i = 0; i < MEMLAT_MX_READY_SPIN; i++)
    if (mx::ld32(mx::READY) != 0u) return true;
  return false;
}

// Background generator 2: the gemmini DMA path.  Raw MVIN commands (DRAM -> scratchpad), which is
// the same datapath the Gemmini kernels' operand loads use and the one the roofline credits with
// the high byte rate.  Explicit MVIN is used rather than a loop_ws with skipped compute because
// the loop FSM's skipped loads leave phantom outstanding completions (see
// kernels/fa_mx_hostsf/flash_attention_mx.cpp).
static uint32_t memlat_dma_bg(volatile __shared uint32_t *stop,
                              volatile __shared uint32_t *flags) {
  uint32_t slot = 0u, cmds = 0u;
  while (*stop == 0u) {
    if (!memlat_mx_ready()) { *flags = *flags | MEMLAT_F_DMA_READY; break; }
    const uint64_t src =
        mx::host_addr(MEMLAT_DMA_ADDR + (slot % MEMLAT_DMA_CHUNKS) * MEMLAT_DMA_CHUNK_BYTES);
    const uint32_t spad = MEMLAT_SPAD_ROW0 + (slot % MEMLAT_SPAD_SLOTS) * MEMLAT_DMA_ROWS;
    mx::cmd(mx::MVIN, src,
            ((uint64_t)MEMLAT_DMA_ROWS << 48) | ((uint64_t)mx::DIM << 32) | (uint64_t)spad);
    slot++;
    cmds++;
  }
  uint32_t i = 0;
  while (mx::ld32(mx::BUSY) != 0u) {
    if (++i >= MEMLAT_MX_DRAIN_SPIN) { *flags = *flags | MEMLAT_F_DMA_DRAIN; break; }
  }
  return cmds * MEMLAT_DMA_CHUNK_BYTES;
}

static void entry(void *, uint32_t tid, uint32_t tpb, uint32_t) {
  const uint32_t lane = tid % MU_NUM_THREADS;
  const uint32_t warp = tid / MU_NUM_THREADS;
  const uint32_t warps = tpb / MU_NUM_THREADS;
  volatile __shared uint32_t *stop = (volatile __shared uint32_t *)MEMLAT_SM_STOP;
  volatile __shared uint32_t *muon_bytes = (volatile __shared uint32_t *)MEMLAT_SM_MUON;
  volatile __shared uint32_t *dma_bytes = (volatile __shared uint32_t *)MEMLAT_SM_DMA;
  // The DMA issuer is a different warp from the probe, so its flags travel through shared memory
  // rather than through a register the publishing thread cannot see.
  volatile __shared uint32_t *dma_flags = (volatile __shared uint32_t *)MEMLAT_SM_FLAGS;

  volatile const uint32_t *seq = (volatile const uint32_t *)MEMLAT_SEQ_ADDR;
  volatile const uint32_t *rnd = (volatile const uint32_t *)MEMLAT_RAND_ADDR;

  uint32_t flags = 0u;
  MemlatChase first_seq = {0u, 0u, 0u, 0u}, first_rand = {0u, 0u, 0u, 0u};

  // One-time gemmini configuration, by the single command-issuing thread of this cluster.
  if (tid == MU_NUM_THREADS) {
    mx::flush_tlb();
    mx::lut_disable();
    mx::config_ex(mx::FP8, mx::FP8, mx::FP8);
    mx::config_ld(16u, 0u);        // one 16 B spad row per 16 B of contiguous DRAM
  } else {
    asm volatile("nop");
  }

  for (uint32_t level = 0; level < MEMLAT_LEVELS; level++) {
    if (tid == 0u) {
      *stop = 0u;
      for (uint32_t w = 0; w < warps; w++) muon_bytes[w] = 0u;
      *dma_bytes = 0u;
      if (level == 0u) *dma_flags = 0u;
    } else {
      asm volatile("nop");
    }
    mu_fence_smem();
    mu_barrier(MEMLAT_BAR_LEVEL_OPEN, warps);

    uint32_t t_open = 0u;
    if (warp == 0u) {
      // The probe.  Only lane 0 chases, so one lane issues each dependent load.
      t_open = nightly_mcycle();
      if (lane == 0u) {
        const MemlatChase s = memlat_chase(seq);
        const MemlatChase r = memlat_chase(rnd);
        if (level == 0u) {
          first_seq = s;
          first_rand = r;
        } else if (s.terminal != first_seq.terminal || s.sum != first_seq.sum ||
                   s.hash != first_seq.hash || r.terminal != first_rand.terminal ||
                   r.sum != first_rand.sum || r.hash != first_rand.hash) {
          flags |= MEMLAT_F_CHASE_DIFF;
        }
        nightly_pb_put(MEMLAT_PB_LEVEL(level) + MEMLAT_PB_SEQ_CYCLES, s.cycles);
        nightly_pb_put(MEMLAT_PB_LEVEL(level) + MEMLAT_PB_RAND_CYCLES, r.cycles);
        *stop = 1u;
      } else {
        asm volatile("nop");
      }
      mu_fence_smem();
    } else if (warp == 1u && MEMLAT_LEVEL_DMA(level)) {
      if (lane == 0u) {
        *dma_bytes = memlat_dma_bg(stop, dma_flags);
      } else {
        asm volatile("nop");
      }
      mu_fence_smem();
    } else if (warp >= 2u && MEMLAT_LEVEL_MUON(level)) {
      const uint32_t bytes = memlat_muon_bg(stop, lane, warp % MEMLAT_BG_SLICES);
      if (lane == 0u) muon_bytes[warp] = bytes;
      else asm volatile("nop");
      mu_fence_smem();
    } else {
      asm volatile("nop");
    }
    mu_barrier(MEMLAT_BAR_LEVEL_CLOSE, warps);

    if (tid == 0u) {
      const uint32_t t_close = nightly_mcycle();
      mu_fence_smem();
      uint32_t total = 0u;
      for (uint32_t w = 0; w < warps; w++) total += muon_bytes[w];
      nightly_pb_put(MEMLAT_PB_LEVEL(level) + MEMLAT_PB_MUON_BYTES, total);
      nightly_pb_put(MEMLAT_PB_LEVEL(level) + MEMLAT_PB_DMA_BYTES, *dma_bytes);
      nightly_pb_put(MEMLAT_PB_LEVEL(level) + MEMLAT_PB_LEVEL_CYCLES, t_close - t_open);
      nightly_heartbeat(0x1EA70000u | level);
    } else {
      asm volatile("nop");
    }
  }

  if (tid == 0u) {
    nightly_pb_put(MEMLAT_PB_TERM_SEQ, first_seq.terminal);
    nightly_pb_put(MEMLAT_PB_SUM_SEQ, first_seq.sum);
    nightly_pb_put(MEMLAT_PB_HASH_SEQ, first_seq.hash);
    nightly_pb_put(MEMLAT_PB_TERM_RAND, first_rand.terminal);
    nightly_pb_put(MEMLAT_PB_SUM_RAND, first_rand.sum);
    nightly_pb_put(MEMLAT_PB_HASH_RAND, first_rand.hash);
    const uint32_t all_flags = (flags | *dma_flags) & 0xFFFFu;
    nightly_pb_put(MEMLAT_PB_STATUS, MEMLAT_STATUS_MAGIC | all_flags);
  } else {
    asm volatile("nop");
  }

  nightly_kernel_end(tid, tpb);
}

int main() { return nightly_main(entry, nullptr); }
