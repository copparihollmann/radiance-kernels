/*
 * nightly/device.h: Muon-side launch, timing and identity helpers for the nightly RTL
 * (see lib/include/nightly/README.md for the RTL versions).
 *
 * Launch protocol, shared with nightly/host.h:
 *   1. every core's boot warp waits in nightly_wait_launch() until the host writes
 *      NIGHTLY_LAUNCH_MAGIC to its cluster's printBuf LAUNCH slot;
 *   2. the kernel runs under mu_schedule(); its entry ends with nightly_kernel_end(), which
 *      synchronises the threadblock, writes the core's L0d back with `fence`, and posts the
 *      core's end mcycle to printBuf;
 *   3. the host waits for every core's END slot, then reads the results from GPU DRAM.
 * Posting END after the fence is what makes the host read safe: the per-core `finished` signal
 * has no memory-drain term, so all_finished alone does not prove the stores left the L0d.
 */
#ifndef NIGHTLY_DEVICE_H
#define NIGHTLY_DEVICE_H

#include <stdint.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <nightly/pb.h>

/* Some long, write-heavy kernels can reach END before dirty global-memory
 * lines have become host-visible.  NIGHTLY_ROBUST_GMEM_END is deliberately
 * opt-in while it is being qualified across the FPGA configurations: it adds
 * no useful work and does not alter the workload partition, but it does make
 * completion mean that the per-core L0d and shared L1 have been evicted by
 * capacity. */
#ifndef NIGHTLY_ROBUST_GMEM_END
#define NIGHTLY_ROBUST_GMEM_END 0
#endif
#ifndef NIGHTLY_ROBUST_SPIN_UNTIL_RESET
#define NIGHTLY_ROBUST_SPIN_UNTIL_RESET 0
#endif
#ifndef NIGHTLY_ROBUST_SERIAL_FENCE
#define NIGHTLY_ROBUST_SERIAL_FENCE 0
#endif
#ifndef NIGHTLY_ROBUST_TOKEN_FENCE
#define NIGHTLY_ROBUST_TOKEN_FENCE 0
#endif
#ifndef NIGHTLY_ROBUST_CLUSTER_L1_MANAGER
#define NIGHTLY_ROBUST_CLUSTER_L1_MANAGER 0
#endif
#ifndef NIGHTLY_ROBUST_SPLIT_L1_DRAIN
#define NIGHTLY_ROBUST_SPLIT_L1_DRAIN 0
#endif
#if NIGHTLY_ROBUST_SPIN_UNTIL_RESET && !NIGHTLY_ROBUST_GMEM_END
#error "NIGHTLY_ROBUST_SPIN_UNTIL_RESET requires NIGHTLY_ROBUST_GMEM_END"
#endif
#if NIGHTLY_ROBUST_SERIAL_FENCE && !NIGHTLY_ROBUST_GMEM_END
#error "NIGHTLY_ROBUST_SERIAL_FENCE requires NIGHTLY_ROBUST_GMEM_END"
#endif
#if NIGHTLY_ROBUST_TOKEN_FENCE && !NIGHTLY_ROBUST_SERIAL_FENCE
#error "NIGHTLY_ROBUST_TOKEN_FENCE requires NIGHTLY_ROBUST_SERIAL_FENCE"
#endif
#if NIGHTLY_ROBUST_CLUSTER_L1_MANAGER && !NIGHTLY_ROBUST_GMEM_END
#error "NIGHTLY_ROBUST_CLUSTER_L1_MANAGER requires NIGHTLY_ROBUST_GMEM_END"
#endif
#if NIGHTLY_ROBUST_SPLIT_L1_DRAIN && !NIGHTLY_ROBUST_GMEM_END
#error "NIGHTLY_ROBUST_SPLIT_L1_DRAIN requires NIGHTLY_ROBUST_GMEM_END"
#endif
#if NIGHTLY_ROBUST_SPLIT_L1_DRAIN && NIGHTLY_ROBUST_CLUSTER_L1_MANAGER
#error "select only one shared-L1 drain candidate"
#endif
#if NIGHTLY_ROBUST_GMEM_END
#include <rad_tapeout.h>
#ifndef NIGHTLY_ROBUST_POST_SPIN
#define NIGHTLY_ROBUST_POST_SPIN RAD_EPI_POST_SPIN
#endif
#endif

/* Number of clusters (SMs) the kernel uses on RadianceHBMConfig (2 clusters): 1 = cluster 0 only
 * (the host does not launch cluster 1), 2 = both.  Set by the kernel Makefile per ELF. */
#ifndef NIGHTLY_CLUSTERS
#define NIGHTLY_CLUSTERS 1
#endif

/* Warps per core the kernel runs (set by nightly.mk from OCC, which also caps the compiler's
 * allocatable GPRs to 256/OCC).  The boot code spawns exactly this many warps, so no idle warp
 * holds physical registers. */
#ifndef NIGHTLY_OCC
#define NIGHTLY_OCC 8
#endif
extern "C" uint32_t __mu_num_warps = NIGHTLY_OCC;
/* mu_start.S: take the cluster index for the warp stacks from printBuf (see mu_cluster_id) */
extern "C" uint32_t __mu_cluster_from_pb = 1;

/* Barrier ids: mu_schedule uses 0, nightly_kernel_end uses 15, and kernels may
 * use 1..14.  Robust completion reuses 14 after the kernel body has converged.
 * Callers enabling NIGHTLY_ROBUST_CLUSTER_L1_MANAGER must additionally reserve
 * 13 from their kernel body so its repeated epilogue rendezvous cannot collide. */
#define NIGHTLY_BAR_END 15u
#define NIGHTLY_BAR_ROBUST_DRAIN 13u

static inline uint32_t nightly_mcycle(void) {
  uint32_t c;
  asm volatile("csrr %0, mcycle" : "=r"(c) :: "memory");
  return c;
}

static inline void nightly_pb_put(uint32_t slot, uint32_t v) {
  const uint32_t a = NIGHTLY_PB_DEV_BASE + slot * 8u;
  asm volatile("sw.shared %0, 0(%1)" :: "r"(v), "r"(a) : "memory");
  asm volatile("sw.shared %0, 4(%1)" :: "r"(v), "r"(a) : "memory");
}

static inline uint32_t nightly_pb_get(uint32_t slot) {
  const uint32_t a = NIGHTLY_PB_DEV_BASE + slot * 8u;
  uint32_t v;
  asm volatile("lw.shared %0, 0(%1)" : "=r"(v) : "r"(a) : "memory");
  return v;
}

/* Cluster index.  The cluster-ID CSR is only 1 bit wide in the RTL (radiance subsystem/Configs.scala
 * builds every core with numClusters = 2), so clusters 2 and 3 read 0 and 1 and redo clusters 0 and
 * 1's work.  nightly_host_launch writes each cluster's index to its own printBuf slot
 * NIGHTLY_PB_CLUSTER before LAUNCH; without the tag (another host) fall back to the CSR.
 * This strong definition overrides mu_schedule.cpp's weak one, so threadblock_id is right too
 * (every nightly kernel is one device translation unit, so the definition is emitted once). */
extern "C" __attribute__((noinline)) uint32_t mu_cluster_id(void) {
  const uint32_t v = nightly_pb_get(NIGHTLY_PB_CLUSTER);
  if ((v & 0xFFFF0000u) == NIGHTLY_CLUSTER_TAG) return v & 0xFFFFu;
  uint32_t c;
  asm volatile("csrr %0, %1" : "=r"(c) : "i"(MU_CSR_CLUSTER_ID));
  return c;
}

static inline uint32_t nightly_cluster_id(void) { return mu_cluster_id(); }

/* Called by the boot warp (one thread) of every core, before mu_schedule(). */
static inline void nightly_wait_launch(void) {
  while (nightly_pb_get(NIGHTLY_PB_LAUNCH) != NIGHTLY_LAUNCH_MAGIC) {
    asm volatile("nop");
  }
  nightly_pb_put(NIGHTLY_PB_START + (uint32_t)vx_core_id(), nightly_mcycle());
}

/* Kernel-defined phase stamp (cluster core 0, one thread). */
static inline void nightly_phase(uint32_t idx) {
  nightly_pb_put(NIGHTLY_PB_PHASE + (idx & 7u), nightly_mcycle());
}

/* Progress value for the host's wait loop (one thread; e.g. (tile << 16) | block).  RTL simulation
 * shows no output until the host finishes, so this is how a hang becomes visible while it runs. */
static inline void nightly_heartbeat(uint32_t v) { nightly_pb_put(NIGHTLY_PB_HEARTBEAT, v); }

#if NIGHTLY_ROBUST_GMEM_END
/* rad_tapeout.h targets a one-cluster part and keys its scratch only by
 * vx_core_id().  Nightly Radiance uses up to four clusters, so give every
 * (cluster, core) pair disjoint lines; otherwise different SMs manufacture
 * cache-line contention in an epilogue whose only job is visibility. */
static inline void nightly_l0d_drain(void) {
  const uint32_t cluster_base = RAD_DRAIN_SCRATCH +
      nightly_cluster_id() * MU_NUM_CORES * RAD_L0D_BYTES;
  rad_l0d_drain_at(cluster_base, RAD_L0D_LINES);
}

static inline void nightly_l1_drain(void) {
#if NIGHTLY_ROBUST_SPLIT_L1_DRAIN
  static_assert(RAD_L1_DRAIN_MULT % MU_NUM_CORES == 0,
                "L1 drain multiplicity must divide across cores");
  constexpr uint32_t per_core_mult = RAD_L1_DRAIN_MULT / MU_NUM_CORES;
  const uint32_t cluster_base = nightly_cluster_id() *
      (RAD_L1_DRAIN_MULT * RAD_L1_BYTES);
  const uint32_t core_base = cluster_base +
      (uint32_t)vx_core_id() * (per_core_mult * RAD_L1_BYTES);
  volatile uint32_t *p = (volatile uint32_t *)(
      RAD_DRAIN_SCRATCH + 0x00100000u + core_base);
  const uint32_t n = (per_core_mult * RAD_L1_BYTES) / RAD_L1_LINE_BYTES;
#else
  const uint32_t global_core =
      nightly_cluster_id() * MU_NUM_CORES + (uint32_t)vx_core_id();
  volatile uint32_t *p = (volatile uint32_t *)(
      RAD_DRAIN_SCRATCH + 0x00100000u +
      global_core * (RAD_L1_DRAIN_MULT * RAD_L1_BYTES));
  const uint32_t n = (RAD_L1_DRAIN_MULT * RAD_L1_BYTES) / RAD_L1_LINE_BYTES;
#endif
  for (uint32_t i = 0; i < n; ++i)
    p[i * (RAD_L1_LINE_BYTES / 4u)] = i + 1u;
  mu_fence();
}
#endif

/* End of a kernel entry: every thread of the threadblock must call this exactly once.
 * noinline + convergent: inlined, clang tail-duplicated its barrier into both arms of a preceding
 * `if (lead) ... else nop` (FA v2: `if (lead) mx::fence()`), so warp 0 ran barrier 15 twice and
 * hung.  A convergent call is not duplicated into divergent paths. */
__attribute__((noinline, convergent))
static void nightly_kernel_end(uint32_t tid_in_threadblock, uint32_t threads_per_threadblock) {
  const uint32_t warps = threads_per_threadblock / MU_NUM_THREADS;
  mu_barrier(NIGHTLY_BAR_END, warps);
#if NIGHTLY_ROBUST_GMEM_END
  const uint32_t warp = tid_in_threadblock / MU_NUM_THREADS;
  const uint32_t lane = tid_in_threadblock % MU_NUM_THREADS;
#if NIGHTLY_ROBUST_SERIAL_FENCE
  /* A barrier only proves that every warp issued its stores; mu_fence() drains
   * only the issuing warp's queue.  Fencing all warps simultaneously is not
   * safe on the taped-out SFU path, which can silently drop overlapping
   * requests.  Hand the uncached printBuf epoch from one warp to the next so
   * exactly one lane issues a fence at a time.  The rest of each warp remains
   * on the divergent path until lane 0 observes the completed epoch, so the
   * warp reconverges before advancing.  All of this remains inside the timed
   * kernel interval and does not add or remove useful workload transfers. */
  if (tid_in_threadblock == 0u) {
    nightly_pb_put(NIGHTLY_PB_FENCE_EPOCH, 0u);
  } else {
    asm volatile("nop");
  }
  mu_barrier(NIGHTLY_BAR_END - 1u, warps);
#if NIGHTLY_ROBUST_TOKEN_FENCE
  /* Pass one token through the warp lane-0s.  This has the same single-fence
   * ordering as the epoch-round implementation below, but each warp waits on
   * only its predecessor instead of polling every epoch.  The trailing
   * barrier is required: no cache-capacity drain may start until the final
   * worker fence has completed. */
  if (lane == 0u) {
    while (nightly_pb_get(NIGHTLY_PB_FENCE_EPOCH) < warp) {
      asm volatile("nop");
    }
    mu_fence();
    nightly_pb_put(NIGHTLY_PB_FENCE_EPOCH, warp + 1u);
  } else {
    asm volatile("nop");
  }
  mu_barrier(NIGHTLY_BAR_END - 1u, warps);
#else
  for (uint32_t round = 0; round < warps; ++round) {
    if (warp == round && lane == 0u) {
      mu_fence();
      nightly_pb_put(NIGHTLY_PB_FENCE_EPOCH, round + 1u);
    } else if (lane == 0u) {
      while (nightly_pb_get(NIGHTLY_PB_FENCE_EPOCH) < round + 1u) {
        asm volatile("nop");
      }
    } else {
      asm volatile("nop");
    }
  }
#endif
#endif
  /* warp_id_rr = warp_in_core*ncores + core, so the first MU_NUM_CORES
   * threadblock warps select one executor on each core.  The settle is needed
   * because the arrival barrier synchronizes warps but does not drain their
   * independent store queues.  Keep the divergent guard explicit: LLVM has
   * previously cloned a following barrier into both arms without the else. */
#if NIGHTLY_ROBUST_CLUSTER_L1_MANAGER
  if (warp < MU_NUM_CORES && lane == 0u) {
    rad_pause(RAD_EPI_QUEUE_SPIN);
    mu_fence();
    nightly_l0d_drain();
  } else {
    asm volatile("nop");
  }
  /* Candidate completion optimization: L0d is private to each core, so one
   * lane on every core still performs every required L0d capacity walk.  L1
   * is shared by the cluster, so only warp 0 / lane 0 capacity-drains it.
   * Both barriers are encountered uniformly by every warp.  The first makes
   * all private L0d drains complete before the manager starts L1; the second
   * prevents the next pass (or END) from racing the shared drain.  Barrier 13
   * is reusable after each completed rendezvous and does not collide with the
   * scheduler (0) or END/fence barrier 14/15.  Opt-in callers reserve 13 from
   * their kernel body for this protocol. */
  mu_barrier(NIGHTLY_BAR_ROBUST_DRAIN, warps);
  if (warp == 0u && lane == 0u) {
    nightly_l1_drain();
  } else {
    asm volatile("nop");
  }
  mu_barrier(NIGHTLY_BAR_ROBUST_DRAIN, warps);

  if (warp < MU_NUM_CORES && lane == 0u) {
    rad_pause(RAD_EPI_SETTLE);
    nightly_l0d_drain();
  } else {
    asm volatile("nop");
  }
  mu_barrier(NIGHTLY_BAR_ROBUST_DRAIN, warps);
  if (warp == 0u && lane == 0u) {
    nightly_l1_drain();
  } else {
    asm volatile("nop");
  }
  mu_barrier(NIGHTLY_BAR_ROBUST_DRAIN, warps);

  /* Keep the third pass that full-size xRAGE required in the default path. */
  if (warp < MU_NUM_CORES && lane == 0u) {
    rad_pause(RAD_EPI_SETTLE);
    nightly_l0d_drain();
  } else {
    asm volatile("nop");
  }
  mu_barrier(NIGHTLY_BAR_ROBUST_DRAIN, warps);
  if (warp == 0u && lane == 0u) {
    nightly_l1_drain();
  } else {
    asm volatile("nop");
  }
  mu_barrier(NIGHTLY_BAR_ROBUST_DRAIN, warps);
  if (warp < MU_NUM_CORES && lane == 0u) {
    rad_pause(NIGHTLY_ROBUST_POST_SPIN);
  } else {
    asm volatile("nop");
  }
#else
  if (warp < MU_NUM_CORES && lane == 0u) {
    rad_pause(RAD_EPI_QUEUE_SPIN);
    mu_fence();
    nightly_l0d_drain();
    nightly_l1_drain();
    rad_pause(RAD_EPI_SETTLE);
    nightly_l0d_drain();
    nightly_l1_drain();
    /* A full-size xRAGE Gather exposed one absent, aligned 32-byte L1 line
     * after two passes even though every input and guard was intact.  A third
     * capacity walk is intentionally part of the measured epilogue: it
     * re-evicts every set and does not add or repair any workload output.
     * Keeping this device-side also prevents a host delay from hiding
     * completion latency or resetting the GPU before the last writeback. */
    rad_pause(RAD_EPI_SETTLE);
    nightly_l0d_drain();
    nightly_l1_drain();
    rad_pause(NIGHTLY_ROBUST_POST_SPIN);
  } else {
    asm volatile("nop");
  }
#endif
  mu_barrier(NIGHTLY_BAR_END - 1u, warps);
  if (warp < MU_NUM_CORES && lane == 0u) {
    nightly_pb_put(NIGHTLY_PB_END + (uint32_t)vx_core_id(), nightly_mcycle());
  } else {
    asm volatile("nop");
  }
  /* Optionally do not return from a robust FPGA kernel.  Returning lets every pcTracker entry
   * become invalid and raises the tile's `finished` edge, which invokes the
   * defective taped-out L0d flush while the Rocket host starts readback.  The
   * host observes every END stamp, asserts GPU soft reset, and only then reads
   * GMEM.  This register-only loop keeps `finished` low until that reset.
   * Cyclotron does not model the soft-reset MMIO, so its correctness build
   * uses the same drain with this FPGA-only handshake disabled. */
#if NIGHTLY_ROBUST_SPIN_UNTIL_RESET
  for (;;) asm volatile("" ::: "memory");
#endif
#else
  /* threadblock warp w runs on core (w % MU_NUM_CORES): warps 0..MU_NUM_CORES-1 cover each core once */
  if (tid_in_threadblock < MU_NUM_CORES * MU_NUM_THREADS) {
    mu_fence();
    if ((tid_in_threadblock % MU_NUM_THREADS) == 0)
      nightly_pb_put(NIGHTLY_PB_END + (uint32_t)vx_core_id(), nightly_mcycle());
  } else {
    asm volatile("nop");
  }
#endif
}

/* Standard GPU main(): wait for the host, run `entry` at NIGHTLY_OCC warps per core. */
static inline int nightly_main(mu_schedule_callback entry, void *arg) {
  nightly_wait_launch();
  mu_schedule(entry, arg, NIGHTLY_OCC);
  return 0;
}

#endif
