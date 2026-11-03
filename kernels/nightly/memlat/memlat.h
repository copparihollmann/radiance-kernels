/*
 * memlat: a dependent-load latency microbenchmark for the nightly RTL.
 *
 * WHY IT EXISTS.  Every latency number the FPGA-1A campaign reports today is a Little's-Law mean,
 * L = outstanding * cycles / reads, derived from the same cycle count it is then used to explain.
 * That is circular, and it has no tail: FASED's latency_histogram.csv is written unconditionally
 * but its histograms are not elaborated, so all 42 jobs contain 1025 bare newlines.  memlat
 * measures latency directly instead: one warp of one core chases a pointer through DRAM, with
 * exactly one request in flight, while a controlled amount of background traffic runs beside it.
 *
 * WHAT IT MEASURES.
 *   cyc/load   cycles per DEPENDENT load, measured as (mcycle delta) / loads over a chase whose
 *              next address is the value just loaded.  The dependence is what defeats both
 *              prefetching (the address does not exist until the load returns) and
 *              memory-level parallelism (the core cannot have two of these in flight).
 *   seq        the chase walks nodes in ascending address order: predictable, DRAM-row friendly.
 *   rand       the chase walks a single random cycle over the same nodes: no locality.
 *              seq and rand run the SAME instruction sequence over the SAME geometry; only the
 *              next-pointer values differ, so the difference between them is memory behaviour and
 *              not code.
 *   levels     0 idle, 1 muon, 2 dma, 3 muon+dma.  Each level reports the bandwidth the
 *              background generator actually achieved, so the latency figure can be placed on a
 *              latency-bandwidth curve rather than quoted as a single number.
 *
 * GEOMETRY.  MEMLAT_NODES nodes, one per MEMLAT_STRIDE_BYTES, so every chase step touches a fresh
 * 64 B L0d line and a fresh 32 B L1 line.  The chase working set (MEMLAT_NODES * stride) is sized
 * well above the 4 KiB per-core L0d and the 32 KiB cluster L1 (rad_tapeout.h) so a step cannot be
 * served by either.
 *
 * WHAT IT DOES NOT MEASURE, so no one reads more out of it than is there.
 *   * Tail latency.  Each figure is the mean over one lap of MEMLAT_LOADS dependent loads.  The
 *     RTL has no latency histogram to read (FASED writes latency_histogram.csv unconditionally
 *     but does not elaborate the histograms), so a distribution is not available from any source.
 *   * Per-request latency.  Only the lap total is timed; individual steps are not stamped.
 *   * Cross-cluster isolation.  Every cluster runs the same four levels from one launch, with no
 *     barrier between clusters, so cluster 1's background traffic overlaps cluster 0's levels and
 *     the two timelines drift apart as the run proceeds.  The drift is reconstructible after the
 *     fact: each cluster's level boundaries follow from its core-0 START stamp plus the
 *     cumulative level_cycles it reports, so an analysis can check how far the levels actually
 *     overlapped instead of assuming they lined up.  A device-side cross-cluster barrier was
 *     deliberately NOT added: it would have to spin on GPU DRAM, and an untested spin is exactly
 *     the way to wedge a card.
 *   * The one-lane probe is a single-lane load, not a warp-wide one.  It measures the latency the
 *     memory system shows one request, which is what a latency number should be; it is not the
 *     latency a fully coalesced 16-lane access sees.
 *
 * Shared by kernel.cpp (Muon) and host.cpp (Rocket): keep it free of anything either side cannot
 * compile, and integer-only (the host Rocket has no FPU).
 */
#ifndef MEMLAT_H
#define MEMLAT_H

#include <stdint.h>
#include <nightly/pb.h>

/* ---- chase geometry --------------------------------------------------------------------- */
/* One node per L0d line.  RAD_L0D_LINE_BYTES is 64 and the cluster L1 line is 32, so a step
 * never shares a line with the step before or after it in either cache. */
#define MEMLAT_STRIDE_BYTES 64u
#define MEMLAT_STRIDE_WORDS (MEMLAT_STRIDE_BYTES / 4u)
/* 4096 nodes * 64 B = 256 KiB per chase buffer: 64x the 4 KiB L0d and 8x the 32 KiB cluster L1. */
#define MEMLAT_NODES 4096u
#define MEMLAT_CHASE_BYTES (MEMLAT_NODES * MEMLAT_STRIDE_BYTES)
/* Exactly one lap of the permutation, which is what makes the terminal value and the visit
 * checksum below verifiable (see "correctness gate"). */
#define MEMLAT_LOADS MEMLAT_NODES
#define MEMLAT_CHASE_UNROLL 4u
/* Order-sensitive visit hash; odd multiplier so the recurrence is a bijection on uint32. */
#define MEMLAT_HASH_MUL 1000003u

/* ---- device memory map (host-written, no ELF blobs) --------------------------------------- */
#define MEMLAT_SEQ_ADDR   0x20000000u              /* .region1 */
#define MEMLAT_RAND_ADDR  (MEMLAT_SEQ_ADDR + 0x00080000u)
#define MEMLAT_MUON_ADDR  0x40000000u              /* .region2: muon background source, 1 MiB */
#define MEMLAT_MUON_BYTES (1u << 20)
#define MEMLAT_DMA_ADDR   0x60000000u              /* .region3: gemmini DMA source, 1 MiB */
#define MEMLAT_DMA_BYTES  (1u << 20)

/* ---- levels ------------------------------------------------------------------------------ */
#define MEMLAT_LEVELS 4u
#define MEMLAT_LEVEL_MUON(l) (((l) & 1u) != 0u)
#define MEMLAT_LEVEL_DMA(l)  (((l) & 2u) != 0u)

/* ---- background generators ---------------------------------------------------------------- */
/* muon: every background warp streams a disjoint 1/8 slice of the muon region with fully
 * coalesced 16-lane loads, so the bytes it reports are bytes it actually requested and no two
 * warps share a line. */
#define MEMLAT_BG_UNROLL 8u
#define MEMLAT_BG_SLICES 8u
#define MEMLAT_BG_SLICE_WORDS ((MEMLAT_MUON_BYTES / 4u) / MEMLAT_BG_SLICES)
#define MEMLAT_BG_SLICE_MASK (MEMLAT_BG_SLICE_WORDS - 1u)
/* bytes one background warp requests per iteration: 16 lanes * UNROLL * 4 B */
#define MEMLAT_BG_ITER_BYTES (16u * MEMLAT_BG_UNROLL * 4u)

/* dma: raw gemmini MVIN, DRAM -> scratchpad.  A spad row is 16 B (mx::SPAD_ROWS 8192 = 128 KiB
 * SMEM), so one command of MEMLAT_DMA_ROWS rows moves MEMLAT_DMA_ROWS * 16 B.  The landing zone
 * is the upper half of SMEM; the kernel's own shared words live below 0x200. */
#define MEMLAT_DMA_ROWS 128u
#define MEMLAT_DMA_CHUNK_BYTES (MEMLAT_DMA_ROWS * 16u)
#define MEMLAT_DMA_CHUNKS (MEMLAT_DMA_BYTES / MEMLAT_DMA_CHUNK_BYTES)
#define MEMLAT_SPAD_ROW0 4096u
#define MEMLAT_SPAD_SLOTS 32u
/* Bounded MMIO waits.  A pure timing kernel must never be able to hang an FPGA run: if gemmini
 * stops accepting commands or never goes idle, the generator stops, records a flag and the level
 * completes with whatever bandwidth it managed. */
#define MEMLAT_MX_READY_SPIN 100000u
#define MEMLAT_MX_DRAIN_SPIN 2000000u

/* ---- shared-memory scratch (byte addresses in the __shared space) -------------------------- */
#define MEMLAT_SM_STOP 0x0000u                     /* background stop flag */
#define MEMLAT_SM_MUON 0x0040u                     /* per-warp muon byte counters, 4 B each */
#define MEMLAT_SM_DMA  0x0100u                     /* dma byte counter */
#define MEMLAT_SM_FLAGS 0x0104u                    /* dma generator flags (see MEMLAT_F_*) */

/* ---- printBuf slots (per cluster) ---------------------------------------------------------- */
/* 21..47 is the range nightly/pb.h reserves for kernel results. */
#define MEMLAT_PB_BASE NIGHTLY_PB_KERNEL_FIRST     /* 21 */
#define MEMLAT_PB_PER_LEVEL 5u
#define MEMLAT_PB_LEVEL(l) (MEMLAT_PB_BASE + (l) * MEMLAT_PB_PER_LEVEL)
#define MEMLAT_PB_SEQ_CYCLES   0u
#define MEMLAT_PB_RAND_CYCLES  1u
#define MEMLAT_PB_MUON_BYTES   2u
#define MEMLAT_PB_DMA_BYTES    3u
#define MEMLAT_PB_LEVEL_CYCLES 4u
/* 41..47: the correctness gate and the run status. */
#define MEMLAT_PB_TERM_SEQ  (MEMLAT_PB_BASE + MEMLAT_LEVELS * MEMLAT_PB_PER_LEVEL + 0u)
#define MEMLAT_PB_SUM_SEQ   (MEMLAT_PB_BASE + MEMLAT_LEVELS * MEMLAT_PB_PER_LEVEL + 1u)
#define MEMLAT_PB_HASH_SEQ   (MEMLAT_PB_BASE + MEMLAT_LEVELS * MEMLAT_PB_PER_LEVEL + 2u)
#define MEMLAT_PB_TERM_RAND (MEMLAT_PB_BASE + MEMLAT_LEVELS * MEMLAT_PB_PER_LEVEL + 3u)
#define MEMLAT_PB_SUM_RAND  (MEMLAT_PB_BASE + MEMLAT_LEVELS * MEMLAT_PB_PER_LEVEL + 4u)
#define MEMLAT_PB_HASH_RAND  (MEMLAT_PB_BASE + MEMLAT_LEVELS * MEMLAT_PB_PER_LEVEL + 5u)
#define MEMLAT_PB_STATUS    (MEMLAT_PB_BASE + MEMLAT_LEVELS * MEMLAT_PB_PER_LEVEL + 6u)

/* status = MEMLAT_STATUS_MAGIC | flags */
#define MEMLAT_STATUS_MAGIC 0xC0DE0000u
#define MEMLAT_STATUS_MASK  0xFFFF0000u
#define MEMLAT_F_CHASE_DIFF 0x0001u   /* a level > 0 chase did not reproduce level 0's traversal */
#define MEMLAT_F_DMA_READY  0x0002u   /* gemmini stopped accepting commands within the spin bound */
#define MEMLAT_F_DMA_DRAIN  0x0004u   /* gemmini was still busy when the drain bound expired */

/* ---- correctness gate ---------------------------------------------------------------------
 *
 * Both next-pointer arrays are a SINGLE cycle of length MEMLAT_NODES over the whole node set, so
 * a chase of exactly MEMLAT_LOADS steps from node 0:
 *
 *   * must end on node 0 again (terminal value), and
 *   * must have visited every node exactly once, which the host checks with the sum of the
 *     visited indices, and
 *   * must have visited them in the ORDER that permutation dictates, which the host checks with
 *     an order-sensitive rolling hash (h = h * MEMLAT_HASH_MUL + index).  The sum and the
 *     terminal alone cannot tell the seq traversal from the rand one -- both are a full lap of a
 *     4096-cycle, so both give terminal 0 and sum N(N-1)/2.  The hash can, so a kernel that
 *     accidentally chased the same buffer twice fails instead of passing.
 * All three golds are computed by the host walking its OWN permutation before launch.
 *
 * The host additionally walks the random permutation it built and refuses to launch unless it
 * really is one N-cycle, so the gold it compares against is verified structure and not an
 * assumption.  Finally the device checks that levels 1..3 reproduce level 0's traversal exactly,
 * which makes all 8 chases of a run part of the gate rather than just the first.  A timing kernel
 * whose loads silently returned garbage would fail every one of these.
 */

#endif
