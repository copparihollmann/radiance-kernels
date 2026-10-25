#include <stdint.h>

#include "generated/config.h"
#include <mu_partition.h>
#include <nightly/device.h>

static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "host and device cluster counts must match");
static_assert(PATHFINDER_COLS % PATHFINDER_INTS_PER_LINE == 0u,
              "rows and per-SM scratch must remain cache-line aligned");

// mu_schedule owns 0; robust completion owns 13 and 14; END owns 15.
// Pathfinder exclusively reuses 12 after every uniformly reached phase.
static constexpr uint32_t kPathfinderPhaseBarrier = 12u;
static_assert(kPathfinderPhaseBarrier == PATHFINDER_PHASE_BARRIER_ID,
              "generated barrier metadata and kernel disagree");
static_assert(kPathfinderPhaseBarrier > 0u &&
                  kPathfinderPhaseBarrier < NIGHTLY_BAR_ROBUST_DRAIN,
              "Pathfinder phase barrier collides with runtime barriers");

extern "C" {
extern __global const int32_t pathfinder_wall[];
extern __global int32_t pathfinder_scratch[];
extern __global int32_t pathfinder_output[];
extern __global uint32_t pathfinder_owners[];
}

static inline int32_t minimum3(int32_t left, int32_t up, int32_t right) {
  int32_t value = left < up ? left : up;
  return right < value ? right : value;
}

__attribute__((noinline, convergent))
static void pathfinder_phase_sync(uint32_t warps) {
  // Every scheduled warp executes both instructions.  The fence publishes
  // that warp's global scratch stores to the cluster-shared hierarchy; the
  // local barrier prevents any warp from consuming the row or reusing the
  // alternate ping-pong buffer until every warp has published its phase.
  mu_fence();
  mu_barrier(kPathfinderPhaseBarrier, warps);
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_block,
                        uint32_t block_id) {
  // A final-row cache line belongs to exactly one physical SM.  Cross-SM
  // barriers are unavailable, so each SM cooperatively advances the complete
  // dependency cone for its own final-row interval in private ping-pong
  // scratch.  Neighboring SMs intentionally recompute overlapping halo
  // columns; they never exchange intermediate state or overlap final writes.
  const MuPartition owned = mu_contiguous_partition(
      PATHFINDER_COLS, NIGHTLY_CLUSTERS, block_id,
      PATHFINDER_INTS_PER_LINE);
  const uint32_t scratch_stride = 2u * PATHFINDER_COLS;
  __global int32_t* prev = pathfinder_scratch + block_id * scratch_stride;
  __global int32_t* next = prev + PATHFINDER_COLS;
  const uint32_t warps = threads_per_block / MU_NUM_THREADS;

  const uint32_t halo = PATHFINDER_ROWS - 1u;
  const uint32_t initial_begin = owned.begin > halo ? owned.begin - halo : 0u;
  uint32_t initial_end = owned.end + halo;
  if (initial_end > PATHFINDER_COLS) initial_end = PATHFINDER_COLS;
  for (uint32_t column = initial_begin + tid; column < initial_end;
       column += threads_per_block)
    prev[column] = pathfinder_wall[column];
  pathfinder_phase_sync(warps);

  for (uint32_t row = 1u; row < PATHFINDER_ROWS; ++row) {
    const uint32_t remaining = PATHFINDER_ROWS - 1u - row;
    const uint32_t begin =
        owned.begin > remaining ? owned.begin - remaining : 0u;
    uint32_t end = owned.end + remaining;
    if (end > PATHFINDER_COLS) end = PATHFINDER_COLS;
    const uint32_t wall_base = row * PATHFINDER_COLS;
    for (uint32_t column = begin + tid; column < end;
         column += threads_per_block) {
      const int32_t up = prev[column];
      const int32_t left = column == 0u ? up : prev[column - 1u];
      const int32_t right =
          column + 1u == PATHFINDER_COLS ? up : prev[column + 1u];
      next[column] = pathfinder_wall[wall_base + column] +
                     minimum3(left, up, right);
    }
    pathfinder_phase_sync(warps);
    __global int32_t* temporary = prev;
    prev = next;
    next = temporary;
  }

  for (uint32_t column = owned.begin + tid; column < owned.end;
       column += threads_per_block) {
    pathfinder_output[column] = prev[column];
    pathfinder_owners[column] = block_id;
  }
  pathfinder_phase_sync(warps);

  nightly_kernel_end(tid, threads_per_block);
}

int main() { return nightly_main(kernel_body, nullptr); }
