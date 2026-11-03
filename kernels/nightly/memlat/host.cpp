// memlat host side (Rocket, rv64, NO FPU -- integer arithmetic only).
//
// The host owns the experiment's ground truth.  It builds both next-pointer arrays itself,
// verifies that the random one really is a single MEMLAT_NODES-cycle before anything is launched,
// walks it to obtain the terminal node, the sum and the xor that a correct traversal must
// produce, and only then launches.  The device measures and reports; the host decides PASS/FAIL.
//
// Output is one machine row per (cluster, level):
//
//   memlat cluster=0 level=0 level_name=idle loads=4096 seq_cycles=... rand_cycles=...
//          seq_cyc_per_load_x100=... rand_cyc_per_load_x100=... muon_bytes=... dma_bytes=...
//          level_cycles=... muon_b_per_cyc_x100=... dma_b_per_cyc_x100=...
//
// preceded by a human-readable line of the same numbers, plus one `memlat_check` row per chase
// and a final `memlat_pass=`.
#include <nightly/host.h>
#include "memlat.h"

static const char *const MEMLAT_LEVEL_NAME[MEMLAT_LEVELS] = {
    "idle", "muon", "dma", "muon_dma"};

// Host-side permutations.  16 KiB each; static so they are not on the Rocket's stack.
static uint32_t next_seq[MEMLAT_NODES];
static uint32_t next_rand[MEMLAT_NODES];

static uint32_t lcg(uint32_t *state) {
  *state = *state * 1664525u + 1013904223u;
  return *state >> 8;   // the low bits of an LCG are the weak ones
}

// Sattolo's algorithm: the resulting mapping i -> p[i] is a single cycle of length n.
static void build_rand(uint32_t *p, uint32_t n, uint32_t seed) {
  for (uint32_t i = 0; i < n; i++) p[i] = i;
  uint32_t state = seed;
  for (uint32_t i = n - 1; i > 0; i--) {
    const uint32_t j = lcg(&state) % i;
    const uint32_t t = p[i];
    p[i] = p[j];
    p[j] = t;
  }
}

struct Walk { uint32_t steps; uint32_t terminal; uint32_t sum; uint32_t hash; };

// Walk `next` from node 0 for exactly n steps, recording what a correct device chase must report.
static Walk walk(const uint32_t *next, uint32_t n) {
  Walk w = {0u, 0u, 0u, 0u};
  uint32_t i = 0;
  for (uint32_t k = 0; k < n; k++) {
    i = next[i];
    w.sum += i;
    w.hash = w.hash * MEMLAT_HASH_MUL + i;
    w.steps++;
  }
  w.terminal = i;
  return w;
}

// Independent structural check that `next` is ONE cycle covering every node: return to node 0
// after exactly n steps and never before.
static int is_single_cycle(const uint32_t *next, uint32_t n) {
  uint32_t i = 0;
  for (uint32_t k = 1; k <= n; k++) {
    i = next[i];
    if (i >= n) return 0;
    if (i == 0 && k != n) return 0;
  }
  return i == 0;
}

static void write_chase(uint32_t dev_addr, const uint32_t *next, uint32_t n) {
  volatile uint32_t *p = nightly_gpu_ptr(dev_addr);
  for (uint32_t i = 0; i < n; i++) p[i * MEMLAT_STRIDE_WORDS] = next[i];
}

static uint64_t ratio_x100(uint64_t num, uint64_t den) {
  return den ? (num * 100ull) / den : 0ull;
}

int main() {
  nh_puts("memlat: nodes="); nh_putu(MEMLAT_NODES);
  nh_puts(" stride="); nh_putu(MEMLAT_STRIDE_BYTES);
  nh_puts(" working_set_bytes="); nh_putu(MEMLAT_CHASE_BYTES);
  nh_puts(" loads_per_chase="); nh_putu(MEMLAT_LOADS);
  nh_puts(" clusters="); nh_putu(NIGHTLY_CLUSTERS);
  nh_puts("\n"); nh_flush();

  for (uint32_t i = 0; i < MEMLAT_NODES; i++) next_seq[i] = (i + 1u) % MEMLAT_NODES;
  build_rand(next_rand, MEMLAT_NODES, 0x5EEDBEEFu);

  if (!is_single_cycle(next_seq, MEMLAT_NODES) ||
      !is_single_cycle(next_rand, MEMLAT_NODES)) {
    nh_puts("memlat: host permutation is not a single cycle\nFAIL\n");
    nh_flush();
    exit(1);
  }
  const Walk gold_seq = walk(next_seq, MEMLAT_LOADS);
  const Walk gold_rand = walk(next_rand, MEMLAT_LOADS);
  nh_puts("memlat_gold seq_terminal="); nh_putu(gold_seq.terminal);
  nh_puts(" seq_sum="); nh_putu(gold_seq.sum);
  nh_puts(" seq_hash="); nh_putu(gold_seq.hash);
  nh_puts(" rand_terminal="); nh_putu(gold_rand.terminal);
  nh_puts(" rand_sum="); nh_putu(gold_rand.sum);
  nh_puts(" rand_hash="); nh_putu(gold_rand.hash);
  nh_puts("\n"); nh_flush();

  // Only the first word of each 64 B node is used, so this writes MEMLAT_NODES words per buffer,
  // not MEMLAT_CHASE_BYTES.  The two background regions are read-only traffic whose values are
  // discarded, so they are left as they are in DRAM instead of spending host time filling them.
  write_chase(MEMLAT_SEQ_ADDR, next_seq, MEMLAT_NODES);
  write_chase(MEMLAT_RAND_ADDR, next_rand, MEMLAT_NODES);

  // Clear this kernel's result slots: a stale value from a previous run must not be able to look
  // like a measurement.
  for (uint32_t cl = 0; cl < NIGHTLY_CLUSTERS; cl++)
    for (uint32_t s = MEMLAT_PB_BASE; s <= MEMLAT_PB_STATUS; s++)
      nightly_pb_write(cl, s, 0);

  const uint64_t h0 = nightly_rdcycle();
  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int ok = nightly_host_wait(NIGHTLY_CLUSTERS);
  const uint64_t h1 = nightly_rdcycle();
  nh_puts("wait ok="); nh_putu(ok); nh_puts(" host_cycles="); nh_putu(h1 - h0);
  nh_puts("\n"); nh_flush();

  nightly_host_report(NIGHTLY_CLUSTERS);

  uint32_t bad = 0;
  for (uint32_t cl = 0; cl < NIGHTLY_CLUSTERS; cl++) {
    const uint32_t status = nightly_pb_read(cl, MEMLAT_PB_STATUS);
    const uint32_t flags = status & 0xFFFFu;
    if ((status & MEMLAT_STATUS_MASK) != MEMLAT_STATUS_MAGIC) {
      nh_puts("memlat_check cluster="); nh_putu(cl);
      nh_puts(" chase=none status_magic_ok=0 ok=0\n");
      bad++;
      continue;
    }
    const uint32_t got[2][3] = {
        {nightly_pb_read(cl, MEMLAT_PB_TERM_SEQ), nightly_pb_read(cl, MEMLAT_PB_SUM_SEQ),
         nightly_pb_read(cl, MEMLAT_PB_HASH_SEQ)},
        {nightly_pb_read(cl, MEMLAT_PB_TERM_RAND), nightly_pb_read(cl, MEMLAT_PB_SUM_RAND),
         nightly_pb_read(cl, MEMLAT_PB_HASH_RAND)},
    };
    const uint32_t want[2][3] = {
        {gold_seq.terminal, gold_seq.sum, gold_seq.hash},
        {gold_rand.terminal, gold_rand.sum, gold_rand.hash},
    };
    for (uint32_t c = 0; c < 2; c++) {
      const int good = got[c][0] == want[c][0] && got[c][1] == want[c][1] &&
                       got[c][2] == want[c][2];
      if (!good) bad++;
      nh_puts("memlat_check cluster="); nh_putu(cl);
      nh_puts(c ? " chase=rand" : " chase=seq");
      nh_puts(" loads="); nh_putu(MEMLAT_LOADS);
      nh_puts(" terminal="); nh_putu(got[c][0]);
      nh_puts(" terminal_expected="); nh_putu(want[c][0]);
      nh_puts(" checksum="); nh_putu(got[c][1]);
      nh_puts(" checksum_expected="); nh_putu(want[c][1]);
      nh_puts(" order_hash="); nh_putu(got[c][2]);
      nh_puts(" order_hash_expected="); nh_putu(want[c][2]);
      nh_puts(" ok="); nh_putu(good ? 1u : 0u);
      nh_puts("\n");
    }
    // Every level's chases must reproduce level 0's traversal, not just the first.
    const uint32_t levels_agree = (flags & MEMLAT_F_CHASE_DIFF) == 0u;
    if (!levels_agree) bad++;
    nh_puts("memlat_check cluster="); nh_putu(cl);
    nh_puts(" chase=all_levels levels="); nh_putu(MEMLAT_LEVELS);
    nh_puts(" ok="); nh_putu(levels_agree);
    nh_puts(" dma_ready_timeout="); nh_putu((flags & MEMLAT_F_DMA_READY) ? 1u : 0u);
    nh_puts(" dma_drain_timeout="); nh_putu((flags & MEMLAT_F_DMA_DRAIN) ? 1u : 0u);
    nh_puts("\n");
    nh_flush();
  }

  for (uint32_t cl = 0; cl < NIGHTLY_CLUSTERS; cl++) {
    for (uint32_t lv = 0; lv < MEMLAT_LEVELS; lv++) {
      const uint32_t base = MEMLAT_PB_LEVEL(lv);
      const uint32_t sc = nightly_pb_read(cl, base + MEMLAT_PB_SEQ_CYCLES);
      const uint32_t rc = nightly_pb_read(cl, base + MEMLAT_PB_RAND_CYCLES);
      const uint32_t mb = nightly_pb_read(cl, base + MEMLAT_PB_MUON_BYTES);
      const uint32_t db = nightly_pb_read(cl, base + MEMLAT_PB_DMA_BYTES);
      const uint32_t lc = nightly_pb_read(cl, base + MEMLAT_PB_LEVEL_CYCLES);
      const uint64_t seq_x100 = ratio_x100(sc, MEMLAT_LOADS);
      const uint64_t rand_x100 = ratio_x100(rc, MEMLAT_LOADS);
      const uint64_t muon_x100 = ratio_x100(mb, lc);
      const uint64_t dma_x100 = ratio_x100(db, lc);

      nh_puts("memlat cluster "); nh_putu(cl);
      nh_puts(" level "); nh_putu(lv); nh_puts(" ");
      nh_puts(MEMLAT_LEVEL_NAME[lv]);
      nh_puts(": seq "); nh_putfix(seq_x100, 2);
      nh_puts(" rand "); nh_putfix(rand_x100, 2);
      nh_puts(" cyc/load  dma_B/cyc "); nh_putfix(dma_x100, 2);
      nh_puts(" muon_B/cyc "); nh_putfix(muon_x100, 2);
      nh_puts(" level_cycles "); nh_putu(lc);
      nh_puts("\n");

      nh_puts("memlat cluster="); nh_putu(cl);
      nh_puts(" level="); nh_putu(lv);
      nh_puts(" level_name="); nh_puts(MEMLAT_LEVEL_NAME[lv]);
      nh_puts(" loads="); nh_putu(MEMLAT_LOADS);
      nh_puts(" seq_cycles="); nh_putu(sc);
      nh_puts(" rand_cycles="); nh_putu(rc);
      nh_puts(" seq_cyc_per_load_x100="); nh_putu(seq_x100);
      nh_puts(" rand_cyc_per_load_x100="); nh_putu(rand_x100);
      nh_puts(" muon_bytes="); nh_putu(mb);
      nh_puts(" dma_bytes="); nh_putu(db);
      nh_puts(" level_cycles="); nh_putu(lc);
      nh_puts(" muon_b_per_cyc_x100="); nh_putu(muon_x100);
      nh_puts(" dma_b_per_cyc_x100="); nh_putu(dma_x100);
      nh_puts("\n");
      nh_flush();

      if (!sc || !rc || !lc) bad++;   // a level that produced no measurement is a failure
    }
  }

  nh_puts("memlat_pass="); nh_putu((ok && !bad) ? 1u : 0u); nh_puts("\n");
  nh_puts(bad ? "FAIL\n" : "PASS\n");
  nh_flush();
  exit(ok && !bad ? 0 : 1);
}
