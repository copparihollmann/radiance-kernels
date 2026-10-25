#include <stdint.h>

#include <nightly/host.h>
#include <radiance.h>

#include "generated/config.h"
#include "generated/symbols.h"

static_assert(LBM_EXPECTED_POPULATION_WORDS ==
                  LBM_ACTIVE_CELLS * LBM_POPULATIONS,
              "population oracle must cover every active destination");
static_assert(LBM_GOLD_VELOCITY_WORDS == LBM_ACTIVE_CELLS * 3u,
              "gold must cover the complete three-component velocity field");

enum Population : uint32_t {
  C = 0, N, S, E, W, T, B, NE, NW, SE, SW,
  NT, NB, ST, SB, ET, EB, WT, WB
};

static inline uint32_t grid_index(int32_t x, int32_t y, int32_t z,
                                  uint32_t entry) {
  const int32_t cell = x + y * static_cast<int32_t>(LBM_PADDED_X) +
      z * static_cast<int32_t>(LBM_PADDED_X * LBM_PADDED_Y);
  return LBM_MARGIN_FLOATS + entry + LBM_CELL_ENTRIES * cell;
}

static float as_float(uint32_t bits) {
  union { uint32_t u; float f; } value = {bits};
  return value.f;
}

static uint32_t as_word(float value) {
  union { float f; uint32_t u; } bits = {value};
  return bits.u;
}

static float abs_float(float value) { return value < 0.0f ? -value : value; }

static bool finite_word(uint32_t bits) {
  return ((bits >> 23u) & 0xffu) != 0xffu;
}

static uint64_t digest_update(uint64_t digest, uint32_t word) {
  digest ^= word;
  return digest * 0x100000001B3ULL;
}

static bool guards_intact() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(LBM_GUARD_BEFORE_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(LBM_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16u; ++i) {
    if (before[i] != (0xA5B10000u ^ i) ||
        after[i] != (0x5A4E0000u ^ i))
      return false;
  }
  return true;
}

static inline float load_gather(volatile uint32_t* grid, uint32_t entry,
                                int32_t x, int32_t y, int32_t z) {
  switch (entry) {
    case C:  return as_float(grid[grid_index(x, y, z, C)]);
    case N:  return as_float(grid[grid_index(x, y - 1, z, N)]);
    case S:  return as_float(grid[grid_index(x, y + 1, z, S)]);
    case E:  return as_float(grid[grid_index(x - 1, y, z, E)]);
    case W:  return as_float(grid[grid_index(x + 1, y, z, W)]);
    case T:  return as_float(grid[grid_index(x, y, z - 1, T)]);
    case B:  return as_float(grid[grid_index(x, y, z + 1, B)]);
    case NE: return as_float(grid[grid_index(x - 1, y - 1, z, NE)]);
    case NW: return as_float(grid[grid_index(x + 1, y - 1, z, NW)]);
    case SE: return as_float(grid[grid_index(x - 1, y + 1, z, SE)]);
    case SW: return as_float(grid[grid_index(x + 1, y + 1, z, SW)]);
    case NT: return as_float(grid[grid_index(x, y - 1, z - 1, NT)]);
    case NB: return as_float(grid[grid_index(x, y - 1, z + 1, NB)]);
    case ST: return as_float(grid[grid_index(x, y + 1, z - 1, ST)]);
    case SB: return as_float(grid[grid_index(x, y + 1, z + 1, SB)]);
    case ET: return as_float(grid[grid_index(x - 1, y, z - 1, ET)]);
    case EB: return as_float(grid[grid_index(x - 1, y, z + 1, EB)]);
    case WT: return as_float(grid[grid_index(x + 1, y, z - 1, WT)]);
    default: return as_float(grid[grid_index(x + 1, y, z + 1, WB)]);
  }
}

static void velocity(volatile uint32_t* grid, int32_t x, int32_t y,
                     int32_t z, float* result) {
  float p[LBM_POPULATIONS];
  for (uint32_t entry = 0; entry < LBM_POPULATIONS; ++entry)
    p[entry] = load_gather(grid, entry, x, y, z);

  float rho = p[C] + p[N];
  rho = rho + p[S]; rho = rho + p[E]; rho = rho + p[W];
  rho = rho + p[T]; rho = rho + p[B]; rho = rho + p[NE];
  rho = rho + p[NW]; rho = rho + p[SE]; rho = rho + p[SW];
  rho = rho + p[NT]; rho = rho + p[NB]; rho = rho + p[ST];
  rho = rho + p[SB]; rho = rho + p[ET]; rho = rho + p[EB];
  rho = rho + p[WT]; rho = rho + p[WB];

  float ux = p[E] - p[W];
  ux = ux + p[NE]; ux = ux - p[NW];
  ux = ux + p[SE]; ux = ux - p[SW];
  ux = ux + p[ET]; ux = ux + p[EB];
  ux = ux - p[WT]; ux = ux - p[WB];
  float uy = p[N] - p[S];
  uy = uy + p[NE]; uy = uy + p[NW];
  uy = uy - p[SE]; uy = uy - p[SW];
  uy = uy + p[NT]; uy = uy + p[NB];
  uy = uy - p[ST]; uy = uy - p[SB];
  float uz = p[T] - p[B];
  uz = uz + p[NT]; uz = uz - p[NB];
  uz = uz + p[ST]; uz = uz - p[SB];
  uz = uz + p[ET]; uz = uz - p[EB];
  uz = uz + p[WT]; uz = uz - p[WB];
  result[0] = ux / rho;
  result[1] = uy / rho;
  result[2] = uz / rho;
}

int main() {
  volatile uint32_t* before = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(LBM_GUARD_BEFORE_ADDR));
  volatile uint32_t* output = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(LBM_OUTPUT_ADDR));
  volatile uint32_t* after = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(LBM_GUARD_AFTER_ADDR));
  for (uint32_t i = 0; i < 16u; ++i) {
    before[i] = 0xA5B10000u ^ i;
    after[i] = 0x5A4E0000u ^ i;
  }

  // Keep the initialized halo and flags needed by the official exporter, but
  // poison every destination population that the one-step kernel must write.
  for (uint32_t z = 0; z < LBM_SIZE_Z; ++z)
    for (uint32_t y = 0; y < LBM_SIZE_Y; ++y)
      for (uint32_t x = 0; x < LBM_SIZE_X; ++x)
        for (uint32_t entry = 0; entry < LBM_POPULATIONS; ++entry)
          output[grid_index(x, y, z, entry)] = 0x7fc00000u;
  asm volatile("fence rw, rw" ::: "memory");

  nightly_host_launch(NIGHTLY_CLUSTERS);
  const int completed = nightly_host_wait(NIGHTLY_CLUSTERS);
  nightly_host_report(NIGHTLY_CLUSTERS);
  asm volatile("fence rw, rw" ::: "memory");

  volatile uint32_t* expected = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(LBM_EXPECTED_POPULATIONS_ADDR));
  uint64_t population_digest = 0xCBF29CE484222325ULL;
  uint64_t expected_digest = 0xCBF29CE484222325ULL;
  uint32_t population_bad = 0u;
  uint32_t population_nonfinite = 0u;
  uint32_t packed = 0u;
  for (uint32_t z = 0; z < LBM_SIZE_Z; ++z) {
    for (uint32_t y = 0; y < LBM_SIZE_Y; ++y) {
      for (uint32_t x = 0; x < LBM_SIZE_X; ++x) {
        for (uint32_t entry = 0; entry < LBM_POPULATIONS; ++entry) {
          const uint32_t actual = output[grid_index(x, y, z, entry)];
          const uint32_t reference = expected[packed++];
          population_digest = digest_update(population_digest, actual);
          expected_digest = digest_update(expected_digest, reference);
          population_nonfinite += !finite_word(actual);
          population_bad += actual != reference;
        }
      }
    }
  }

  volatile uint32_t* gold = reinterpret_cast<volatile uint32_t*>(
      rad_device_to_host_address(LBM_GOLD_VELOCITY_ADDR));
  float max_gold = 0.0f;
  uint64_t gold_digest = 0xCBF29CE484222325ULL;
  for (uint32_t i = 0; i < LBM_GOLD_VELOCITY_WORDS; ++i) {
    const uint32_t word = gold[i];
    const float magnitude = abs_float(as_float(word));
    if (magnitude > max_gold) max_gold = magnitude;
    gold_digest = digest_update(gold_digest, word);
  }
  const float absolute_tolerance = 1.0e-4f * max_gold;
  uint64_t velocity_digest = 0xCBF29CE484222325ULL;
  uint32_t velocity_bad = 0u;
  uint32_t velocity_nonfinite = 0u;
  uint32_t velocity_exact = 0u;
  uint32_t velocity_index = 0u;
  for (uint32_t z = 0; z < LBM_SIZE_Z; ++z) {
    for (uint32_t y = 0; y < LBM_SIZE_Y; ++y) {
      for (uint32_t x = 0; x < LBM_SIZE_X; ++x) {
        float actual[3];
        velocity(output, static_cast<int32_t>(x), static_cast<int32_t>(y),
                 static_cast<int32_t>(z), actual);
        for (uint32_t axis = 0; axis < 3u; ++axis) {
          const uint32_t actual_word = as_word(actual[axis]);
          const uint32_t gold_word = gold[velocity_index++];
          velocity_digest = digest_update(velocity_digest, actual_word);
          velocity_exact += actual_word == gold_word;
          if (!finite_word(actual_word)) {
            ++velocity_nonfinite;
            ++velocity_bad;
          } else {
            const float reference = as_float(gold_word);
            const float difference = abs_float(reference - actual[axis]);
            if (!(difference <= absolute_tolerance ||
                  difference < 0.002f * abs_float(reference)))
              ++velocity_bad;
          }
        }
      }
    }
  }

  const bool guards_ok = guards_intact();
  const bool expected_digest_ok =
      expected_digest == LBM_EXPECTED_POPULATION_DIGEST;
  const bool population_digest_ok =
      population_digest == LBM_EXPECTED_POPULATION_DIGEST;
  const bool gold_digest_ok = gold_digest == LBM_GOLD_VELOCITY_DIGEST;
  const bool velocity_exact_digest =
      velocity_digest == LBM_GOLD_VELOCITY_DIGEST;

  nh_puts("LBM_POPULATION_CHECK n="); nh_putu(packed);
  nh_puts(" bad="); nh_putu(population_bad);
  nh_puts(" nonfinite="); nh_putu(population_nonfinite);
  nh_puts(" guards="); nh_putu(guards_ok);
  nh_puts(" population_digest="); nh_putx(population_digest, 16);
  nh_puts(" digest_ok="); nh_putu(population_digest_ok);
  nh_puts(" expected_digest_ok="); nh_putu(expected_digest_ok);
  nh_puts("\n");
  nh_puts("LBM_VELOCITY_CHECK n="); nh_putu(velocity_index);
  nh_puts(" exact="); nh_putu(velocity_exact);
  nh_puts(" bad="); nh_putu(velocity_bad);
  nh_puts(" nonfinite="); nh_putu(velocity_nonfinite);
  nh_puts(" velocity_digest="); nh_putx(velocity_digest, 16);
  nh_puts(" exact_digest="); nh_putu(velocity_exact_digest);
  nh_puts(" gold_digest_ok="); nh_putu(gold_digest_ok);
  nh_puts("\n");

  const bool passed = completed && guards_ok && expected_digest_ok &&
      population_digest_ok && population_bad == 0u &&
      population_nonfinite == 0u && gold_digest_ok &&
      velocity_bad == 0u && velocity_nonfinite == 0u;
  nh_puts(passed ? "PASS\n" : "FAIL\n");
  nh_flush();
  exit(passed ? 0 : 1);
}
