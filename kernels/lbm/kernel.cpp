#include <stdint.h>

#include "generated/config.h"
#include <nightly/device.h>

static_assert(NIGHTLY_CLUSTERS == HPC_SM_COUNT,
              "host and device cluster counts must match");
static_assert(LBM_SIZE_X == 32u && LBM_SIZE_Y == 32u && LBM_SIZE_Z == 8u,
              "only the official 32x32x8 contract is supported");
static_assert(LBM_POPULATIONS == 19u && LBM_CELL_ENTRIES == 20u,
              "the pinned D3Q19 AoS layout changed");
static_assert(LBM_GROUP_CELLS == 4u && LBM_GROUP_FLOATS == 80u,
              "each work unit must be five 64-byte cache lines");

extern "C" {
extern __global const float lbm_input_raw[];
extern __global float lbm_output_raw[];
}

static constexpr uint32_t kGroupBoundaries[HPC_SM_COUNT + 1u] =
    LBM_SM_GROUP_BOUNDARIES;

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

static inline float load_gather(uint32_t entry, int32_t x, int32_t y,
                                int32_t z) {
  switch (entry) {
    case C:  return lbm_input_raw[grid_index(x, y, z, C)];
    case N:  return lbm_input_raw[grid_index(x, y - 1, z, N)];
    case S:  return lbm_input_raw[grid_index(x, y + 1, z, S)];
    case E:  return lbm_input_raw[grid_index(x - 1, y, z, E)];
    case W:  return lbm_input_raw[grid_index(x + 1, y, z, W)];
    case T:  return lbm_input_raw[grid_index(x, y, z - 1, T)];
    case B:  return lbm_input_raw[grid_index(x, y, z + 1, B)];
    case NE: return lbm_input_raw[grid_index(x - 1, y - 1, z, NE)];
    case NW: return lbm_input_raw[grid_index(x + 1, y - 1, z, NW)];
    case SE: return lbm_input_raw[grid_index(x - 1, y + 1, z, SE)];
    case SW: return lbm_input_raw[grid_index(x + 1, y + 1, z, SW)];
    case NT: return lbm_input_raw[grid_index(x, y - 1, z - 1, NT)];
    case NB: return lbm_input_raw[grid_index(x, y - 1, z + 1, NB)];
    case ST: return lbm_input_raw[grid_index(x, y + 1, z - 1, ST)];
    case SB: return lbm_input_raw[grid_index(x, y + 1, z + 1, SB)];
    case ET: return lbm_input_raw[grid_index(x - 1, y, z - 1, ET)];
    case EB: return lbm_input_raw[grid_index(x - 1, y, z + 1, EB)];
    case WT: return lbm_input_raw[grid_index(x + 1, y, z - 1, WT)];
    default: return lbm_input_raw[grid_index(x + 1, y, z + 1, WB)];
  }
}

#pragma clang fp contract(off)
static inline float collide_axis(float old_value, float velocity,
                                 float sign, float temp1, float temp2,
                                 float u2) {
  const float inner = 4.5f * velocity + sign * 3.0f;
  const float equilibrium = velocity * inner - u2;
  return temp2 * old_value + temp1 * equilibrium;
}

#pragma clang fp contract(off)
static inline float collide_diagonal(float old_value, float velocity,
                                     float temp1, float temp2, float u2) {
  const float inner = 4.5f * velocity + 3.0f;
  const float equilibrium = velocity * inner - u2;
  return temp2 * old_value + temp1 * equilibrium;
}

#pragma clang fp contract(off)
static inline void stream_collide(int32_t x, int32_t y, int32_t z) {
  // Keep every population as a distinct scalar. Muon's lane-interleaved stack
  // cannot safely represent a conventionally indexed local C array.
  float tempC = load_gather(C, x, y, z);
  float tempN = load_gather(N, x, y, z);
  float tempS = load_gather(S, x, y, z);
  float tempE = load_gather(E, x, y, z);
  float tempW = load_gather(W, x, y, z);
  float tempT = load_gather(T, x, y, z);
  float tempB = load_gather(B, x, y, z);
  float tempNE = load_gather(NE, x, y, z);
  float tempNW = load_gather(NW, x, y, z);
  float tempSE = load_gather(SE, x, y, z);
  float tempSW = load_gather(SW, x, y, z);
  float tempNT = load_gather(NT, x, y, z);
  float tempNB = load_gather(NB, x, y, z);
  float tempST = load_gather(ST, x, y, z);
  float tempSB = load_gather(SB, x, y, z);
  float tempET = load_gather(ET, x, y, z);
  float tempEB = load_gather(EB, x, y, z);
  float tempWT = load_gather(WT, x, y, z);
  float tempWB = load_gather(WB, x, y, z);

  const uint32_t flags = __builtin_bit_cast(
      uint32_t, lbm_input_raw[grid_index(x, y, z, LBM_POPULATIONS)]);
  if (flags & 1u) {
    float swap;
    swap = tempN; tempN = tempS; tempS = swap;
    swap = tempE; tempE = tempW; tempW = swap;
    swap = tempT; tempT = tempB; tempB = swap;
    swap = tempNE; tempNE = tempSW; tempSW = swap;
    swap = tempNW; tempNW = tempSE; tempSE = swap;
    swap = tempNT; tempNT = tempSB; tempSB = swap;
    swap = tempNB; tempNB = tempST; tempST = swap;
    swap = tempET; tempET = tempWB; tempWB = swap;
    swap = tempEB; tempEB = tempWT; tempWT = swap;
  } else {
    float rho = tempC + tempN;
    rho = rho + tempS; rho = rho + tempE; rho = rho + tempW;
    rho = rho + tempT; rho = rho + tempB; rho = rho + tempNE;
    rho = rho + tempNW; rho = rho + tempSE; rho = rho + tempSW;
    rho = rho + tempNT; rho = rho + tempNB; rho = rho + tempST;
    rho = rho + tempSB; rho = rho + tempET; rho = rho + tempEB;
    rho = rho + tempWT; rho = rho + tempWB;

    float ux = tempE - tempW;
    ux = ux + tempNE; ux = ux - tempNW;
    ux = ux + tempSE; ux = ux - tempSW;
    ux = ux + tempET; ux = ux + tempEB;
    ux = ux - tempWT; ux = ux - tempWB;
    float uy = tempN - tempS;
    uy = uy + tempNE; uy = uy + tempNW;
    uy = uy - tempSE; uy = uy - tempSW;
    uy = uy + tempNT; uy = uy + tempNB;
    uy = uy - tempST; uy = uy - tempSB;
    float uz = tempT - tempB;
    uz = uz + tempNT; uz = uz - tempNB;
    uz = uz + tempST; uz = uz - tempSB;
    uz = uz + tempET; uz = uz - tempEB;
    uz = uz + tempWT; uz = uz - tempWB;

    ux = ux / rho; uy = uy / rho; uz = uz / rho;
    if (flags & 2u) {
      ux = 0.005f; uy = 0.002f; uz = 0.0f;
    }

    float squared = ux * ux + uy * uy;
    squared = squared + uz * uz;
    const float u2 = 1.5f * squared - 1.0f;
    float temp_base = 1.95f * rho;
    float temp1 = (1.0f / 3.0f) * temp_base;
    // These duplicate assignments are retained from the pinned kernel.cl.
    temp_base = 1.95f * rho;
    temp1 = (1.0f / 3.0f) * temp_base;
    const float temp2 = 1.0f - 1.95f;
    tempC = temp2 * tempC + temp1 * (-u2);

    temp1 = (1.0f / 18.0f) * temp_base;
    tempN = collide_axis(tempN, uy, +1.0f, temp1, temp2, u2);
    tempS = collide_axis(tempS, uy, -1.0f, temp1, temp2, u2);
    tempT = collide_axis(tempT, uz, +1.0f, temp1, temp2, u2);
    tempB = collide_axis(tempB, uz, -1.0f, temp1, temp2, u2);
    tempE = collide_axis(tempE, ux, +1.0f, temp1, temp2, u2);
    tempW = collide_axis(tempW, ux, -1.0f, temp1, temp2, u2);

    temp1 = (1.0f / 36.0f) * temp_base;
    tempNT = collide_diagonal(tempNT, +uy + uz, temp1, temp2, u2);
    tempNB = collide_diagonal(tempNB, +uy - uz, temp1, temp2, u2);
    tempST = collide_diagonal(tempST, -uy + uz, temp1, temp2, u2);
    tempSB = collide_diagonal(tempSB, -uy - uz, temp1, temp2, u2);
    tempNE = collide_diagonal(tempNE, +ux + uy, temp1, temp2, u2);
    tempSE = collide_diagonal(tempSE, +ux - uy, temp1, temp2, u2);
    tempET = collide_diagonal(tempET, +ux + uz, temp1, temp2, u2);
    tempEB = collide_diagonal(tempEB, +ux - uz, temp1, temp2, u2);
    tempNW = collide_diagonal(tempNW, -ux + uy, temp1, temp2, u2);
    tempSW = collide_diagonal(tempSW, -ux - uy, temp1, temp2, u2);
    tempWT = collide_diagonal(tempWT, -ux + uz, temp1, temp2, u2);
    tempWB = collide_diagonal(tempWB, -ux - uz, temp1, temp2, u2);
  }

  lbm_output_raw[grid_index(x, y, z, C)] = tempC;
  lbm_output_raw[grid_index(x, y, z, N)] = tempN;
  lbm_output_raw[grid_index(x, y, z, S)] = tempS;
  lbm_output_raw[grid_index(x, y, z, E)] = tempE;
  lbm_output_raw[grid_index(x, y, z, W)] = tempW;
  lbm_output_raw[grid_index(x, y, z, T)] = tempT;
  lbm_output_raw[grid_index(x, y, z, B)] = tempB;
  lbm_output_raw[grid_index(x, y, z, NE)] = tempNE;
  lbm_output_raw[grid_index(x, y, z, NW)] = tempNW;
  lbm_output_raw[grid_index(x, y, z, SE)] = tempSE;
  lbm_output_raw[grid_index(x, y, z, SW)] = tempSW;
  lbm_output_raw[grid_index(x, y, z, NT)] = tempNT;
  lbm_output_raw[grid_index(x, y, z, NB)] = tempNB;
  lbm_output_raw[grid_index(x, y, z, ST)] = tempST;
  lbm_output_raw[grid_index(x, y, z, SB)] = tempSB;
  lbm_output_raw[grid_index(x, y, z, ET)] = tempET;
  lbm_output_raw[grid_index(x, y, z, EB)] = tempEB;
  lbm_output_raw[grid_index(x, y, z, WT)] = tempWT;
  lbm_output_raw[grid_index(x, y, z, WB)] = tempWB;
}

static void kernel_body(void*, uint32_t tid, uint32_t threads_per_threadblock,
                        uint32_t physical_threadblock_id) {
  const uint32_t begin = kGroupBoundaries[physical_threadblock_id];
  const uint32_t end = kGroupBoundaries[physical_threadblock_id + 1u];
  for (uint32_t group = begin + tid; group < end;
       group += threads_per_threadblock) {
    const uint32_t first_cell = group * LBM_GROUP_CELLS;
    for (uint32_t lane = 0; lane < LBM_GROUP_CELLS; ++lane) {
      const uint32_t cell = first_cell + lane;
      const int32_t x = static_cast<int32_t>(cell % LBM_SIZE_X);
      const uint32_t yz = cell / LBM_SIZE_X;
      const int32_t y = static_cast<int32_t>(yz % LBM_SIZE_Y);
      const int32_t z = static_cast<int32_t>(yz / LBM_SIZE_Y);
      stream_collide(x, y, z);
    }
  }
  nightly_kernel_end(tid, threads_per_threadblock);
}

int main() { return nightly_main(kernel_body, nullptr); }
