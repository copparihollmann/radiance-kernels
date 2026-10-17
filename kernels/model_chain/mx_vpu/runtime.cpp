#include "mx_vpu/backend.hpp"

namespace model_chain { namespace mx_model {
alignas(64) RK_GLOBAL const uint8_t unit_scales[64] = {
  127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,
  127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,
  127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,
  127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127};
} }

#ifndef RADIANCE_DEVICE
#include "gemmini_radiance.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace model_chain { namespace mx_model {
namespace {
uint64_t cycle = 0;
uint64_t commands = 0, mx_commands = 0, vpu_commands = 0;
std::vector<uint8_t> memory(128 << 10);
uint64_t load(void*, uint64_t address, unsigned bytes) {
  uintptr_t a = uintptr_t(address & 0xffffffffu);
  constexpr uintptr_t base = 0x1000;
  assert(a >= base && a+bytes <= base+sizeof unit_scales && bytes <= 8);
  uint64_t result = 0;
  memcpy(&result, unit_scales+(a-base), bytes);
  return result;
}
void store(void*, uint64_t, uint64_t, unsigned) { std::abort(); }
gemmini_radiance_t* device() {
  static gemmini_radiance_t* instance = [] {
    gemmini_radiance_soc_t soc{nullptr, load, store, memory.data(), memory.size()};
    // Preserve the Rakanic fork's shipped preset, including its fallback VPU unit.
    auto* result = gemmini_radiance_new(GEMMINI_RADIANCE_BOTH, "radiance", "", &soc);
    assert(result);
    return result;
  }();
  return instance;
}
struct Report {
  ~Report() {
    if (commands) {
      gemmini_radiance_finish(device(), nullptr);
      std::fprintf(stderr, "MX_VPU_MODEL commands=%llu mx=%llu vpu=%llu cycles=%llu\n",
          (unsigned long long)commands, (unsigned long long)mx_commands,
          (unsigned long long)vpu_commands, (unsigned long long)cycle);
      gemmini_radiance_free(device());
    }
  }
} report;
}
void shared_store(uint32_t address, uint32_t value) {
  assert(address+4 <= memory.size());
  memcpy(memory.data()+address, &value, 4);
  ++cycle;
}
uint32_t shared_load(uint32_t address) {
  assert(address+4 <= memory.size());
  uint32_t result;
  memcpy(&result, memory.data()+address, 4);
  ++cycle;
  return result;
}
void command(uint32_t funct, uint64_t rs1, uint64_t rs2) {
  auto* d = device();
  for (unsigned half = 0; half < 2; ++half) {
    gemmini_radiance_write(d, cycle++, 0x10+4*half, rs1 >> (32*half), 4);
    gemmini_radiance_write(d, cycle++, 0x18+4*half, rs2 >> (32*half), 4);
  }
  while (!gemmini_radiance_write(d, cycle++, 0, 0x7bu | (funct << 25), 4)) {}
  ++commands;
  mx_commands += funct == 8;
  vpu_commands += funct == 33;
}
void fence() { while (gemmini_radiance_busy(device(), cycle++)) {} }
} }
#endif
