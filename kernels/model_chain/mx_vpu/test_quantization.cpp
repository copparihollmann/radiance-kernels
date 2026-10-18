// Independent nearest-value oracle for the packing code used by device MX.
#include "backend.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

static float from_bits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof value);
  return value;
}

static uint8_t oracle(float value) {
  const bool negative = std::signbit(value);
  const double magnitude = std::fabs(double(value));
  double best_distance = INFINITY;
  unsigned best = 0;
  for (unsigned code = 0; code <= 126; ++code) {
    const unsigned exponent = code >> 3, fraction = code & 7;
    const double decoded = exponent ? std::ldexp(double(8 + fraction), int(exponent) - 10)
                                    : std::ldexp(double(fraction), -9);
    const double distance = std::fabs(magnitude - decoded);
    if (distance < best_distance || (distance == best_distance && !(code & 1))) {
      best = code;
      best_distance = distance;
    }
  }
  return uint8_t(best | (negative ? 128 : 0));
}

int main() {
  uint64_t checked = 0;
  auto check = [&](float value) {
    if (!std::isfinite(value)) return;
    const auto actual = model_chain::mx_model::fp8(value), expected = oracle(value);
    if (actual != expected) {
      std::fprintf(stderr, "fp8(%a) = %u, expected %u\n", value, actual, expected);
      std::abort();
    }
    ++checked;
  };
  // Every finite BF16 value, both signs, including zeros and subnormals.
  for (uint32_t bits = 0; bits < 65536; ++bits) check(from_bits(bits << 16));
  // Every E4M3 rounding boundary and the adjacent FP32 values.
  float previous = 0;
  for (unsigned code = 1; code <= 126; ++code) {
    const unsigned exponent = code >> 3, fraction = code & 7;
    const float next = exponent ? std::ldexp(float(8 + fraction), int(exponent) - 10)
                                : std::ldexp(float(fraction), -9);
    const float midpoint = (next + previous) * 0.5f;
    for (float v : {std::nextafter(midpoint, -INFINITY), midpoint,
                    std::nextafter(midpoint, INFINITY)}) { check(v); check(-v); }
    previous = next;
  }
  uint32_t random = 0x853c49e6u;
  uint64_t normalization_checks = 0;
  for (unsigned i = 0; i < 1000000; ++i) {
    random ^= random << 13; random ^= random >> 17; random ^= random << 5;
    const float value = from_bits(random);
    check(value);
    if (std::isfinite(value)) {
      const int exponent = value ? model_chain::mx_model::block_exponent(std::fabs(value)) : 0;
      const float divided = value / model_chain::mx_model::power2(exponent);
      const float multiplied = value * model_chain::mx_model::power2(-exponent);
      assert(std::memcmp(&divided, &multiplied, sizeof divided) == 0);
      ++normalization_checks;
    }
  }
  std::printf("{\"finite_fp8_checks\":%llu,\"normalization_checks\":%llu}\n",
              (unsigned long long)checked, (unsigned long long)normalization_checks);
}
