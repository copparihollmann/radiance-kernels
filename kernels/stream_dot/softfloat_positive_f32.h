#pragma once

#include <stdint.h>

static inline uint32_t stream_dot_shift_right_jam(uint32_t value,
                                                   uint32_t distance) {
  if (distance == 0u) return value;
  if (distance < 32u) {
    return (value >> distance) |
        static_cast<uint32_t>((value << (32u - distance)) != 0u);
  }
  return static_cast<uint32_t>(value != 0u);
}

// Add positive, finite, normal binary32 words with IEEE-754
// round-to-nearest, ties-to-even semantics. Zero is accepted as the reduction
// identity. Other encodings fail closed through valid.
static inline uint32_t stream_dot_add_positive_finite_f32_rne(
    uint32_t left, uint32_t right, bool* valid) {
  if (((left | right) & 0x80000000u) != 0u ||
      (left & 0x7f800000u) == 0x7f800000u ||
      (right & 0x7f800000u) == 0x7f800000u) {
    *valid = false;
    return 0u;
  }
  if ((left & 0x7fffffffu) == 0u) return right;
  if ((right & 0x7fffffffu) == 0u) return left;

  uint32_t left_exp = (left >> 23u) & 0xffu;
  uint32_t right_exp = (right >> 23u) & 0xffu;
  if (left_exp == 0u || right_exp == 0u) {
    *valid = false;
    return 0u;
  }
  if (right_exp > left_exp) {
    const uint32_t word = left;
    left = right;
    right = word;
    const uint32_t exponent = left_exp;
    left_exp = right_exp;
    right_exp = exponent;
  }

  uint32_t left_sig = (0x00800000u | (left & 0x007fffffu)) << 3u;
  uint32_t right_sig = (0x00800000u | (right & 0x007fffffu)) << 3u;
  right_sig = stream_dot_shift_right_jam(right_sig, left_exp - right_exp);
  uint32_t sum = left_sig + right_sig;
  if ((sum & 0x08000000u) != 0u) {
    sum = stream_dot_shift_right_jam(sum, 1u);
    ++left_exp;
  }

  uint32_t significand = sum >> 3u;
  const uint32_t round_bits = sum & 7u;
  if (round_bits > 4u || (round_bits == 4u && (significand & 1u) != 0u)) {
    ++significand;
    if ((significand & 0x01000000u) != 0u) {
      significand >>= 1u;
      ++left_exp;
    }
  }
  if (left_exp >= 0xffu) {
    *valid = false;
    return 0u;
  }
  return (left_exp << 23u) | (significand & 0x007fffffu);
}
