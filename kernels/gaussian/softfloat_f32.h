#pragma once

#include <stdint.h>

// A deliberately small integer-only IEEE-754 binary32 implementation for the
// RV64 Gaussian checker.  It accepts finite operands, implements
// round-to-nearest/ties-to-even, and fails closed on NaN, infinity, division by
// zero, or a finite-operation overflow.  No C/C++ floating-point type appears
// in this file, so the checker is usable on hosts where the F/D state is not a
// trusted part of the benchmark environment.

struct GaussianF32 {
  bool sign;
  bool zero;
  uint32_t significand;  // normalized with bit 23 set when nonzero
  int32_t exponent;      // unbiased exponent of significand bit 23
};

static inline uint64_t gaussian_shift_right_jam_u64(uint64_t value,
                                                     uint32_t distance) {
  if (distance == 0u) return value;
  if (distance < 64u) {
    const uint64_t mask = (UINT64_C(1) << distance) - 1u;
    return (value >> distance) | static_cast<uint64_t>((value & mask) != 0u);
  }
  return static_cast<uint64_t>(value != 0u);
}

static inline bool gaussian_f32_decode(uint32_t word, GaussianF32* out) {
  out->sign = (word >> 31u) != 0u;
  const uint32_t encoded_exponent = (word >> 23u) & 0xffu;
  uint32_t significand = word & 0x007fffffu;
  if (encoded_exponent == 0xffu) return false;
  if (encoded_exponent == 0u) {
    if (significand == 0u) {
      out->zero = true;
      out->significand = 0u;
      out->exponent = -126;
      return true;
    }
    int32_t exponent = -126;
    while ((significand & 0x00800000u) == 0u) {
      significand <<= 1u;
      --exponent;
    }
    out->zero = false;
    out->significand = significand;
    out->exponent = exponent;
    return true;
  }
  out->zero = false;
  out->significand = 0x00800000u | significand;
  out->exponent = static_cast<int32_t>(encoded_exponent) - 127;
  return true;
}

// Pack a normalized significand whose hidden bit is at bit 26.  Bits 2:0 are
// guard, round, and sticky respectively.
static inline uint32_t gaussian_f32_round_pack(bool sign, int32_t exponent,
                                                uint64_t significand,
                                                bool* valid) {
  if (significand == 0u) return static_cast<uint32_t>(sign) << 31u;
  while (significand >= (UINT64_C(1) << 27u)) {
    significand = gaussian_shift_right_jam_u64(significand, 1u);
    ++exponent;
  }
  while (significand < (UINT64_C(1) << 26u)) {
    significand <<= 1u;
    --exponent;
  }
  if (exponent < -126) {
    const int32_t distance = -126 - exponent;
    significand = gaussian_shift_right_jam_u64(
        significand, distance >= 64 ? 64u : static_cast<uint32_t>(distance));
    exponent = -126;
  }

  uint32_t rounded = static_cast<uint32_t>(significand >> 3u);
  const uint32_t round_bits = static_cast<uint32_t>(significand & 7u);
  if (round_bits > 4u || (round_bits == 4u && (rounded & 1u) != 0u)) {
    ++rounded;
  }
  if ((rounded & 0x01000000u) != 0u) {
    rounded >>= 1u;
    ++exponent;
  }
  if (exponent > 127) {
    *valid = false;
    return 0u;
  }

  const uint32_t sign_word = static_cast<uint32_t>(sign) << 31u;
  if (rounded == 0u) return sign_word;
  if (exponent == -126 && rounded < 0x00800000u)
    return sign_word | rounded;
  if (exponent < -126 || rounded < 0x00800000u) {
    *valid = false;
    return 0u;
  }
  return sign_word |
      (static_cast<uint32_t>(exponent + 127) << 23u) |
      (rounded & 0x007fffffu);
}

static inline uint32_t gaussian_f32_add(uint32_t left, uint32_t right,
                                         bool* valid) {
  GaussianF32 a, b;
  if (!gaussian_f32_decode(left, &a) || !gaussian_f32_decode(right, &b)) {
    *valid = false;
    return 0u;
  }
  if (a.zero && b.zero) {
    return static_cast<uint32_t>(a.sign && b.sign) << 31u;
  }
  if (a.zero) return right;
  if (b.zero) return left;

  if (a.exponent < b.exponent ||
      (a.exponent == b.exponent && a.significand < b.significand)) {
    const GaussianF32 temporary = a;
    a = b;
    b = temporary;
  }
  uint64_t a_sig = static_cast<uint64_t>(a.significand) << 3u;
  uint64_t b_sig = static_cast<uint64_t>(b.significand) << 3u;
  const int32_t exponent_difference = a.exponent - b.exponent;
  b_sig = gaussian_shift_right_jam_u64(
      b_sig, exponent_difference >= 64
          ? 64u : static_cast<uint32_t>(exponent_difference));

  if (a.sign == b.sign) {
    return gaussian_f32_round_pack(
        a.sign, a.exponent, a_sig + b_sig, valid);
  }
  if (a_sig == b_sig) return 0u;
  return gaussian_f32_round_pack(
      a.sign, a.exponent, a_sig - b_sig, valid);
}

static inline uint32_t gaussian_f32_sub(uint32_t left, uint32_t right,
                                         bool* valid) {
  return gaussian_f32_add(left, right ^ 0x80000000u, valid);
}

static inline uint32_t gaussian_f32_mul(uint32_t left, uint32_t right,
                                         bool* valid) {
  GaussianF32 a, b;
  if (!gaussian_f32_decode(left, &a) || !gaussian_f32_decode(right, &b)) {
    *valid = false;
    return 0u;
  }
  const bool sign = a.sign != b.sign;
  if (a.zero || b.zero) return static_cast<uint32_t>(sign) << 31u;
  const uint64_t product =
      static_cast<uint64_t>(a.significand) * b.significand;
  const bool high = (product & (UINT64_C(1) << 47u)) != 0u;
  const uint32_t shift = high ? 21u : 20u;
  const int32_t exponent = a.exponent + b.exponent + (high ? 1 : 0);
  return gaussian_f32_round_pack(
      sign, exponent, gaussian_shift_right_jam_u64(product, shift), valid);
}

static inline uint32_t gaussian_f32_div(uint32_t numerator,
                                         uint32_t denominator,
                                         bool* valid) {
  GaussianF32 a, b;
  if (!gaussian_f32_decode(numerator, &a) ||
      !gaussian_f32_decode(denominator, &b) || b.zero) {
    *valid = false;
    return 0u;
  }
  const bool sign = a.sign != b.sign;
  if (a.zero) return static_cast<uint32_t>(sign) << 31u;
  int32_t exponent = a.exponent - b.exponent;
  const bool less_than_one = a.significand < b.significand;
  const uint32_t shift = less_than_one ? 27u : 26u;
  if (less_than_one) --exponent;
  const uint64_t scaled = static_cast<uint64_t>(a.significand) << shift;
  uint64_t quotient = scaled / b.significand;
  if (scaled % b.significand != 0u) quotient |= 1u;
  return gaussian_f32_round_pack(sign, exponent, quotient, valid);
}

static inline bool gaussian_f32_is_finite(uint32_t word) {
  return ((word >> 23u) & 0xffu) != 0xffu;
}

static inline bool gaussian_f32_is_zero(uint32_t word) {
  return (word & 0x7fffffffu) == 0u;
}

