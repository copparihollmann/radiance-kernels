#pragma once

#include <stdint.h>

#include "softfloat_f32.h"

enum GaussianCheckerFailure : uint32_t {
  GAUSSIAN_CHECKER_OK = 0u,
  GAUSSIAN_CHECKER_BAD_ARGUMENT = 1u,
  GAUSSIAN_CHECKER_NONFINITE_INPUT = 2u,
  GAUSSIAN_CHECKER_ZERO_PIVOT = 3u,
  GAUSSIAN_CHECKER_SOFTFLOAT_FAILURE = 4u,
  GAUSSIAN_CHECKER_RESIDUAL_RANGE = 5u,
  GAUSSIAN_CHECKER_ZERO_DENOMINATOR_LOWER_BOUND = 6u,
  GAUSSIAN_CHECKER_RESIDUAL_TOO_LARGE = 7u,
};

struct GaussianBackSubstitution {
  bool valid;
  uint32_t failure;
  uint32_t row;
  uint32_t column;
  uint64_t solution_digest;
};

struct GaussianResidualCertificate {
  bool valid;
  bool certified;
  uint32_t failure;
  uint32_t row;
  uint64_t residual_upper_coefficient;
  int32_t residual_upper_exponent;
  uint32_t denominator_lower_word;
  uint32_t tolerance_shift;
};

static inline uint64_t gaussian_checker_fnv_word(uint64_t digest,
                                                  uint32_t word) {
  return (digest ^ word) * UINT64_C(0x100000001B3);
}

static inline uint32_t gaussian_floor_log2_u64(uint64_t value) {
  uint32_t result = 0u;
  while (value > 1u) {
    value >>= 1u;
    ++result;
  }
  return result;
}

static inline GaussianBackSubstitution gaussian_back_substitute_actual(
    const volatile uint32_t* transformed_a,
    const volatile uint32_t* transformed_b,
    uint32_t size, uint32_t* solution) {
  GaussianBackSubstitution result = {
      false, GAUSSIAN_CHECKER_BAD_ARGUMENT, 0u, 0u,
      UINT64_C(0xCBF29CE484222325)};
  if (transformed_a == nullptr || transformed_b == nullptr ||
      solution == nullptr || size == 0u) return result;

  for (uint32_t descending = size; descending != 0u; --descending) {
    const uint32_t row = descending - 1u;
    uint32_t value = transformed_b[row];
    if (!gaussian_f32_is_finite(value)) {
      result.failure = GAUSSIAN_CHECKER_NONFINITE_INPUT;
      result.row = row;
      return result;
    }
    for (uint32_t column = row + 1u; column < size; ++column) {
      const uint32_t coefficient = transformed_a[row * size + column];
      if (!gaussian_f32_is_finite(coefficient)) {
        result.failure = GAUSSIAN_CHECKER_NONFINITE_INPUT;
        result.row = row;
        result.column = column;
        return result;
      }
      bool valid = true;
      const uint32_t product = gaussian_f32_mul(
          coefficient, solution[column], &valid);
      value = gaussian_f32_sub(value, product, &valid);
      if (!valid || !gaussian_f32_is_finite(value)) {
        result.failure = GAUSSIAN_CHECKER_SOFTFLOAT_FAILURE;
        result.row = row;
        result.column = column;
        return result;
      }
    }
    const uint32_t diagonal = transformed_a[row * size + row];
    if (!gaussian_f32_is_finite(diagonal)) {
      result.failure = GAUSSIAN_CHECKER_NONFINITE_INPUT;
      result.row = row;
      result.column = row;
      return result;
    }
    if (gaussian_f32_is_zero(diagonal)) {
      result.failure = GAUSSIAN_CHECKER_ZERO_PIVOT;
      result.row = row;
      result.column = row;
      return result;
    }
    bool valid = true;
    solution[row] = gaussian_f32_div(value, diagonal, &valid);
    if (!valid || !gaussian_f32_is_finite(solution[row])) {
      result.failure = GAUSSIAN_CHECKER_SOFTFLOAT_FAILURE;
      result.row = row;
      result.column = row;
      return result;
    }
  }
  for (uint32_t row = 0u; row < size; ++row)
    result.solution_digest = gaussian_checker_fnv_word(
        result.solution_digest, solution[row]);
  result.valid = true;
  result.failure = GAUSSIAN_CHECKER_OK;
  result.row = size;
  result.column = size;
  return result;
}

struct GaussianExactTerm {
  bool negative;
  uint64_t coefficient;
  int32_t exponent;
};

// Decode a finite binary32 word exactly as coefficient * 2^exponent.  Unlike
// GaussianF32, this representation is intentionally not normalized.
static inline bool gaussian_exact_term(uint32_t word, bool invert_sign,
                                        GaussianExactTerm* term) {
  const uint32_t encoded_exponent = (word >> 23u) & 0xffu;
  if (encoded_exponent == 0xffu) return false;
  term->negative = ((word >> 31u) != 0u) != invert_sign;
  if (encoded_exponent == 0u) {
    term->coefficient = word & 0x007fffffu;
    term->exponent = -149;
  } else {
    term->coefficient = 0x00800000u | (word & 0x007fffffu);
    term->exponent = static_cast<int32_t>(encoded_exponent) - 150;
  }
  return true;
}

static inline bool gaussian_exact_product(uint32_t left, uint32_t right,
                                           GaussianExactTerm* term) {
  GaussianExactTerm a, b;
  if (!gaussian_exact_term(left, false, &a) ||
      !gaussian_exact_term(right, false, &b)) return false;
  term->negative = a.negative != b.negative;
  term->coefficient = a.coefficient * b.coefficient;
  term->exponent = a.exponent + b.exponent;
  return true;
}

static inline bool gaussian_scale_term(const GaussianExactTerm& term,
                                        int32_t unit_exponent,
                                        uint64_t* magnitude,
                                        bool* discarded) {
  *discarded = false;
  if (term.coefficient == 0u) {
    *magnitude = 0u;
    return true;
  }
  const int32_t shift = term.exponent - unit_exponent;
  if (shift >= 0) {
    if (shift >= 64 ||
        gaussian_floor_log2_u64(term.coefficient) +
            static_cast<uint32_t>(shift) >= 63u) return false;
    *magnitude = term.coefficient << static_cast<uint32_t>(shift);
    return true;
  }
  const uint32_t right = static_cast<uint32_t>(-shift);
  if (right >= 64u) {
    *magnitude = 0u;
    *discarded = true;
    return true;
  }
  *magnitude = term.coefficient >> right;
  const uint64_t mask = (UINT64_C(1) << right) - 1u;
  *discarded = (term.coefficient & mask) != 0u;
  return true;
}

static inline bool gaussian_scaled_unsigned_leq(
    uint64_t left_coefficient, int32_t left_exponent,
    uint64_t right_coefficient, int32_t right_exponent) {
  if (left_coefficient == 0u) return true;
  if (right_coefficient == 0u) return false;
  const uint32_t left_log = gaussian_floor_log2_u64(left_coefficient);
  const uint32_t right_log = gaussian_floor_log2_u64(right_coefficient);
  const int32_t left_top = left_exponent + static_cast<int32_t>(left_log);
  const int32_t right_top = right_exponent + static_cast<int32_t>(right_log);
  if (left_top != right_top) return left_top < right_top;
  if (left_exponent == right_exponent)
    return left_coefficient <= right_coefficient;
  if (left_exponent < right_exponent) {
    const uint32_t shift = static_cast<uint32_t>(
        right_exponent - left_exponent);
    if (shift >= 64u || right_coefficient > (UINT64_MAX >> shift))
      return false;
    return left_coefficient <= (right_coefficient << shift);
  }
  const uint32_t shift = static_cast<uint32_t>(
      left_exponent - right_exponent);
  if (shift >= 64u || left_coefficient > (UINT64_MAX >> shift))
    return false;
  return (left_coefficient << shift) <= right_coefficient;
}

// Certify the actual solution against the original system.  Each product in
// A*x-b is represented exactly.  Terms are then accumulated in a signed
// fixed-point bin with 54 magnitude bits; every discarded tail contributes a
// full unit to an outward error bound.  Thus
//
//   residual_upper_coefficient * 2^residual_upper_exponent
//
// rigorously bounds ||A*x-b||inf.  For a fail-closed denominator we use
// ||b||inf <= ||A||inf*||x||inf + ||b||inf.  Proving the bound against
// 2^-tolerance_shift * ||b||inf is therefore stricter than proving the
// documented normwise backward-error criterion.
static inline GaussianResidualCertificate gaussian_certify_actual_solution(
    const volatile uint32_t* original_a,
    const volatile uint32_t* original_b,
    const uint32_t* solution, uint32_t size, uint32_t tolerance_shift) {
  GaussianResidualCertificate result = {
      false, false, GAUSSIAN_CHECKER_BAD_ARGUMENT, 0u,
      0u, 0, 0u, tolerance_shift};
  if (original_a == nullptr || original_b == nullptr || solution == nullptr ||
      size == 0u) return result;

  uint32_t denominator_lower_word = 0u;
  for (uint32_t row = 0u; row < size; ++row) {
    const uint32_t word = original_b[row];
    if (!gaussian_f32_is_finite(word)) {
      result.failure = GAUSSIAN_CHECKER_NONFINITE_INPUT;
      result.row = row;
      return result;
    }
    const uint32_t magnitude = word & 0x7fffffffu;
    if (magnitude > denominator_lower_word)
      denominator_lower_word = magnitude;
  }
  if (denominator_lower_word == 0u) {
    result.failure = GAUSSIAN_CHECKER_ZERO_DENOMINATOR_LOWER_BOUND;
    return result;
  }
  result.denominator_lower_word = denominator_lower_word;

  uint64_t maximum_upper = 0u;
  int32_t maximum_exponent = 0;
  bool have_maximum = false;
  for (uint32_t row = 0u; row < size; ++row) {
    int32_t top = -100000;
    for (uint32_t column = 0u; column < size; ++column) {
      GaussianExactTerm term;
      if (!gaussian_exact_product(original_a[row * size + column],
                                  solution[column], &term)) {
        result.failure = GAUSSIAN_CHECKER_NONFINITE_INPUT;
        result.row = row;
        return result;
      }
      if (term.coefficient != 0u) {
        const int32_t term_top = term.exponent + static_cast<int32_t>(
            gaussian_floor_log2_u64(term.coefficient));
        if (term_top > top) top = term_top;
      }
    }
    GaussianExactTerm b_term;
    if (!gaussian_exact_term(original_b[row], true, &b_term)) {
      result.failure = GAUSSIAN_CHECKER_NONFINITE_INPUT;
      result.row = row;
      return result;
    }
    if (b_term.coefficient != 0u) {
      const int32_t term_top = b_term.exponent + static_cast<int32_t>(
          gaussian_floor_log2_u64(b_term.coefficient));
      if (term_top > top) top = term_top;
    }
    if (top == -100000) continue;

    const int32_t unit_exponent = top - 54;
    int64_t accumulator = 0;
    uint64_t discarded_units = 0u;
    for (uint32_t column = 0u; column <= size; ++column) {
      GaussianExactTerm term;
      const bool decoded = column == size
          ? gaussian_exact_term(original_b[row], true, &term)
          : gaussian_exact_product(original_a[row * size + column],
                                   solution[column], &term);
      uint64_t magnitude = 0u;
      bool discarded = false;
      if (!decoded || !gaussian_scale_term(
              term, unit_exponent, &magnitude, &discarded) ||
          magnitude > static_cast<uint64_t>(INT64_MAX)) {
        result.failure = GAUSSIAN_CHECKER_RESIDUAL_RANGE;
        result.row = row;
        return result;
      }
      if (discarded) ++discarded_units;
      const int64_t signed_magnitude = static_cast<int64_t>(magnitude);
      if (term.negative) {
        if (accumulator < INT64_MIN + signed_magnitude) {
          result.failure = GAUSSIAN_CHECKER_RESIDUAL_RANGE;
          result.row = row;
          return result;
        }
        accumulator -= signed_magnitude;
      } else {
        if (accumulator > INT64_MAX - signed_magnitude) {
          result.failure = GAUSSIAN_CHECKER_RESIDUAL_RANGE;
          result.row = row;
          return result;
        }
        accumulator += signed_magnitude;
      }
    }
    const uint64_t absolute = accumulator < 0
        ? static_cast<uint64_t>(-(accumulator + 1)) + 1u
        : static_cast<uint64_t>(accumulator);
    if (absolute > UINT64_MAX - discarded_units) {
      result.failure = GAUSSIAN_CHECKER_RESIDUAL_RANGE;
      result.row = row;
      return result;
    }
    const uint64_t upper = absolute + discarded_units;
    if (!have_maximum || !gaussian_scaled_unsigned_leq(
            upper, unit_exponent, maximum_upper, maximum_exponent)) {
      maximum_upper = upper;
      maximum_exponent = unit_exponent;
      result.row = row;
      have_maximum = true;
    }
  }

  GaussianExactTerm denominator;
  if (!gaussian_exact_term(denominator_lower_word, false, &denominator)) {
    result.failure = GAUSSIAN_CHECKER_NONFINITE_INPUT;
    return result;
  }
  result.residual_upper_coefficient = maximum_upper;
  result.residual_upper_exponent = maximum_exponent;
  result.valid = true;
  result.certified = gaussian_scaled_unsigned_leq(
      maximum_upper,
      maximum_exponent + static_cast<int32_t>(tolerance_shift),
      denominator.coefficient, denominator.exponent);
  result.failure = result.certified
      ? GAUSSIAN_CHECKER_OK : GAUSSIAN_CHECKER_RESIDUAL_TOO_LARGE;
  return result;
}

