#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <vector>

#include "actual_checker.h"

static uint32_t to_word(float value) {
  uint32_t word;
  memcpy(&word, &value, sizeof(word));
  return word;
}

static float to_float(uint32_t word) {
  float value;
  memcpy(&value, &word, sizeof(value));
  return value;
}

static uint32_t native_operation(uint32_t left, uint32_t right,
                                 uint32_t operation) {
  volatile float a = to_float(left);
  volatile float b = to_float(right);
  volatile float result = 0.0f;
  if (operation == 0u) result = a + b;
  if (operation == 1u) result = a - b;
  if (operation == 2u) result = a * b;
  if (operation == 3u) result = a / b;
  return to_word(result);
}

static uint32_t software_operation(uint32_t left, uint32_t right,
                                   uint32_t operation, bool* valid) {
  if (operation == 0u) return gaussian_f32_add(left, right, valid);
  if (operation == 1u) return gaussian_f32_sub(left, right, valid);
  if (operation == 2u) return gaussian_f32_mul(left, right, valid);
  return gaussian_f32_div(left, right, valid);
}

static bool finite_word(uint32_t word) {
  return (word & 0x7f800000u) != 0x7f800000u;
}

static int check_pair(uint32_t left, uint32_t right, uint32_t operation) {
  const uint32_t expected = native_operation(left, right, operation);
  bool valid = true;
  const uint32_t actual = software_operation(left, right, operation, &valid);
  const bool operands_finite = finite_word(left) && finite_word(right);
  const bool denominator_nonzero = operation != 3u ||
      (right & 0x7fffffffu) != 0u;
  const bool expected_valid = operands_finite && denominator_nonzero &&
      finite_word(expected);
  if (valid != expected_valid || (valid && actual != expected)) {
    fprintf(stderr,
            "softfloat mismatch op=%u left=%08x right=%08x "
            "actual=%08x expected=%08x valid=%u expected_valid=%u\n",
            operation, left, right, actual, expected,
            static_cast<unsigned>(valid),
            static_cast<unsigned>(expected_valid));
    return 1;
  }
  return 0;
}

static uint32_t next_random(uint32_t* state) {
  uint32_t value = *state;
  value ^= value << 13u;
  value ^= value >> 17u;
  value ^= value << 5u;
  *state = value;
  return value;
}

static int softfloat_self_test() {
  if (fesetround(FE_TONEAREST) != 0) return 2;
  const uint32_t edge[] = {
      0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
      0x00000002u, 0x007fffffu, 0x807fffffu, 0x00800000u,
      0x80800000u, 0x00800001u, 0x3effffffu, 0x3f000000u,
      0x3f000001u, 0x3f7fffffu, 0x3f800000u, 0xbf800000u,
      0x3f800001u, 0x40000000u, 0xc0000000u, 0x7f7ffffeu,
      0xff7ffffeu, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u,
      0xff800000u, 0x7fc00000u,
  };
  uint64_t checks = 0u;
  for (uint32_t operation = 0u; operation < 4u; ++operation) {
    for (uint32_t i = 0u; i < sizeof(edge) / sizeof(edge[0]); ++i) {
      for (uint32_t j = 0u; j < sizeof(edge) / sizeof(edge[0]); ++j) {
        if (check_pair(edge[i], edge[j], operation) != 0) return 3;
        ++checks;
      }
    }
  }
  uint32_t state = 0x91e10da5u;
  for (uint32_t i = 0u; i < 250000u; ++i) {
    uint32_t left = next_random(&state);
    uint32_t right = next_random(&state);
    // Most random encodings are finite; explicitly retain nonfinite cases to
    // exercise fail-closed handling as well.
    const uint32_t operation = i & 3u;
    if (check_pair(left, right, operation) != 0) return 4;
    ++checks;
  }
  printf("softfloat_checks=%llu\n", static_cast<unsigned long long>(checks));
  return 0;
}

static bool read_words(const char* path, std::vector<uint32_t>* words) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) return false;
  const std::streamsize bytes = input.tellg();
  if (bytes < 0 || bytes % 4 != 0) return false;
  input.seekg(0);
  words->resize(static_cast<size_t>(bytes / 4));
  return static_cast<bool>(input.read(
      reinterpret_cast<char*>(words->data()), bytes));
}

static int check_files(int argc, char** argv) {
  if (argc != 8) return 10;
  const uint32_t size = static_cast<uint32_t>(strtoul(argv[2], nullptr, 0));
  const uint32_t tolerance_shift = static_cast<uint32_t>(
      strtoul(argv[3], nullptr, 0));
  std::vector<uint32_t> original_a, original_b, transformed_a, transformed_b;
  if (!read_words(argv[4], &original_a) ||
      !read_words(argv[5], &original_b) ||
      !read_words(argv[6], &transformed_a) ||
      !read_words(argv[7], &transformed_b)) return 11;
  if (original_a.size() != static_cast<size_t>(size) * size ||
      original_b.size() != size || transformed_a.size() != original_a.size() ||
      transformed_b.size() != size) return 12;
  std::vector<uint32_t> solution(size);
  const GaussianBackSubstitution back = gaussian_back_substitute_actual(
      transformed_a.data(), transformed_b.data(), size, solution.data());
  const GaussianResidualCertificate certificate = back.valid
      ? gaussian_certify_actual_solution(original_a.data(), original_b.data(),
                                         solution.data(), size, tolerance_shift)
      : GaussianResidualCertificate{};
  printf("back_valid=%u back_failure=%u solution_digest=%016llx "
         "certificate_valid=%u certified=%u certificate_failure=%u "
         "residual_coeff=%llu residual_exp=%d denominator_word=%08x "
         "row=%u tolerance_shift=%u\n",
         static_cast<unsigned>(back.valid), back.failure,
         static_cast<unsigned long long>(back.solution_digest),
         static_cast<unsigned>(certificate.valid),
         static_cast<unsigned>(certificate.certified), certificate.failure,
         static_cast<unsigned long long>(
             certificate.residual_upper_coefficient),
         certificate.residual_upper_exponent,
         certificate.denominator_lower_word, certificate.row,
         certificate.tolerance_shift);
  return back.valid && certificate.valid && certificate.certified ? 0 : 1;
}

static int certify_files(int argc, char** argv) {
  if (argc != 7) return 20;
  const uint32_t size = static_cast<uint32_t>(strtoul(argv[2], nullptr, 0));
  const uint32_t tolerance_shift = static_cast<uint32_t>(
      strtoul(argv[3], nullptr, 0));
  std::vector<uint32_t> original_a, original_b, solution;
  if (!read_words(argv[4], &original_a) ||
      !read_words(argv[5], &original_b) ||
      !read_words(argv[6], &solution)) return 21;
  if (original_a.size() != static_cast<size_t>(size) * size ||
      original_b.size() != size || solution.size() != size) return 22;
  const GaussianResidualCertificate certificate =
      gaussian_certify_actual_solution(
          original_a.data(), original_b.data(), solution.data(), size,
          tolerance_shift);
  printf("certificate_valid=%u certified=%u certificate_failure=%u "
         "residual_coeff=%llu residual_exp=%d denominator_word=%08x "
         "row=%u tolerance_shift=%u\n",
         static_cast<unsigned>(certificate.valid),
         static_cast<unsigned>(certificate.certified), certificate.failure,
         static_cast<unsigned long long>(
             certificate.residual_upper_coefficient),
         certificate.residual_upper_exponent,
         certificate.denominator_lower_word, certificate.row,
         certificate.tolerance_shift);
  return certificate.valid ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc == 2 && strcmp(argv[1], "selftest") == 0)
    return softfloat_self_test();
  if (argc >= 2 && strcmp(argv[1], "check") == 0)
    return check_files(argc, argv);
  if (argc >= 2 && strcmp(argv[1], "certify") == 0)
    return certify_files(argc, argv);
  return 64;
}

