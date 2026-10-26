#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "parallel_scatter_winner.hpp"

namespace {

constexpr uint32_t kTransfers = 6664304u;
constexpr uint32_t kOutputs = 2051101u;
constexpr uint32_t kExpectedDestinations = 1191878u;

template <typename T>
std::vector<T> read_little_endian(const std::string& path, size_t expected) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream.good()) throw std::runtime_error("cannot open " + path);
  const std::streamsize bytes = stream.tellg();
  if (bytes != static_cast<std::streamsize>(expected * sizeof(T)))
    throw std::runtime_error("unexpected byte count in " + path);
  stream.seekg(0);
  std::vector<T> values(expected);
  stream.read(reinterpret_cast<char*>(values.data()), bytes);
  if (!stream.good()) throw std::runtime_error("short read from " + path);
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
  if constexpr (sizeof(T) == 4) {
    for (T& value : values) value = __builtin_bswap32(value);
  } else if constexpr (sizeof(T) == 8) {
    for (T& value : values) value = __builtin_bswap64(value);
  }
#endif
  return values;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("usage: check_parallel_scatter_dump PATTERN OUTPUT");
    const std::vector<uint32_t> pattern =
        read_little_endian<uint32_t>(argv[1], kTransfers);
    const std::vector<uint64_t> output =
        read_little_endian<uint64_t>(argv[2], kOutputs);
    std::vector<uint8_t> intended(kOutputs, 0u);
    for (uint32_t destination : pattern) {
      if (destination >= kOutputs)
        throw std::runtime_error("pattern destination is out of range");
      intended[destination] = 1u;
    }
    const uint32_t intended_count = static_cast<uint32_t>(
        std::count(intended.begin(), intended.end(), uint8_t{1u}));
    if (intended_count != kExpectedDestinations)
      throw std::runtime_error("written-destination count changed");

    uint32_t admissible = 0u;
    uint32_t missing = 0u;
    uint32_t invalid_source = 0u;
    uint32_t wrong_destination = 0u;
    uint32_t unexpected_nonzero = 0u;
    using Status = spatter::ParallelScatterWinnerStatus;
    for (uint32_t destination = 0; destination < kOutputs; ++destination) {
      const uint64_t value = output[destination];
      if (value == 0u) {
        missing += intended[destination] != 0u;
        continue;
      }
      unexpected_nonzero += intended[destination] == 0u;
      const auto winner = spatter::classify_parallel_scatter_winner(
          value, destination, pattern.data(), kTransfers, 1u, 1u, 8u, 2u);
      switch (winner.status) {
        case Status::kAdmissible: ++admissible; break;
        case Status::kMissing: ++missing; break;
        case Status::kInvalidSource: ++invalid_source; break;
        case Status::kWrongDestination: ++wrong_destination; break;
      }
    }
    const uint64_t illegal = static_cast<uint64_t>(invalid_source) +
                             wrong_destination + unexpected_nonzero;
    const bool passed = admissible == kExpectedDestinations && missing == 0u &&
                        illegal == 0u;
    std::cout << "XRAGE_EXACT_WINNER_CHECK transfers=" << kTransfers
              << " outputs=" << kOutputs
              << " intended=" << intended_count
              << " admissible=" << admissible
              << " missing=" << missing
              << " invalid_source=" << invalid_source
              << " wrong_destination=" << wrong_destination
              << " unexpected_nonzero=" << unexpected_nonzero
              << " illegal=" << illegal
              << (passed ? " PASS\n" : " FAIL\n");
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 2;
  }
}
