#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kTransfers = 6664304u;
constexpr uint32_t kOutputs = 2051101u;
constexpr uint32_t kWrittenDestinations = 1191878u;
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

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

uint64_t payload(uint64_t index) {
  uint64_t value = index + 2ULL * 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

uint64_t fnv(const std::vector<uint64_t>& values) {
  uint64_t digest = kFnvOffset;
  for (uint64_t value : values) {
    digest = (digest ^ static_cast<uint32_t>(value)) * kFnvPrime;
    digest = (digest ^ static_cast<uint32_t>(value >> 32)) * kFnvPrime;
  }
  return digest;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3)
      throw std::runtime_error(
          "usage: check_destination_owner_dump PATTERN OUTPUT");
    const std::vector<uint32_t> pattern =
        read_little_endian<uint32_t>(argv[1], kTransfers);
    const std::vector<uint64_t> observed =
        read_little_endian<uint64_t>(argv[2], kOutputs);
    std::vector<uint64_t> expected(kOutputs, 0u);
    for (uint32_t task = 0; task < kTransfers; ++task) {
      const uint32_t destination = pattern[task];
      if (destination >= kOutputs)
        throw std::runtime_error("pattern destination is out of range");
      // Stable grouping preserves this original task order per destination.
      expected[destination] = payload(task);
    }
    const uint32_t written = static_cast<uint32_t>(std::count_if(
        expected.begin(), expected.end(), [](uint64_t value) { return value != 0; }));
    if (written != kWrittenDestinations)
      throw std::runtime_error("written-destination count changed");

    uint64_t mismatches = 0;
    uint32_t first_mismatch = 0;
    for (uint32_t destination = 0; destination < kOutputs; ++destination) {
      if (observed[destination] != expected[destination]) {
        if (mismatches == 0) first_mismatch = destination;
        ++mismatches;
      }
    }
    const uint64_t expected_digest = fnv(expected);
    const uint64_t observed_digest = fnv(observed);
    const bool passed = mismatches == 0 && observed_digest == expected_digest;
    std::cout << "XRAGE_DESTINATION_OWNER_EXACT transfers=" << kTransfers
              << " outputs=" << kOutputs << " written=" << written
              << " mismatches=" << mismatches;
    if (mismatches != 0) std::cout << " first_mismatch=" << first_mismatch;
    std::cout << " expected_fnv=" << std::hex << std::setw(16)
              << std::setfill('0') << expected_digest
              << " observed_fnv=" << std::setw(16) << observed_digest
              << std::dec << (passed ? " PASS\n" : " FAIL\n");
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 2;
  }
}
