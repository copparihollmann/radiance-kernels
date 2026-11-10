#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "parallel_scatter_winner.hpp"

namespace {

constexpr uint64_t kMask = std::numeric_limits<uint64_t>::max();

uint64_t payload(uint32_t tag, uint64_t index) {
  uint64_t x = index + static_cast<uint64_t>(tag) * 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30u)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27u)) * 0x94d049bb133111ebULL;
  return (x ^ (x >> 31u)) & kMask;
}

void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

spatter::ParallelScatterWinner classify(
    uint64_t value, uint32_t destination, const std::vector<uint32_t>& pattern,
    uint32_t count = 1u, uint32_t wrap = 1u, uint32_t delta = 8u,
    uint32_t tag = 2u) {
  return spatter::classify_parallel_scatter_winner(
      value, destination, pattern.data(), pattern.size(), count, wrap, delta,
      tag);
}

void focused_tests() {
  using Status = spatter::ParallelScatterWinnerStatus;
  const std::vector<uint32_t> pattern{0u, 0u, 1u, 0u};
  const uint64_t a = payload(2u, 0u);
  const uint64_t b = payload(2u, 1u);
  const uint64_t c = payload(2u, 2u);
  const uint64_t d = payload(2u, 3u);

  require(classify(a, 0u, pattern).status == Status::kAdmissible,
          "first colliding winner rejected");
  require(classify(b, 0u, pattern).status == Status::kAdmissible,
          "second colliding winner rejected");
  require(classify(d, 0u, pattern).status == Status::kAdmissible,
          "last colliding winner rejected");
  require(classify(c, 1u, pattern).status == Status::kAdmissible,
          "non-colliding winner rejected");
  require(classify(0u, 0u, pattern).status == Status::kMissing,
          "zero did not report a missing destination");
  require(classify(c, 0u, pattern).status == Status::kWrongDestination,
          "complete value from the wrong destination was accepted");
  require(classify(payload(2u, pattern.size()), 0u, pattern).status ==
              Status::kInvalidSource,
          "out-of-range generated payload was accepted");

  const uint64_t torn_ab = (a & 0xffffffffULL) | (b & 0xffffffff00000000ULL);
  const uint64_t torn_ba = (b & 0xffffffffULL) | (a & 0xffffffff00000000ULL);
  require(torn_ab != a && torn_ab != b && torn_ab != d,
          "focused tear accidentally equals a legitimate winner");
  require(torn_ba != a && torn_ba != b && torn_ba != d,
          "reverse focused tear accidentally equals a legitimate winner");
  require(classify(torn_ab, 0u, pattern).status != Status::kAdmissible,
          "low(a)/high(b) torn word was accepted");
  require(classify(torn_ba, 0u, pattern).status != Status::kAdmissible,
          "low(b)/high(a) torn word was accepted");

  // Exercise nontrivial count/wrap/delta inversion as used by the general
  // basic-Scatter mapping, while retaining every original transfer.
  const std::vector<uint32_t> repeated{3u, 1u};
  for (uint32_t i = 0; i < 5u; ++i) {
    for (uint32_t j = 0; j < repeated.size(); ++j) {
      const uint32_t source = j + repeated.size() * (i % 2u);
      const uint32_t destination = repeated[j] + 7u * i;
      require(classify(payload(2u, source), destination, repeated, 5u, 2u,
                       7u).status == Status::kAdmissible,
              "count/wrap/delta transfer was rejected");
    }
  }
  std::cout << "focused admissible-winner tests: PASS\n";
}

std::vector<uint32_t> read_u32le(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  require(stream.good(), "cannot open pinned xRAGE pattern: " + path);
  const std::streamsize bytes = stream.tellg();
  require(bytes > 0 && bytes % 4 == 0, "pattern is empty or not u32 aligned");
  stream.seekg(0);
  std::vector<uint32_t> values(static_cast<size_t>(bytes / 4));
  stream.read(reinterpret_cast<char*>(values.data()), bytes);
  require(stream.good(), "short read from pinned xRAGE pattern");
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
  for (uint32_t& value : values)
    value = __builtin_bswap32(value);
#endif
  return values;
}

void full_xrage_test(const std::string& path) {
  using Status = spatter::ParallelScatterWinnerStatus;
  const std::vector<uint32_t> pattern = read_u32le(path);
  require(pattern.size() == 6664304u,
          "pinned xRAGE9 transfer count changed");
  const uint32_t maximum = *std::max_element(pattern.begin(), pattern.end());
  require(maximum == 2051100u, "pinned xRAGE9 output extent changed");

  std::vector<uint64_t> output(static_cast<size_t>(maximum) + 1u, 0u);
  std::vector<int32_t> first_source(output.size(), -1);
  uint64_t repeated_transfers = 0u;
  uint32_t first_collision_destination = 0u;
  uint32_t first_collision_a = 0u;
  uint32_t first_collision_b = 0u;
  bool have_collision = false;
  for (uint32_t j = 0; j < pattern.size(); ++j) {
    const uint32_t destination = pattern[j];
    require(classify(payload(2u, j), destination, pattern).status ==
                Status::kAdmissible,
            "full xRAGE legitimate transfer rejected at source " +
                std::to_string(j));
    if (first_source[destination] >= 0) {
      ++repeated_transfers;
      if (!have_collision) {
        have_collision = true;
        first_collision_destination = destination;
        first_collision_a = static_cast<uint32_t>(first_source[destination]);
        first_collision_b = j;
      }
    } else {
      first_source[destination] = static_cast<int32_t>(j);
    }
    // A serial schedule is just one allowed schedule.  It preserves the exact
    // original transfer list and selects a complete legitimate last winner.
    output[destination] = payload(2u, j);
  }

  uint64_t written_destinations = 0u;
  for (uint32_t destination = 0; destination < output.size(); ++destination) {
    if (output[destination] == 0u)
      continue;
    ++written_destinations;
    require(classify(output[destination], destination, pattern).status ==
                Status::kAdmissible,
            "full xRAGE legitimate winner rejected at destination " +
                std::to_string(destination));
  }
  require(written_destinations == 1191878u,
          "pinned xRAGE9 written-destination count changed");
  require(repeated_transfers == 5472426u,
          "pinned xRAGE9 collision count changed");
  require(have_collision, "pinned xRAGE9 unexpectedly has no collisions");

  const uint64_t a = payload(2u, first_collision_a);
  const uint64_t b = payload(2u, first_collision_b);
  const uint64_t torn = (a & 0xffffffffULL) | (b & 0xffffffff00000000ULL);
  if (torn != a && torn != b) {
    require(classify(torn, first_collision_destination, pattern).status !=
                Status::kAdmissible,
            "full-pattern collision tear was accepted");
  }

  std::cout << "full pinned xRAGE9: transfers=" << pattern.size()
            << " written_destinations=" << written_destinations
            << " repeated_transfers=" << repeated_transfers << " PASS\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    focused_tests();
    if (argc == 2)
      full_xrage_test(argv[1]);
    else if (argc != 1)
      throw std::runtime_error("usage: test_parallel_scatter_winner [xrage-scatter.u32le]");
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
