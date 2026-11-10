#pragma once

#include <stdint.h>

namespace spatter {

// Parallel Scatter deliberately retains the benchmark's racing, non-atomic
// pair of 32-bit stores.  Its final value is therefore schedule-dependent.
// A correct final word must nevertheless be one complete source payload from
// a transfer that targeted this destination.
enum class ParallelScatterWinnerStatus : uint32_t {
  kAdmissible = 0,
  kMissing = 1,
  kInvalidSource = 2,
  kWrongDestination = 3,
};

struct ParallelScatterWinner {
  ParallelScatterWinnerStatus status;
  uint64_t source_index;
};

static inline uint64_t undo_xor_shift_right(uint64_t value, uint32_t shift) {
  uint64_t result = value;
  for (uint32_t amount = shift; amount < 64u; amount += shift)
    result ^= value >> amount;
  return result;
}

// Inverse of run.py's SplitMix64-style payload().  Both odd multipliers are
// invertible modulo 2^64, so this maps every observed word to exactly one
// candidate source index without a probabilistic hash lookup.
static inline uint64_t decode_payload_index(uint64_t value, uint32_t tag) {
  uint64_t decoded = undo_xor_shift_right(value, 31u);
  decoded *= 0x319642b2d24d8ec3ULL;
  decoded = undo_xor_shift_right(decoded, 27u);
  decoded *= 0x96de1b173f119089ULL;
  decoded = undo_xor_shift_right(decoded, 30u);
  return decoded - static_cast<uint64_t>(tag) * 0x9e3779b97f4a7c15ULL;
}

// Check the original basic-Scatter address maps directly.  This does not
// impose an ordering and does not change, serialize, or claim atomicity for
// the device stores.
static inline ParallelScatterWinner classify_parallel_scatter_winner(
    uint64_t value, uint32_t destination,
    const volatile uint32_t* pattern, uint32_t length, uint32_t count,
    uint32_t wrap, uint32_t delta, uint32_t tag,
    uint64_t source_index_base = 0u) {
  if (value == 0u)
    return {ParallelScatterWinnerStatus::kMissing, 0u};

  const uint64_t absolute_source = decode_payload_index(value, tag);
  if (absolute_source < source_index_base || length == 0u || wrap == 0u)
    return {ParallelScatterWinnerStatus::kInvalidSource, absolute_source};
  const uint64_t source = absolute_source - source_index_base;
  const uint64_t source_length =
      static_cast<uint64_t>(length) * static_cast<uint64_t>(wrap);
  if (source >= source_length)
    return {ParallelScatterWinnerStatus::kInvalidSource, absolute_source};

  const uint32_t residue = static_cast<uint32_t>(source / length);
  const uint32_t j = static_cast<uint32_t>(source % length);
  if (residue >= count)
    return {ParallelScatterWinnerStatus::kWrongDestination, absolute_source};

  const uint64_t first_destination = pattern[j];
  if (delta == 0u) {
    return {destination == first_destination
                ? ParallelScatterWinnerStatus::kAdmissible
                : ParallelScatterWinnerStatus::kWrongDestination,
            absolute_source};
  }
  if (destination < first_destination)
    return {ParallelScatterWinnerStatus::kWrongDestination, absolute_source};
  const uint64_t difference = destination - first_destination;
  if (difference % delta != 0u)
    return {ParallelScatterWinnerStatus::kWrongDestination, absolute_source};
  const uint64_t iteration = difference / delta;
  const bool admissible = iteration < count && iteration % wrap == residue;
  return {admissible ? ParallelScatterWinnerStatus::kAdmissible
                     : ParallelScatterWinnerStatus::kWrongDestination,
          absolute_source};
}

}  // namespace spatter
