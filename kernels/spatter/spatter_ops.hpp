#pragma once

#include <stdint.h>

// The Spatter families differ only in how they compose these address maps.
// Keep the actual 64-bit transfer in one place so every variant makes the
// same two 32-bit reads and two 32-bit writes on Muon.
namespace spatter {

static inline uint32_t pattern_index(const __global uint32_t* pattern,
                                     uint32_t j) {
  return pattern[j];
}

static inline uint32_t nested_pattern_index(const __global uint32_t* pattern,
                                            const __global uint32_t* inner,
                                            uint32_t j) {
  return pattern[inner[j]];
}

static inline uint32_t affine_index(uint32_t base, uint32_t delta,
                                    uint32_t iteration) {
  return base + delta * iteration;
}

static inline uint32_t dense_index(uint32_t j, uint32_t length,
                                   uint32_t iteration, uint32_t wrap) {
  return j + length * (iteration % wrap);
}

static inline void copy_value(volatile __global uint32_t* dst,
                              uint32_t dst_index,
                              const volatile __global uint32_t* src,
                              uint32_t src_index) {
  const uint32_t src_word = src_index * 2u;
  const uint32_t dst_word = dst_index * 2u;
  const uint32_t low = src[src_word];
  const uint32_t high = src[src_word + 1u];
  dst[dst_word] = low;
  dst[dst_word + 1u] = high;
}

}  // namespace spatter
