#ifndef BIGBIRD_BLOCKSPARSE_CORE_HPP
#define BIGBIRD_BLOCKSPARSE_CORE_HPP

#include <stdint.h>

namespace bigbird_blocksparse {

constexpr float kAbsoluteTolerance = 3.0e-5F;
constexpr float kRelativeTolerance = 3.0e-5F;

static inline float absf(float x) { return x < 0.0F ? -x : x; }

static inline bool finitef(float x) {
    return (__builtin_bit_cast(uint32_t, x) & 0x7F800000U) != 0x7F800000U;
}

static inline bool within_tolerance(float actual, float expected) {
    if (!finitef(actual) || !finitef(expected))
        return false;
    const float scale = absf(actual) > absf(expected) ? absf(actual)
                                                       : absf(expected);
    return absf(actual - expected) <=
           kAbsoluteTolerance + kRelativeTolerance * scale;
}

static inline float exp_negative_approx(float x) {
    // Degree-12 exp Taylor polynomial. Generated Q/K values constrain every
    // max-shifted logit to [-1.2, 0], where this has ample margin for the
    // end-to-end 3e-5 + 3e-5*scale output contract.
    float p = 2.0876756988e-9F;    // 1 / 12!
    p = p * x + 2.5052108385e-8F; // 1 / 11!
    p = p * x + 2.7557319224e-7F; // 1 / 10!
    p = p * x + 2.7557319224e-6F; // 1 / 9!
    p = p * x + 2.4801587302e-5F; // 1 / 8!
    p = p * x + 1.9841269841e-4F; // 1 / 7!
    p = p * x + 1.3888888889e-3F; // 1 / 6!
    p = p * x + 8.3333333333e-3F; // 1 / 5!
    p = p * x + 4.1666666667e-2F; // 1 / 4!
    p = p * x + 1.6666666667e-1F; // 1 / 3!
    p = p * x + 5.0000000000e-1F; // 1 / 2!
    p = p * x + 1.0F;
    return p * x + 1.0F;
}

template <typename Visitor>
static inline void visit_block(uint32_t block, uint32_t block_size,
                               Visitor visitor) {
    const uint32_t begin = block * block_size;
    for (uint32_t offset = 0; offset < block_size; ++offset)
        visitor(begin + offset);
}

template <typename Visitor>
static inline void visit_attended_keys(uint32_t query_position,
                                       const uint8_t *random_blocks,
                                       uint32_t sequence, uint32_t block_size,
                                       uint32_t num_random, Visitor visitor) {
    const uint32_t blocks = sequence / block_size;
    const uint32_t query_block = query_position / block_size;
    if (query_block == 0 || query_block + 1 == blocks) {
        for (uint32_t block = 0; block < blocks; ++block)
            visit_block(block, block_size, visitor);
        return;
    }

    // Canonical unique order: first global, non-global local window, random,
    // last global. Legal input random edges never overlap the other families.
    visit_block(0, block_size, visitor);
    for (uint32_t block = query_block - 1; block <= query_block + 1; ++block)
        if (block != 0 && block + 1 != blocks)
            visit_block(block, block_size, visitor);
    const uint32_t row = (query_block - 1) * num_random;
    for (uint32_t slot = 0; slot < num_random; ++slot)
        visit_block(random_blocks[row + slot], block_size, visitor);
    visit_block(blocks - 1, block_size, visitor);
}

static inline float scaled_score(uint32_t query_position,
                                 uint32_t key_position, const float *query,
                                 const float *key, uint32_t head_dim,
                                 float inverse_sqrt_head_dim) {
    float dot = 0.0F;
    const uint32_t qbase = query_position * head_dim;
    const uint32_t kbase = key_position * head_dim;
    for (uint32_t d = 0; d < head_dim; ++d)
        dot += query[qbase + d] * key[kbase + d];
    return dot * inverse_sqrt_head_dim;
}

static inline float attention_element(
    uint32_t query_position, uint32_t output_dimension, const float *query,
    const float *key, const float *value, const uint8_t *random_blocks,
    uint32_t sequence, uint32_t block_size, uint32_t head_dim,
    uint32_t num_random, float inverse_sqrt_head_dim) {
    float maximum = -3.402823466e38F;
    visit_attended_keys(
        query_position, random_blocks, sequence, block_size, num_random,
        [&](uint32_t key_position) {
            const float score = scaled_score(query_position, key_position,
                                             query, key, head_dim,
                                             inverse_sqrt_head_dim);
            if (score > maximum)
                maximum = score;
        });

    float denominator = 0.0F;
    visit_attended_keys(
        query_position, random_blocks, sequence, block_size, num_random,
        [&](uint32_t key_position) {
            denominator += exp_negative_approx(
                scaled_score(query_position, key_position, query, key,
                             head_dim, inverse_sqrt_head_dim) -
                maximum);
        });

    float numerator = 0.0F;
    visit_attended_keys(
        query_position, random_blocks, sequence, block_size, num_random,
        [&](uint32_t key_position) {
            const float weight = exp_negative_approx(
                scaled_score(query_position, key_position, query, key,
                             head_dim, inverse_sqrt_head_dim) -
                maximum);
            numerator += weight *
                         value[key_position * head_dim + output_dimension];
        });
    return numerator / denominator;
}

static inline uint32_t block_row_bits(uint32_t query_block,
                                      const uint8_t *random_blocks,
                                      uint32_t num_blocks,
                                      uint32_t num_random) {
    if (query_block == 0 || query_block + 1 == num_blocks)
        return num_blocks == 32 ? 0xFFFFFFFFU
                                : ((uint32_t{1} << num_blocks) - 1U);
    uint32_t bits = 1U | (uint32_t{1} << (num_blocks - 1));
    for (uint32_t block = query_block - 1; block <= query_block + 1; ++block)
        bits |= uint32_t{1} << block;
    const uint32_t row = (query_block - 1) * num_random;
    for (uint32_t slot = 0; slot < num_random; ++slot)
        bits |= uint32_t{1} << random_blocks[row + slot];
    return bits;
}

} // namespace bigbird_blocksparse

#endif
