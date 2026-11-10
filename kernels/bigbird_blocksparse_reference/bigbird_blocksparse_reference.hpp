#ifndef BIGBIRD_BLOCKSPARSE_REFERENCE_HPP
#define BIGBIRD_BLOCKSPARSE_REFERENCE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bigbird_reference {

// Config-derived parameter count for the pinned repository's default
// BERT-style BigBird-Base encoder (including embeddings and pooler, excluding
// task/MLM heads).  The bundled gpt2.model contains 50,358 pieces.
inline constexpr std::uint64_t official_base_encoder_parameters() {
    constexpr std::uint64_t vocab = 50358;
    constexpr std::uint64_t hidden = 768;
    constexpr std::uint64_t intermediate = 3072;
    constexpr std::uint64_t layers = 12;
    constexpr std::uint64_t positions = 4096;
    constexpr std::uint64_t token_types = 2;
    constexpr std::uint64_t embeddings =
        vocab * hidden + positions * hidden + token_types * hidden;
    constexpr std::uint64_t per_layer =
        4 * hidden * hidden + 2 * hidden * intermediate +
        (9 * hidden + intermediate);
    constexpr std::uint64_t encoder_input_norm = 2 * hidden;
    constexpr std::uint64_t pooler = hidden * hidden + hidden;
    return embeddings + layers * per_layer + encoder_input_norm + pooler;
}

struct Dimensions {
    std::size_t sequence;
    std::size_t block_size;
    std::size_t head_dim;
    std::size_t random_blocks;

    std::size_t blocks() const { return sequence / block_size; }
};

struct Graph {
    Dimensions dims;
    // Rows correspond to non-global query blocks 1..blocks-2.
    std::vector<std::uint8_t> random;

    std::uint8_t random_block(std::size_t query_block,
                              std::size_t slot) const {
        return random.at((query_block - 1) * dims.random_blocks + slot);
    }
};

struct StructuralStats {
    std::size_t global_query_rows = 0;
    std::size_t interior_query_rows = 0;
    std::size_t random_edges = 0;
    std::size_t attended_block_edges = 0;
    std::size_t noncontiguous_random_edges = 0;
};

inline void validate_dimensions(const Dimensions &dims) {
    if (dims.sequence == 0 || dims.block_size == 0 || dims.head_dim == 0 ||
        dims.random_blocks == 0)
        throw std::invalid_argument("BigBird dimensions must be nonzero");
    if (dims.sequence % dims.block_size != 0)
        throw std::invalid_argument("sequence must be divisible by block_size");
    if (dims.blocks() < 7)
        throw std::invalid_argument("at least seven blocks are required");
    // Interior candidates exclude two globals and a three-block local window.
    if (dims.random_blocks > dims.blocks() - 5)
        throw std::invalid_argument("too many random blocks for legal graph");
    if (dims.blocks() > 32)
        throw std::invalid_argument("bounded oracle supports at most 32 blocks");
}

inline std::uint32_t next_random(std::uint32_t &state) {
    // Fully specified xorshift32: deterministic across C++ implementations.
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
}

inline Graph make_deterministic_graph(const Dimensions &dims,
                                      std::uint32_t seed = 0xB16B00B5U) {
    validate_dimensions(dims);
    Graph graph{dims, {}};
    graph.random.reserve((dims.blocks() - 2) * dims.random_blocks);
    std::uint32_t state = seed;
    for (std::size_t query = 1; query + 1 < dims.blocks(); ++query) {
        std::vector<std::uint8_t> candidates;
        for (std::size_t key = 1; key + 1 < dims.blocks(); ++key) {
            if (key + 1 < query || key > query + 1)
                candidates.push_back(static_cast<std::uint8_t>(key));
        }
        for (std::size_t i = candidates.size(); i > 1; --i) {
            const std::size_t j = next_random(state) % i;
            std::swap(candidates[i - 1], candidates[j]);
        }
        if (candidates.size() < dims.random_blocks)
            throw std::logic_error("insufficient legal random candidates");
        graph.random.insert(graph.random.end(), candidates.begin(),
                            candidates.begin() + dims.random_blocks);
    }
    return graph;
}

inline std::vector<std::uint8_t> build_dense_block_mask(const Graph &graph) {
    validate_dimensions(graph.dims);
    const std::size_t blocks = graph.dims.blocks();
    if (graph.random.size() !=
        (blocks - 2) * graph.dims.random_blocks)
        throw std::invalid_argument("random adjacency has wrong size");
    std::vector<std::uint8_t> mask(blocks * blocks, 0);
    for (std::size_t query = 0; query < blocks; ++query) {
        if (query == 0 || query + 1 == blocks) {
            std::fill(mask.begin() + query * blocks,
                      mask.begin() + (query + 1) * blocks, 1);
            continue;
        }
        mask[query * blocks] = 1;
        mask[query * blocks + blocks - 1] = 1;
        for (std::size_t key = query - 1; key <= query + 1; ++key)
            mask[query * blocks + key] = 1;
        for (std::size_t slot = 0; slot < graph.dims.random_blocks; ++slot) {
            const std::size_t key = graph.random_block(query, slot);
            if (key >= blocks)
                throw std::invalid_argument("random block is out of range");
            mask[query * blocks + key] = 1;
        }
    }
    return mask;
}

inline StructuralStats validate_graph(const Graph &graph) {
    const std::size_t blocks = graph.dims.blocks();
    const std::vector<std::uint8_t> mask = build_dense_block_mask(graph);
    StructuralStats stats;
    for (std::size_t query = 0; query < blocks; ++query) {
        if (query == 0 || query + 1 == blocks) {
            ++stats.global_query_rows;
        } else {
            ++stats.interior_query_rows;
            std::vector<std::uint8_t> seen(blocks, 0);
            for (std::size_t slot = 0; slot < graph.dims.random_blocks;
                 ++slot) {
                const std::size_t key = graph.random_block(query, slot);
                if (key == 0 || key + 1 == blocks ||
                    (key + 1 >= query && key <= query + 1))
                    throw std::invalid_argument(
                        "random edge overlaps global/local edge");
                if (seen[key])
                    throw std::invalid_argument("duplicate random edge");
                seen[key] = 1;
                ++stats.random_edges;
                if (key + 1 < query || key > query + 1)
                    ++stats.noncontiguous_random_edges;
            }
        }
        for (std::size_t key = 0; key < blocks; ++key)
            stats.attended_block_edges += mask[query * blocks + key] != 0;
    }
    if (stats.global_query_rows != 2 || stats.interior_query_rows != blocks - 2)
        throw std::logic_error("global/interior row accounting failed");
    if (stats.random_edges == 0 ||
        stats.noncontiguous_random_edges != stats.random_edges)
        throw std::logic_error("noncontiguous random edges were not exercised");
    return stats;
}

inline void validate_tensor(const std::vector<float> &tensor,
                            const Dimensions &dims, const char *name) {
    if (tensor.size() != dims.sequence * dims.head_dim)
        throw std::invalid_argument(std::string(name) + " has wrong size");
}

inline std::vector<float>
dense_masked_attention(const Dimensions &dims,
                       const std::vector<std::uint8_t> &block_mask,
                       const std::vector<float> &query,
                       const std::vector<float> &key,
                       const std::vector<float> &value) {
    validate_dimensions(dims);
    validate_tensor(query, dims, "query");
    validate_tensor(key, dims, "key");
    validate_tensor(value, dims, "value");
    const std::size_t blocks = dims.blocks();
    if (block_mask.size() != blocks * blocks)
        throw std::invalid_argument("block mask has wrong size");

    // This deliberately scans every key position and consults a dense mask.
    // The device path instead enumerates sparse blocks; the indexing and
    // normalization implementations are independent.
    const double scale = 1.0 / std::sqrt(static_cast<double>(dims.head_dim));
    std::vector<float> output(dims.sequence * dims.head_dim, 0.0F);
    std::vector<double> logits(dims.sequence,
                               -std::numeric_limits<double>::infinity());
    for (std::size_t q = 0; q < dims.sequence; ++q) {
        const std::size_t query_block = q / dims.block_size;
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < dims.sequence; ++k) {
            const std::size_t key_block = k / dims.block_size;
            if (!block_mask[query_block * blocks + key_block])
                continue;
            double dot = 0.0;
            for (std::size_t d = 0; d < dims.head_dim; ++d)
                dot += static_cast<double>(query[q * dims.head_dim + d]) *
                       static_cast<double>(key[k * dims.head_dim + d]);
            logits[k] = dot * scale;
            maximum = std::max(maximum, logits[k]);
        }
        double denominator = 0.0;
        for (std::size_t k = 0; k < dims.sequence; ++k) {
            const std::size_t key_block = k / dims.block_size;
            if (block_mask[query_block * blocks + key_block])
                denominator += std::exp(logits[k] - maximum);
        }
        if (!(denominator > 0.0) || !std::isfinite(denominator))
            throw std::runtime_error("softmax denominator is invalid");
        for (std::size_t d = 0; d < dims.head_dim; ++d) {
            double sum = 0.0;
            for (std::size_t k = 0; k < dims.sequence; ++k) {
                const std::size_t key_block = k / dims.block_size;
                if (block_mask[query_block * blocks + key_block])
                    sum += std::exp(logits[k] - maximum) *
                           static_cast<double>(value[k * dims.head_dim + d]);
            }
            output[q * dims.head_dim + d] =
                static_cast<float>(sum / denominator);
        }
    }
    return output;
}

inline std::vector<float> attention(const Graph &graph,
                                    const std::vector<float> &query,
                                    const std::vector<float> &key,
                                    const std::vector<float> &value) {
    validate_graph(graph);
    return dense_masked_attention(graph.dims, build_dense_block_mask(graph),
                                  query, key, value);
}

inline std::uint32_t block_row_bits(const std::vector<std::uint8_t> &mask,
                                    std::size_t blocks,
                                    std::size_t query_block) {
    if (blocks > 32 || mask.size() != blocks * blocks || query_block >= blocks)
        throw std::invalid_argument("cannot pack block row");
    std::uint32_t bits = 0;
    for (std::size_t key = 0; key < blocks; ++key)
        if (mask[query_block * blocks + key])
            bits |= std::uint32_t{1} << key;
    return bits;
}

} // namespace bigbird_reference

#endif
