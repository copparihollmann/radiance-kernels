#include "bigbird_blocksparse_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint32_t rng_state = 0x91E10DA5U;

float deterministic_value(float scale, float offset = 0.0F) {
    rng_state = rng_state * 1664525U + 1013904223U;
    const float unit = static_cast<float>((rng_state >> 8U) & 0x00FFFFFFU) /
                       static_cast<float>(0x01000000U);
    return offset + scale * (2.0F * unit - 1.0F);
}

std::vector<float> values(std::size_t count, float scale) {
    std::vector<float> result(count);
    for (float &value : result)
        value = deterministic_value(scale);
    return result;
}

std::vector<float> sparse_attention_independent(
    const bigbird_reference::Graph &graph, const std::vector<float> &query,
    const std::vector<float> &key, const std::vector<float> &value) {
    const auto &dims = graph.dims;
    const std::size_t blocks = dims.blocks();
    const double scale = 1.0 / std::sqrt(static_cast<double>(dims.head_dim));
    std::vector<float> output(dims.sequence * dims.head_dim, 0.0F);
    for (std::size_t q = 0; q < dims.sequence; ++q) {
        const std::size_t qb = q / dims.block_size;
        std::vector<std::size_t> selected;
        if (qb == 0 || qb + 1 == blocks) {
            for (std::size_t block = 0; block < blocks; ++block)
                selected.push_back(block);
        } else {
            selected.push_back(0);
            for (std::size_t block = qb - 1; block <= qb + 1; ++block)
                if (block != 0 && block + 1 != blocks)
                    selected.push_back(block);
            for (std::size_t slot = 0; slot < dims.random_blocks; ++slot)
                selected.push_back(graph.random_block(qb, slot));
            selected.push_back(blocks - 1);
        }
        double maximum = -1.0e300;
        std::vector<double> logits;
        logits.reserve(selected.size() * dims.block_size);
        for (std::size_t block : selected) {
            for (std::size_t in_block = 0; in_block < dims.block_size;
                 ++in_block) {
                const std::size_t kpos = block * dims.block_size + in_block;
                double dot = 0.0;
                for (std::size_t d = 0; d < dims.head_dim; ++d)
                    dot += static_cast<double>(query[q * dims.head_dim + d]) *
                           static_cast<double>(key[kpos * dims.head_dim + d]);
                logits.push_back(dot * scale);
                maximum = std::max(maximum, logits.back());
            }
        }
        double denominator = 0.0;
        for (double logit : logits)
            denominator += std::exp(logit - maximum);
        for (std::size_t d = 0; d < dims.head_dim; ++d) {
            double sum = 0.0;
            std::size_t index = 0;
            for (std::size_t block : selected) {
                for (std::size_t in_block = 0; in_block < dims.block_size;
                     ++in_block) {
                    const std::size_t kpos =
                        block * dims.block_size + in_block;
                    sum += std::exp(logits[index++] - maximum) *
                           static_cast<double>(value[kpos * dims.head_dim + d]);
                }
            }
            output[q * dims.head_dim + d] =
                static_cast<float>(sum / denominator);
        }
    }
    return output;
}

void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

void mutation_control(const bigbird_reference::Graph &graph,
                      const std::vector<float> &query,
                      const std::vector<float> &key,
                      const std::vector<float> &value) {
    const auto &dims = graph.dims;
    const std::size_t blocks = dims.blocks();
    const std::size_t qb = blocks / 2;
    std::vector<std::uint8_t> baseline_mask =
        bigbird_reference::build_dense_block_mask(graph);
    const std::vector<float> baseline = bigbird_reference::dense_masked_attention(
        dims, baseline_mask, query, key, value);
    const std::size_t probes[] = {0, qb, graph.random_block(qb, 0)};
    const char *names[] = {"global", "local", "random"};
    for (std::size_t family = 0; family < 3; ++family) {
        std::vector<std::uint8_t> mutated = baseline_mask;
        mutated[qb * blocks + probes[family]] = 0;
        const std::vector<float> changed =
            bigbird_reference::dense_masked_attention(dims, mutated, query, key,
                                                      value);
        float largest = 0.0F;
        const std::size_t begin = qb * dims.block_size * dims.head_dim;
        const std::size_t end = (qb + 1) * dims.block_size * dims.head_dim;
        for (std::size_t i = begin; i < end; ++i)
            largest = std::max(largest, std::fabs(baseline[i] - changed[i]));
        require(largest > 1.0e-5F,
                std::string(names[family]) +
                    " edge-family mutation did not affect the result");
    }
}

} // namespace

int main() {
    try {
        require(bigbird_reference::official_base_encoder_parameters() ==
                    127468800ULL,
                "pinned BigBird-Base parameter count changed");
        const bigbird_reference::Dimensions dims{/*sequence=*/1024,
                                                  /*block_size=*/64,
                                                  /*head_dim=*/8,
                                                  /*random_blocks=*/3};
        const bigbird_reference::Graph graph =
            bigbird_reference::make_deterministic_graph(dims);
        const bigbird_reference::Graph graph_again =
            bigbird_reference::make_deterministic_graph(dims);
        require(graph.random == graph_again.random,
                "graph generation is not deterministic");
        const bigbird_reference::StructuralStats stats =
            bigbird_reference::validate_graph(graph);
        require(stats.global_query_rows == 2, "expected two global rows");
        require(stats.random_edges == 42, "expected 14x3 random edges");

        const std::vector<float> query = values(dims.sequence * dims.head_dim,
                                                0.45F);
        const std::vector<float> key = values(dims.sequence * dims.head_dim,
                                              0.45F);
        const std::vector<float> value = values(dims.sequence * dims.head_dim,
                                                0.60F);
        const std::vector<float> dense =
            bigbird_reference::attention(graph, query, key, value);
        const std::vector<float> sparse =
            sparse_attention_independent(graph, query, key, value);
        float maximum_error = 0.0F;
        for (std::size_t i = 0; i < dense.size(); ++i)
            maximum_error =
                std::max(maximum_error, std::fabs(dense[i] - sparse[i]));
        require(maximum_error <= 2.0e-7F,
                "independent sparse traversal differs from dense oracle");
        mutation_control(graph, query, key, value);

        const std::vector<std::uint8_t> mask =
            bigbird_reference::build_dense_block_mask(graph);
        require(bigbird_reference::block_row_bits(mask, dims.blocks(), 0) ==
                    0xFFFFU,
                "first global row is not dense");
        require(bigbird_reference::block_row_bits(mask, dims.blocks(), 15) ==
                    0xFFFFU,
                "last global row is not dense");

        std::cout << "PASS: deterministic BigBird dense-mask oracle\n";
        std::cout << "official_base_encoder_parameters="
                  << bigbird_reference::official_base_encoder_parameters()
                  << "\n";
        std::cout << "sequence=" << dims.sequence
                  << " block_size=" << dims.block_size
                  << " blocks=" << dims.blocks()
                  << " head_dim=" << dims.head_dim
                  << " random_blocks=" << dims.random_blocks << '\n';
        std::cout << "global_rows=" << stats.global_query_rows
                  << " local_rows=" << stats.interior_query_rows
                  << " random_noncontiguous_edges="
                  << stats.noncontiguous_random_edges
                  << " attended_block_edges=" << stats.attended_block_edges
                  << '\n';
        std::cout << "checked_outputs=" << dense.size()
                  << " max_dense_sparse_error=" << maximum_error
                  << " mutation_controls=global,local,random\n";
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
