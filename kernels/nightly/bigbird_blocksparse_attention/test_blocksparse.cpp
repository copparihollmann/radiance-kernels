#include "../../bigbird_blocksparse_reference/bigbird_blocksparse_reference.hpp"
#include "blocksparse_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint32_t rng_state = 0x91E10DA5U;
std::size_t checked_outputs = 0;
std::size_t checked_masks = 0;
float maximum_absolute_error = 0.0F;
float maximum_tolerance_fraction = 0.0F;

float deterministic_value(float scale) {
    rng_state = rng_state * 1664525U + 1013904223U;
    const float unit = static_cast<float>((rng_state >> 8U) & 0x00FFFFFFU) /
                       static_cast<float>(0x01000000U);
    return scale * (2.0F * unit - 1.0F);
}

std::vector<float> values(std::size_t count, float scale) {
    std::vector<float> result(count);
    for (float &value : result)
        value = deterministic_value(scale);
    return result;
}

void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

void run_case(const bigbird_reference::Dimensions &dims,
              const std::string &label) {
    const bigbird_reference::Graph graph =
        bigbird_reference::make_deterministic_graph(dims);
    const auto stats = bigbird_reference::validate_graph(graph);
    require(stats.noncontiguous_random_edges == stats.random_edges,
            label + ": random coverage failure");
    const std::vector<float> query = values(dims.sequence * dims.head_dim,
                                            0.45F);
    const std::vector<float> key = values(dims.sequence * dims.head_dim, 0.45F);
    const std::vector<float> value = values(dims.sequence * dims.head_dim,
                                            0.60F);
    const std::vector<float> expected =
        bigbird_reference::attention(graph, query, key, value);
    const float inverse_sqrt =
        1.0F / std::sqrt(static_cast<float>(dims.head_dim));
    for (std::size_t q = 0; q < dims.sequence; ++q) {
        for (std::size_t d = 0; d < dims.head_dim; ++d) {
            const float actual = bigbird_blocksparse::attention_element(
                static_cast<std::uint32_t>(q), static_cast<std::uint32_t>(d),
                query.data(), key.data(), value.data(), graph.random.data(),
                static_cast<std::uint32_t>(dims.sequence),
                static_cast<std::uint32_t>(dims.block_size),
                static_cast<std::uint32_t>(dims.head_dim),
                static_cast<std::uint32_t>(dims.random_blocks), inverse_sqrt);
            const float reference = expected[q * dims.head_dim + d];
            const float error = std::fabs(actual - reference);
            const float tolerance = bigbird_blocksparse::kAbsoluteTolerance +
                                    bigbird_blocksparse::kRelativeTolerance *
                                        std::max(std::fabs(actual),
                                                 std::fabs(reference));
            maximum_absolute_error = std::max(maximum_absolute_error, error);
            maximum_tolerance_fraction =
                std::max(maximum_tolerance_fraction, error / tolerance);
            ++checked_outputs;
            require(bigbird_blocksparse::within_tolerance(actual, reference),
                    label + ": numerical mismatch at q=" +
                        std::to_string(q) + " d=" + std::to_string(d));
        }
    }

    const std::vector<std::uint8_t> mask =
        bigbird_reference::build_dense_block_mask(graph);
    for (std::size_t row = 0; row < dims.blocks(); ++row) {
        const std::uint32_t expected_bits =
            bigbird_reference::block_row_bits(mask, dims.blocks(), row);
        const std::uint32_t actual_bits =
            bigbird_blocksparse::block_row_bits(
                static_cast<std::uint32_t>(row), graph.random.data(),
                static_cast<std::uint32_t>(dims.blocks()),
                static_cast<std::uint32_t>(dims.random_blocks));
        require(actual_bits == expected_bits,
                label + ": structural row mismatch at " +
                    std::to_string(row));
        ++checked_masks;
    }
}

void test_fail_closed() {
    require(bigbird_blocksparse::within_tolerance(1.0F, 1.00001F),
            "near value should pass tolerance");
    require(!bigbird_blocksparse::within_tolerance(1.0F, 1.01F),
            "far value should fail tolerance");
    require(!bigbird_blocksparse::within_tolerance(
                std::numeric_limits<float>::infinity(), 0.0F),
            "infinity must fail tolerance");
    require(!bigbird_blocksparse::within_tolerance(
                std::numeric_limits<float>::quiet_NaN(), 0.0F),
            "NaN must fail tolerance");

    const bigbird_reference::Dimensions dims{128, 8, 4, 2};
    bigbird_reference::Graph bad =
        bigbird_reference::make_deterministic_graph(dims);
    bad.random[0] = 1; // query block 1: overlaps its local window
    bool rejected = false;
    try {
        (void)bigbird_reference::validate_graph(bad);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "illegal random/local overlap was not rejected");
}

} // namespace

int main() {
    try {
        test_fail_closed();
        run_case(bigbird_reference::Dimensions{/*sequence=*/128,
                                                /*block_size=*/8,
                                                /*head_dim=*/4,
                                                /*random_blocks=*/2},
                 "small");
        run_case(bigbird_reference::Dimensions{/*sequence=*/1024,
                                                /*block_size=*/64,
                                                /*head_dim=*/8,
                                                /*random_blocks=*/3},
                 "device_shape");
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "PASS: Radiance sparse traversal matches independent CPU "
                 "oracle\n";
    std::cout << "checked_outputs=" << checked_outputs
              << " checked_structural_rows=" << checked_masks << '\n';
    std::cout << "max_absolute_error=" << maximum_absolute_error
              << " max_tolerance_fraction=" << maximum_tolerance_fraction
              << " atol=" << bigbird_blocksparse::kAbsoluteTolerance
              << " rtol=" << bigbird_blocksparse::kRelativeTolerance << '\n';
    return EXIT_SUCCESS;
}
