#include "../../qwen35_deltanet_reference/qwen35_deltanet_reference.hpp"
#include "deltanet_state_core.hpp"

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

using qwen35_deltanet_reference::Dimensions;
using qwen35_deltanet_reference::Parameters;
using qwen35_deltanet_reference::StepTrace;
using qwen35_deltanet_reference::TokenInput;

std::uint32_t rng_state = 0x35D37A11U;
std::size_t checked_values = 0;
float max_absolute_error = 0.0F;
float max_tolerance_fraction = 0.0F;

float random_value(float scale, float offset = 0.0F) {
    rng_state = rng_state * 1664525U + 1013904223U;
    const float unit = static_cast<float>((rng_state >> 8U) & 0x00FFFFFFU) /
                       static_cast<float>(0x01000000U);
    return offset + scale * (2.0F * unit - 1.0F);
}

std::vector<float> values(std::size_t count, float scale,
                          float offset = 0.0F) {
    std::vector<float> result(count);
    for (float &value : result)
        value = random_value(scale, offset);
    return result;
}

void check(float actual, float expected, const std::string &label) {
    ++checked_values;
    const float error = std::fabs(actual - expected);
    const float scale = std::max(std::fabs(actual), std::fabs(expected));
    const float limit = qwen35_deltanet_state::kAbsoluteTolerance +
                        qwen35_deltanet_state::kRelativeTolerance * scale;
    max_absolute_error = std::max(max_absolute_error, error);
    max_tolerance_fraction =
        std::max(max_tolerance_fraction, error / limit);
    if (!qwen35_deltanet_state::within_tolerance(actual, expected)) {
        throw std::runtime_error(label + ": actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected) +
                                 " error=" + std::to_string(error));
    }
}

void test_exp_contract() {
    for (int i = 0; i <= 800; ++i) {
        const float x = -0.001F * static_cast<float>(i);
        check(qwen35_deltanet_state::exp_approx(x), std::exp(x),
              "exp[" + std::to_string(i) + "]");
    }
}

void test_fail_closed() {
    if (!qwen35_deltanet_state::within_tolerance(1.0F, 1.00001F) ||
        qwen35_deltanet_state::within_tolerance(1.0F, 1.001F) ||
        qwen35_deltanet_state::within_tolerance(
            std::numeric_limits<float>::infinity(), 0.0F) ||
        qwen35_deltanet_state::within_tolerance(
            std::numeric_limits<float>::quiet_NaN(), 0.0F)) {
        throw std::runtime_error("tolerance contract did not fail closed");
    }
}

void run_case(std::size_t key_dim, std::size_t value_dim,
              std::size_t sequence, const std::string &label) {
    const Dimensions dims{/*heads=*/1, key_dim, value_dim};
    const Parameters parameters{{-0.75F}, {-0.80F}};
    std::vector<TokenInput> tokens(sequence);
    for (TokenInput &token : tokens) {
        token.q = values(key_dim, 0.75F);
        token.k = values(key_dim, 0.75F);
        token.v = values(value_dim, 0.50F);
        token.a = values(1, 0.35F);
        token.b = values(1, 0.80F);
    }
    const std::vector<StepTrace> oracle =
        qwen35_deltanet_reference::run_full_sequence(dims, parameters, tokens);
    std::vector<float> state(value_dim * key_dim, 0.0F);
    const float query_scale = 1.0F / std::sqrt(static_cast<float>(key_dim));

    for (std::size_t time = 0; time < sequence; ++time) {
        const float q_inv = qwen35_deltanet_state::inverse_l2_norm(
            tokens[time].q.data(), static_cast<std::uint32_t>(key_dim));
        const float k_inv = qwen35_deltanet_state::inverse_l2_norm(
            tokens[time].k.data(), static_cast<std::uint32_t>(key_dim));
        if (oracle[time].log_decay[0] < -0.8F ||
            oracle[time].log_decay[0] > 0.0F)
            throw std::runtime_error(label + ": log-decay outside contract");
        for (std::size_t value = 0; value < value_dim; ++value) {
            const float output = qwen35_deltanet_state::step_column(
                tokens[time].q.data(), tokens[time].k.data(),
                tokens[time].v[value], oracle[time].beta[0],
                oracle[time].log_decay[0], q_inv, k_inv, query_scale,
                state.data() + value * key_dim,
                static_cast<std::uint32_t>(key_dim));
            check(output, oracle[time].output[value],
                  label + ".output[t=" + std::to_string(time) +
                      ",v=" + std::to_string(value) + "]");
        }
        // Check every recurrent-state element at every token, transposing the
        // device [value][key] layout against the oracle [key][value] layout.
        for (std::size_t value = 0; value < value_dim; ++value)
            for (std::size_t key = 0; key < key_dim; ++key)
                check(state[value * key_dim + key],
                      oracle[time].state[key * value_dim + value],
                      label + ".state[t=" + std::to_string(time) +
                          ",v=" + std::to_string(value) +
                          ",k=" + std::to_string(key) + "]");
    }
}

} // namespace

int main() {
    try {
        test_exp_contract();
        test_fail_closed();
        run_case(/*key_dim=*/5, /*value_dim=*/7, /*sequence=*/11, "small");
        run_case(/*key_dim=*/128, /*value_dim=*/128, /*sequence=*/9,
                 "device_shape");
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "PASS: bounded Qwen3.5 DeltaNet state core matches oracle\n";
    std::cout << "checked_values=" << checked_values << '\n';
    std::cout << "max_absolute_error=" << max_absolute_error << '\n';
    std::cout << "max_tolerance_fraction=" << max_tolerance_fraction << '\n';
    std::cout << "absolute_tolerance="
              << qwen35_deltanet_state::kAbsoluteTolerance
              << " relative_tolerance="
              << qwen35_deltanet_state::kRelativeTolerance << '\n';
    return EXIT_SUCCESS;
}
