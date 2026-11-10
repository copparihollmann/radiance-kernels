#include "qwen35_deltanet_reference.hpp"

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
using qwen35_deltanet_reference::StreamingReference;
using qwen35_deltanet_reference::TokenInput;

constexpr float kAbsoluteTolerance = 2.0e-5F;
constexpr float kRelativeTolerance = 2.0e-5F;
std::uint32_t rng_state = 0x35D37A11U;
std::size_t checked_values = 0;

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

Parameters make_parameters(const Dimensions &dims) {
    Parameters p;
    p.a_log = values(dims.heads, 0.45F, -0.75F);
    p.dt_bias = values(dims.heads, 0.20F, -0.80F);
    return p;
}

std::vector<TokenInput> make_tokens(const Dimensions &dims,
                                    std::size_t length) {
    std::vector<TokenInput> tokens(length);
    for (TokenInput &token : tokens) {
        token.q = values(dims.heads * dims.key_dim, 0.75F);
        token.k = values(dims.heads * dims.key_dim, 0.75F);
        token.v = values(dims.heads * dims.value_dim, 0.50F);
        token.a = values(dims.heads, 0.35F);
        token.b = values(dims.heads, 0.80F);
    }
    return tokens;
}

void check(float actual, float expected, const std::string &label) {
    ++checked_values;
    const float error = std::fabs(actual - expected);
    const float limit = kAbsoluteTolerance +
                        kRelativeTolerance *
                            std::max(std::fabs(actual), std::fabs(expected));
    if (!std::isfinite(actual) || !std::isfinite(expected) || error > limit) {
        throw std::runtime_error(label + ": actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected) +
                                 " error=" + std::to_string(error));
    }
}

void compare(const StepTrace &actual, const StepTrace &expected,
             const std::string &label) {
    const auto compare_vector = [&](const std::vector<float> &a,
                                    const std::vector<float> &e,
                                    const std::string &field) {
        if (a.size() != e.size())
            throw std::runtime_error(label + "." + field + " size mismatch");
        for (std::size_t i = 0; i < a.size(); ++i)
            check(a[i], e[i], label + "." + field + "[" +
                                      std::to_string(i) + "]");
    };
    compare_vector(actual.beta, expected.beta, "beta");
    compare_vector(actual.log_decay, expected.log_decay, "log_decay");
    compare_vector(actual.output, expected.output, "output");
    compare_vector(actual.state, expected.state, "state");
}

void test_transparent_gate_case() {
    const Dimensions dims{/*heads=*/1, /*key_dim=*/1, /*value_dim=*/1};
    Parameters p{{0.0F}, {0.0F}};
    TokenInput token{{1.0F}, {1.0F}, {2.0F}, {0.0F}, {0.0F}};
    StreamingReference reference(dims, p);
    const StepTrace first = reference.step(token);
    // b=0 => beta=1/2. A_log=0 and softplus(0)=log(2), so g=-log(2)
    // and the decay is exactly 1/2 apart from host transcendental rounding.
    check(first.beta[0], 0.5F, "known.beta");
    check(std::exp(first.log_decay[0]), 0.5F, "known.decay");
    if (!(first.output[0] > 0.99F && first.output[0] < 1.01F))
        throw std::runtime_error("known first update did not write v*beta");
    const StepTrace second = reference.step(token);
    if (!(second.state[0] > first.state[0] && second.state[0] < 2.0F))
        throw std::runtime_error("known second update did not retain gated state");
}

void run_equivalence_case(const Dimensions &dims, std::size_t length,
                          const std::string &label) {
    const Parameters p = make_parameters(dims);
    const std::vector<TokenInput> tokens = make_tokens(dims, length);
    const std::vector<StepTrace> full =
        qwen35_deltanet_reference::run_full_sequence(dims, p, tokens);
    const std::vector<StepTrace> streaming =
        qwen35_deltanet_reference::run_streaming(dims, p, tokens);
    for (std::size_t time = 0; time < length; ++time)
        compare(streaming[time], full[time],
                label + ".time_" + std::to_string(time));

    // Explicit prefill/decode boundary: retain state through all but one token,
    // then perform one cached decode step and compare the entire resulting state.
    StreamingReference split(dims, p);
    for (std::size_t time = 0; time + 1 < length; ++time)
        split.step(tokens[time]);
    const StepTrace decode = split.step(tokens.back());
    compare(decode, full.back(), label + ".cached_decode");
}

void test_shape_validation() {
    const Dimensions dims{/*heads=*/2, /*key_dim=*/3, /*value_dim=*/4};
    Parameters p = make_parameters(dims);
    p.a_log.pop_back();
    bool threw = false;
    try {
        StreamingReference invalid(dims, p);
        (void)invalid;
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    if (!threw)
        throw std::runtime_error("truncated gate parameters were accepted");
}

} // namespace

int main() {
    try {
        test_transparent_gate_case();
        run_equivalence_case(
            Dimensions{/*heads=*/3, /*key_dim=*/5, /*value_dim=*/7}, 11,
            "small");
        run_equivalence_case(Dimensions::official_recurrent(), 3,
                             "official_recurrent_shape");
        test_shape_validation();
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "PASS: Qwen3.5 gated DeltaNet CPU oracle\n";
    std::cout << "checked_values=" << checked_values << '\n';
    std::cout << "official_expanded_state_floats="
              << Dimensions::official_recurrent().heads *
                     Dimensions::official_recurrent().key_dim *
                     Dimensions::official_recurrent().value_dim
              << '\n';
    std::cout << "absolute_tolerance=" << kAbsoluteTolerance
              << " relative_tolerance=" << kRelativeTolerance << '\n';
    return EXIT_SUCCESS;
}
