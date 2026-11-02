#include "../../mamba_state_reference/mamba_state_reference.hpp"
#include "selective_state_core.hpp"

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

using mamba_reference::Dimensions;
using mamba_reference::Parameters;
using mamba_reference::StepTrace;
using mamba_reference::TokenInput;

std::uint32_t rng_state = 0x43C2A91DU;
std::size_t values_checked = 0;
float max_absolute_error = 0.0F;
float max_tolerance_fraction = 0.0F;

float deterministic_value(float scale, float offset = 0.0F) {
    rng_state = rng_state * 1664525U + 1013904223U;
    const float unit = static_cast<float>((rng_state >> 8U) & 0x00FFFFFFU) /
                       static_cast<float>(0x01000000U);
    return offset + scale * (2.0F * unit - 1.0F);
}

std::vector<float> values(std::size_t count, float scale, float offset = 0.0F) {
    std::vector<float> result(count);
    for (float &value : result)
        value = deterministic_value(scale, offset);
    return result;
}

Parameters parameters(const Dimensions &dims) {
    Parameters p;
    p.conv_weight = values(dims.d_inner * dims.conv_width, 0.20F);
    p.conv_bias = values(dims.d_inner, 0.05F);
    p.dt_weight = values(dims.d_inner * dims.dt_rank, 0.12F);
    p.dt_bias = values(dims.d_inner, 0.20F, -1.0F);
    p.a_log = values(dims.d_inner * dims.d_state, 0.60F, -0.90F);
    p.d_skip = values(dims.d_inner, 0.30F);
    return p;
}

std::vector<TokenInput> tokens(const Dimensions &dims, std::size_t length) {
    std::vector<TokenInput> result(length);
    for (TokenInput &token : result) {
        token.x = values(dims.d_inner, 0.50F);
        token.dt_features = values(dims.dt_rank, 0.40F);
        token.b = values(dims.d_state, 0.35F);
        token.c = values(dims.d_state, 0.35F);
        token.gate = values(dims.d_inner, 0.80F);
    }
    return result;
}

void check(float actual, float expected, const std::string &label) {
    ++values_checked;
    const float absolute = std::fabs(actual - expected);
    const float scale = std::max(std::fabs(actual), std::fabs(expected));
    const float limit = mamba_selective_state::kAbsoluteTolerance +
                        mamba_selective_state::kRelativeTolerance * scale;
    max_absolute_error = std::max(max_absolute_error, absolute);
    max_tolerance_fraction = std::max(max_tolerance_fraction, absolute / limit);
    if (!mamba_selective_state::within_tolerance(actual, expected)) {
        throw std::runtime_error(label + ": actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected) +
                                 " abs_error=" + std::to_string(absolute));
    }
}

void run_case(const Dimensions &dims, std::size_t length,
              const std::string &label) {
    const Parameters p = parameters(dims);
    const std::vector<TokenInput> input = tokens(dims, length);
    const std::vector<StepTrace> oracle =
        mamba_reference::run_full_sequence(dims, p, input);
    std::vector<float> state(dims.d_inner * dims.d_state, 0.0F);

    // The oracle supplies delta and conv_activated, making the boundary with
    // the deliberately omitted convolution and timestep projection unambiguous.
    for (std::size_t time = 0; time < length; ++time) {
        for (std::size_t channel = 0; channel < dims.d_inner; ++channel) {
            const float actual = mamba_selective_state::step(
                oracle[time].delta[channel],
                oracle[time].conv_activated[channel], input[time].b.data(),
                input[time].c.data(), p.a_log.data() + channel * dims.d_state,
                p.d_skip[channel], input[time].gate[channel],
                state.data() + channel * dims.d_state,
                static_cast<std::uint32_t>(dims.d_state));
            check(actual, oracle[time].output[channel],
                  label + ".output[t=" + std::to_string(time) +
                      ",c=" + std::to_string(channel) + "]");
            for (std::size_t j = 0; j < dims.d_state; ++j) {
                const std::size_t index = channel * dims.d_state + j;
                check(state[index], oracle[time].ssm_state[index],
                      label + ".state[t=" + std::to_string(time) +
                          ",i=" + std::to_string(index) + "]");
            }
        }
    }
}

void test_tolerance_fail_closed() {
    if (!mamba_selective_state::within_tolerance(1.0F, 1.00001F) ||
        mamba_selective_state::within_tolerance(1.0F, 1.001F) ||
        mamba_selective_state::within_tolerance(
            std::numeric_limits<float>::infinity(), 0.0F) ||
        mamba_selective_state::within_tolerance(
            std::numeric_limits<float>::quiet_NaN(), 0.0F)) {
        throw std::runtime_error("tolerance contract did not fail closed");
    }
}

} // namespace

int main() {
    try {
        test_tolerance_fail_closed();
        run_case(Dimensions{/*d_model=*/5, /*d_inner=*/7, /*d_state=*/3,
                            /*conv_width=*/4, /*dt_rank=*/2},
                 11, "small");
        run_case(Dimensions{/*d_model=*/64, /*d_inner=*/64, /*d_state=*/16,
                            /*conv_width=*/4, /*dt_rank=*/8},
                 17, "device_shape");
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "PASS: bounded selective-state scan matches CPU oracle\n";
    std::cout << "checked_values=" << values_checked << '\n';
    std::cout << "max_absolute_error=" << max_absolute_error << '\n';
    std::cout << "max_tolerance_fraction=" << max_tolerance_fraction << '\n';
    std::cout << "absolute_tolerance="
              << mamba_selective_state::kAbsoluteTolerance
              << " relative_tolerance="
              << mamba_selective_state::kRelativeTolerance << '\n';
    return EXIT_SUCCESS;
}
