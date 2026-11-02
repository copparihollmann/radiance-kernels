#include "mamba_state_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mamba_reference::Dimensions;
using mamba_reference::Parameters;
using mamba_reference::StepTrace;
using mamba_reference::StreamingReference;
using mamba_reference::TokenInput;

constexpr float kAbsoluteTolerance = 2.0e-5F;
constexpr float kRelativeTolerance = 2.0e-5F;

std::uint32_t g_rng_state = 0x5A17C9E3U;
std::size_t g_values_checked = 0;

float deterministic_value(float scale = 1.0F, float offset = 0.0F) {
  // Fixed LCG: deterministic across standard-library implementations.
  g_rng_state = g_rng_state * 1664525U + 1013904223U;
  const float unit = static_cast<float>((g_rng_state >> 8U) & 0x00FFFFFFU) /
                     static_cast<float>(0x01000000U);
  return offset + scale * (2.0F * unit - 1.0F);
}

std::vector<float> make_values(std::size_t count, float scale,
                               float offset = 0.0F) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = deterministic_value(scale, offset);
  }
  return values;
}

Parameters make_parameters(const Dimensions& dims) {
  Parameters parameters;
  parameters.conv_weight =
      make_values(dims.d_inner * dims.conv_width, 0.20F);
  parameters.conv_bias = make_values(dims.d_inner, 0.05F);
  parameters.dt_weight = make_values(dims.d_inner * dims.dt_rank, 0.12F);
  parameters.dt_bias = make_values(dims.d_inner, 0.20F, -1.0F);
  parameters.a_log =
      make_values(dims.d_inner * dims.d_state, 0.60F, -0.90F);
  parameters.d_skip = make_values(dims.d_inner, 0.30F);
  return parameters;
}

std::vector<TokenInput> make_tokens(const Dimensions& dims,
                                    std::size_t sequence_length) {
  std::vector<TokenInput> tokens(sequence_length);
  for (TokenInput& token : tokens) {
    token.x = make_values(dims.d_inner, 0.50F);
    token.dt_features = make_values(dims.dt_rank, 0.40F);
    token.b = make_values(dims.d_state, 0.35F);
    token.c = make_values(dims.d_state, 0.35F);
    token.gate = make_values(dims.d_inner, 0.80F);
  }
  return tokens;
}

void require_close(float actual, float expected, const std::string& context) {
  ++g_values_checked;
  const float error = std::fabs(actual - expected);
  const float limit =
      kAbsoluteTolerance +
      kRelativeTolerance * std::max(std::fabs(actual), std::fabs(expected));
  if (!std::isfinite(actual) || !std::isfinite(expected) || error > limit) {
    throw std::runtime_error(context + ": actual=" + std::to_string(actual) +
                             ", expected=" + std::to_string(expected) +
                             ", error=" + std::to_string(error) +
                             ", tolerance=" + std::to_string(limit));
  }
}

void require_vector_close(const std::vector<float>& actual,
                          const std::vector<float>& expected,
                          const std::string& context) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(context + ": size mismatch");
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    require_close(actual[index], expected[index],
                  context + "[" + std::to_string(index) + "]");
  }
}

void compare_trace(const StepTrace& actual, const StepTrace& expected,
                   const std::string& context) {
  require_vector_close(actual.conv_raw, expected.conv_raw,
                       context + ".conv_raw");
  require_vector_close(actual.conv_activated, expected.conv_activated,
                       context + ".conv_activated");
  require_vector_close(actual.delta, expected.delta, context + ".delta");
  require_vector_close(actual.output, expected.output, context + ".output");
  require_vector_close(actual.conv_state, expected.conv_state,
                       context + ".conv_state");
  require_vector_close(actual.ssm_state, expected.ssm_state,
                       context + ".ssm_state");
}

void test_known_ring_order() {
  const Dimensions dims{/*d_model=*/2, /*d_inner=*/2, /*d_state=*/1,
                        /*conv_width=*/4, /*dt_rank=*/1};
  Parameters parameters;
  parameters.conv_weight = {1.0F, 10.0F, 100.0F, 1000.0F,
                            1.0F, 10.0F, 100.0F, 1000.0F};
  parameters.conv_bias = {0.0F, 0.0F};
  parameters.dt_weight = {0.0F, 0.0F};
  parameters.dt_bias = {-1.0F, -1.0F};
  parameters.a_log = {-1.0F, -1.0F};
  parameters.d_skip = {0.0F, 0.0F};

  StreamingReference reference(dims, parameters);
  StepTrace trace;
  for (int time = 1; time <= 6; ++time) {
    TokenInput token;
    token.x = {static_cast<float>(time), static_cast<float>(10 * time)};
    token.dt_features = {0.0F};
    token.b = {0.0F};
    token.c = {0.0F};
    token.gate = {0.0F, 0.0F};
    trace = reference.step(token);
  }
  require_vector_close(trace.conv_state,
                       {3.0F, 4.0F, 5.0F, 6.0F,
                        30.0F, 40.0F, 50.0F, 60.0F},
                       "known_ring.state");
  require_close(trace.conv_raw[0], 3.0F + 40.0F + 500.0F + 6000.0F,
                "known_ring.output_channel_0");
  require_close(trace.conv_raw[1], 30.0F + 400.0F + 5000.0F + 60000.0F,
                "known_ring.output_channel_1");
}

void test_prefill_decode_equivalence(const Dimensions& dims,
                                     std::size_t sequence_length,
                                     std::size_t prefill_length,
                                     const std::string& label) {
  if (prefill_length + 1 != sequence_length) {
    throw std::invalid_argument("test requires exactly one decode token");
  }
  const Parameters parameters = make_parameters(dims);
  const std::vector<TokenInput> tokens = make_tokens(dims, sequence_length);
  const std::vector<StepTrace> full =
      mamba_reference::run_full_sequence(dims, parameters, tokens);

  // Streaming prefill followed by exactly one cached decode token.
  StreamingReference split(dims, parameters);
  std::vector<StepTrace> split_traces;
  split_traces.reserve(sequence_length);
  for (std::size_t time = 0; time < prefill_length; ++time) {
    split_traces.push_back(split.step(tokens[time]));
  }
  const std::vector<float> prefill_conv_state = split.canonical_conv_state();
  const std::vector<float> prefill_ssm_state = split.ssm_state();
  split_traces.push_back(split.step(tokens[prefill_length]));

  // Compare every intermediate output and every convolution/recurrent state
  // element, not only the final token.
  for (std::size_t time = 0; time < sequence_length; ++time) {
    compare_trace(split_traces[time], full[time],
                  label + ".time_" + std::to_string(time));
  }
  require_vector_close(prefill_conv_state, full[prefill_length - 1].conv_state,
                       label + ".prefill_boundary.conv_state");
  require_vector_close(prefill_ssm_state, full[prefill_length - 1].ssm_state,
                       label + ".prefill_boundary.ssm_state");

  // Also exercise the convenience whole-stream path.
  const std::vector<StepTrace> streaming =
      mamba_reference::run_streaming(dims, parameters, tokens);
  for (std::size_t time = 0; time < sequence_length; ++time) {
    compare_trace(streaming[time], full[time],
                  label + ".whole_stream.time_" + std::to_string(time));
  }
}

void test_shape_validation() {
  Dimensions dims{/*d_model=*/3, /*d_inner=*/4, /*d_state=*/2,
                  /*conv_width=*/4, /*dt_rank=*/2};
  Parameters parameters = make_parameters(dims);
  parameters.a_log.pop_back();
  bool threw = false;
  try {
    StreamingReference reference(dims, parameters);
    (void)reference;
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) {
    throw std::runtime_error("shape validation accepted a truncated a_log");
  }
}

}  // namespace

int main() {
  try {
    test_known_ring_order();
    test_prefill_decode_equivalence(
        Dimensions{/*d_model=*/5, /*d_inner=*/7, /*d_state=*/3,
                   /*conv_width=*/4, /*dt_rank=*/2},
        /*sequence_length=*/11, /*prefill_length=*/10, "small");
    test_prefill_decode_equivalence(Dimensions::canonical(),
                                    /*sequence_length=*/5,
                                    /*prefill_length=*/4, "canonical");
    test_shape_validation();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "PASS: Mamba convolution and selective-state reference\n";
  std::cout << "checked_values=" << g_values_checked << '\n';
  std::cout << "absolute_tolerance=" << kAbsoluteTolerance
            << " relative_tolerance=" << kRelativeTolerance << '\n';
  std::cout << "state_precision=FP32\n";
  return EXIT_SUCCESS;
}
