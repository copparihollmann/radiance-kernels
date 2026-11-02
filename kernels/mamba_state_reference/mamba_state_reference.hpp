#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mamba_reference {

// Canonical dimensions for state-spaces/mamba-790m-hf. Tests may use smaller
// dimensions by constructing Dimensions directly.
inline constexpr std::size_t kCanonicalDModel = 1536;
inline constexpr std::size_t kCanonicalDInner = 3072;
inline constexpr std::size_t kCanonicalDState = 16;
inline constexpr std::size_t kCanonicalConvWidth = 4;
inline constexpr std::size_t kCanonicalDtRank = 96;

struct Dimensions {
  std::size_t d_model = kCanonicalDModel;
  std::size_t d_inner = kCanonicalDInner;
  std::size_t d_state = kCanonicalDState;
  std::size_t conv_width = kCanonicalConvWidth;
  std::size_t dt_rank = kCanonicalDtRank;

  static Dimensions canonical() { return {}; }

  void validate() const {
    if (d_model == 0 || d_inner == 0 || d_state == 0 || conv_width == 0 ||
        dt_rank == 0) {
      throw std::invalid_argument("all Mamba dimensions must be nonzero");
    }
  }
};

struct Parameters {
  // Layouts are row-major:
  //   conv_weight[d_inner][conv_width], oldest tap to newest tap
  //   dt_weight[d_inner][dt_rank]
  //   a_log[d_inner][d_state]
  std::vector<float> conv_weight;
  std::vector<float> conv_bias;
  std::vector<float> dt_weight;
  std::vector<float> dt_bias;
  std::vector<float> a_log;
  std::vector<float> d_skip;

  void validate(const Dimensions& dims) const {
    dims.validate();
    require_size(conv_weight, dims.d_inner * dims.conv_width,
                 "conv_weight");
    require_size(conv_bias, dims.d_inner, "conv_bias");
    require_size(dt_weight, dims.d_inner * dims.dt_rank, "dt_weight");
    require_size(dt_bias, dims.d_inner, "dt_bias");
    require_size(a_log, dims.d_inner * dims.d_state, "a_log");
    require_size(d_skip, dims.d_inner, "d_skip");
  }

 private:
  static void require_size(const std::vector<float>& values,
                           std::size_t expected, const char* name) {
    if (values.size() != expected) {
      throw std::invalid_argument(std::string(name) + " has size " +
                                  std::to_string(values.size()) +
                                  ", expected " +
                                  std::to_string(expected));
    }
  }
};

struct TokenInput {
  // x is the depthwise-convolution input. dt_features, b and c are the
  // sequence-dependent values produced by Mamba's x projection. gate is z.
  std::vector<float> x;
  std::vector<float> dt_features;
  std::vector<float> b;
  std::vector<float> c;
  std::vector<float> gate;

  void validate(const Dimensions& dims) const {
    require_size(x, dims.d_inner, "x");
    require_size(dt_features, dims.dt_rank, "dt_features");
    require_size(b, dims.d_state, "b");
    require_size(c, dims.d_state, "c");
    require_size(gate, dims.d_inner, "gate");
  }

 private:
  static void require_size(const std::vector<float>& values,
                           std::size_t expected, const char* name) {
    if (values.size() != expected) {
      throw std::invalid_argument(std::string(name) + " has size " +
                                  std::to_string(values.size()) +
                                  ", expected " +
                                  std::to_string(expected));
    }
  }
};

struct StepTrace {
  std::vector<float> conv_raw;
  std::vector<float> conv_activated;
  std::vector<float> delta;
  std::vector<float> output;

  // Canonical snapshots make every state element directly comparable:
  // conv_state[channel][age], ordered oldest to newest, and
  // ssm_state[channel][state_index]. Both are FP32.
  std::vector<float> conv_state;
  std::vector<float> ssm_state;
};

inline float silu(float x) { return x / (1.0F + std::exp(-x)); }

inline float softplus(float x) {
  // Stable FP32 softplus. The branches avoid overflow and avoid losing the
  // small exponential tail.
  if (x > 20.0F) {
    return x;
  }
  if (x < -20.0F) {
    return std::exp(x);
  }
  return std::log1p(std::exp(x));
}

class StreamingReference {
 public:
  StreamingReference(Dimensions dims, Parameters parameters)
      : dims_(dims),
        parameters_(std::move(parameters)),
        conv_ring_(dims_.d_inner * dims_.conv_width, 0.0F),
        ssm_state_(dims_.d_inner * dims_.d_state, 0.0F) {
    dims_.validate();
    parameters_.validate(dims_);
  }

  StepTrace step(const TokenInput& input) {
    input.validate(dims_);
    StepTrace trace;
    trace.conv_raw.assign(dims_.d_inner, 0.0F);
    trace.conv_activated.assign(dims_.d_inner, 0.0F);
    trace.delta.assign(dims_.d_inner, 0.0F);
    trace.output.assign(dims_.d_inner, 0.0F);

    // Overwrite the oldest slot with the newest token, then advance the
    // oldest-slot pointer. Logical tap conv_width-1 is therefore "now".
    for (std::size_t channel = 0; channel < dims_.d_inner; ++channel) {
      conv_ring_[channel * dims_.conv_width + oldest_slot_] = input.x[channel];
    }
    oldest_slot_ = (oldest_slot_ + 1) % dims_.conv_width;

    for (std::size_t channel = 0; channel < dims_.d_inner; ++channel) {
      float sum = parameters_.conv_bias[channel];
      for (std::size_t tap = 0; tap < dims_.conv_width; ++tap) {
        const std::size_t physical = (oldest_slot_ + tap) % dims_.conv_width;
        sum += conv_ring_[channel * dims_.conv_width + physical] *
               parameters_.conv_weight[channel * dims_.conv_width + tap];
      }
      trace.conv_raw[channel] = sum;
      trace.conv_activated[channel] = silu(sum);
    }

    for (std::size_t channel = 0; channel < dims_.d_inner; ++channel) {
      float projected_dt = parameters_.dt_bias[channel];
      for (std::size_t rank = 0; rank < dims_.dt_rank; ++rank) {
        projected_dt +=
            parameters_.dt_weight[channel * dims_.dt_rank + rank] *
            input.dt_features[rank];
      }
      const float delta = softplus(projected_dt);
      trace.delta[channel] = delta;

      float y = 0.0F;
      for (std::size_t state_index = 0; state_index < dims_.d_state;
           ++state_index) {
        const std::size_t index = channel * dims_.d_state + state_index;
        const float a = -std::exp(parameters_.a_log[index]);
        const float decay = std::exp(delta * a);
        const float next = decay * ssm_state_[index] +
                           delta * input.b[state_index] *
                               trace.conv_activated[channel];
        ssm_state_[index] = next;
        y += next * input.c[state_index];
      }
      y += parameters_.d_skip[channel] * trace.conv_activated[channel];
      trace.output[channel] = y * silu(input.gate[channel]);
    }

    trace.conv_state = canonical_conv_state();
    trace.ssm_state = ssm_state_;
    return trace;
  }

  std::vector<float> canonical_conv_state() const {
    std::vector<float> result(dims_.d_inner * dims_.conv_width);
    for (std::size_t channel = 0; channel < dims_.d_inner; ++channel) {
      for (std::size_t age = 0; age < dims_.conv_width; ++age) {
        const std::size_t physical = (oldest_slot_ + age) % dims_.conv_width;
        result[channel * dims_.conv_width + age] =
            conv_ring_[channel * dims_.conv_width + physical];
      }
    }
    return result;
  }

  const std::vector<float>& ssm_state() const { return ssm_state_; }

 private:
  Dimensions dims_;
  Parameters parameters_;
  std::vector<float> conv_ring_;
  std::vector<float> ssm_state_;
  std::size_t oldest_slot_ = 0;
};

inline std::vector<StepTrace> run_streaming(const Dimensions& dims,
                                            const Parameters& parameters,
                                            const std::vector<TokenInput>& tokens) {
  StreamingReference reference(dims, parameters);
  std::vector<StepTrace> traces;
  traces.reserve(tokens.size());
  for (const TokenInput& token : tokens) {
    traces.push_back(reference.step(token));
  }
  return traces;
}

// Independent full-sequence formulation. Convolution samples are indexed
// directly from the input sequence rather than through StreamingReference's
// ring buffer. The recurrent state is separately allocated and updated here.
// This provides a non-tautological oracle for prefill+decode tests.
inline std::vector<StepTrace> run_full_sequence(
    const Dimensions& dims, const Parameters& parameters,
    const std::vector<TokenInput>& tokens) {
  dims.validate();
  parameters.validate(dims);
  for (const TokenInput& token : tokens) {
    token.validate(dims);
  }

  std::vector<float> ssm_state(dims.d_inner * dims.d_state, 0.0F);
  std::vector<StepTrace> traces(tokens.size());

  for (std::size_t time = 0; time < tokens.size(); ++time) {
    StepTrace& trace = traces[time];
    trace.conv_raw.assign(dims.d_inner, 0.0F);
    trace.conv_activated.assign(dims.d_inner, 0.0F);
    trace.delta.assign(dims.d_inner, 0.0F);
    trace.output.assign(dims.d_inner, 0.0F);
    trace.conv_state.assign(dims.d_inner * dims.conv_width, 0.0F);

    for (std::size_t channel = 0; channel < dims.d_inner; ++channel) {
      float conv_sum = parameters.conv_bias[channel];
      for (std::size_t tap = 0; tap < dims.conv_width; ++tap) {
        const long sequence_index = static_cast<long>(time) +
                                    static_cast<long>(tap) + 1L -
                                    static_cast<long>(dims.conv_width);
        float sample = 0.0F;
        if (sequence_index >= 0) {
          sample = tokens[static_cast<std::size_t>(sequence_index)].x[channel];
        }
        trace.conv_state[channel * dims.conv_width + tap] = sample;
        conv_sum += sample *
                    parameters.conv_weight[channel * dims.conv_width + tap];
      }
      trace.conv_raw[channel] = conv_sum;
      trace.conv_activated[channel] = silu(conv_sum);

      float projected_dt = parameters.dt_bias[channel];
      for (std::size_t rank = 0; rank < dims.dt_rank; ++rank) {
        projected_dt += parameters.dt_weight[channel * dims.dt_rank + rank] *
                        tokens[time].dt_features[rank];
      }
      trace.delta[channel] = softplus(projected_dt);

      float y = 0.0F;
      for (std::size_t state_index = 0; state_index < dims.d_state;
           ++state_index) {
        const std::size_t index = channel * dims.d_state + state_index;
        const float a = -std::exp(parameters.a_log[index]);
        ssm_state[index] =
            std::exp(trace.delta[channel] * a) * ssm_state[index] +
            trace.delta[channel] * tokens[time].b[state_index] *
                trace.conv_activated[channel];
        y += ssm_state[index] * tokens[time].c[state_index];
      }
      y += parameters.d_skip[channel] * trace.conv_activated[channel];
      trace.output[channel] = y * silu(tokens[time].gate[channel]);
    }
    trace.ssm_state = ssm_state;
  }
  return traces;
}

}  // namespace mamba_reference
