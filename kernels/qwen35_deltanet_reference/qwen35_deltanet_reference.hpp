#ifndef QWEN35_DELTANET_REFERENCE_HPP
#define QWEN35_DELTANET_REFERENCE_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qwen35_deltanet_reference {

// Qwen3.5-4B's recurrent interface after the 16 Q/K heads have been repeated
// to the 32 value heads. Tests may construct smaller dimensions.
inline constexpr std::size_t kOfficialValueHeads = 32;
inline constexpr std::size_t kOfficialKeyHeads = 16;
inline constexpr std::size_t kOfficialKeyDim = 128;
inline constexpr std::size_t kOfficialValueDim = 128;
inline constexpr float kL2Epsilon = 1.0e-6F;

struct Dimensions {
    std::size_t heads = kOfficialValueHeads;
    std::size_t key_dim = kOfficialKeyDim;
    std::size_t value_dim = kOfficialValueDim;

    static Dimensions official_recurrent() { return {}; }

    void validate() const {
        if (heads == 0 || key_dim == 0 || value_dim == 0)
            throw std::invalid_argument("all DeltaNet dimensions must be nonzero");
    }
};

inline void require_size(const std::vector<float> &values,
                         std::size_t expected, const char *name) {
    if (values.size() != expected) {
        throw std::invalid_argument(std::string(name) + " has size " +
                                    std::to_string(values.size()) +
                                    ", expected " +
                                    std::to_string(expected));
    }
}

struct Parameters {
    // One scalar per expanded value head.
    std::vector<float> a_log;
    std::vector<float> dt_bias;

    void validate(const Dimensions &dims) const {
        dims.validate();
        require_size(a_log, dims.heads, "a_log");
        require_size(dt_bias, dims.heads, "dt_bias");
    }
};

struct TokenInput {
    // q and k are the post-convolution vectors after Q/K-head repetition.
    // a and b are projection outputs used to form log-decay g and beta.
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> a;
    std::vector<float> b;

    void validate(const Dimensions &dims) const {
        require_size(q, dims.heads * dims.key_dim, "q");
        require_size(k, dims.heads * dims.key_dim, "k");
        require_size(v, dims.heads * dims.value_dim, "v");
        require_size(a, dims.heads, "a");
        require_size(b, dims.heads, "b");
    }
};

struct StepTrace {
    std::vector<float> beta;
    std::vector<float> log_decay;
    std::vector<float> output;
    // Canonical row-major layout [head][key][value].
    std::vector<float> state;
};

inline float sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }

inline float softplus(float x) {
    if (x > 20.0F)
        return x;
    if (x < -20.0F)
        return std::exp(x);
    return std::log1p(std::exp(x));
}

inline float inverse_l2_norm(const float *x, std::size_t count) {
    float sum = 0.0F;
    for (std::size_t i = 0; i < count; ++i)
        sum += x[i] * x[i];
    // This is the exact normalization convention used by the pinned source:
    // x * rsqrt(sum(x*x) + 1e-6), not max(norm, epsilon).
    return 1.0F / std::sqrt(sum + kL2Epsilon);
}

class StreamingReference {
  public:
    StreamingReference(Dimensions dims, Parameters parameters)
        : dims_(dims), parameters_(std::move(parameters)),
          state_(dims_.heads * dims_.key_dim * dims_.value_dim, 0.0F) {
        dims_.validate();
        parameters_.validate(dims_);
    }

    StepTrace step(const TokenInput &input) {
        input.validate(dims_);
        StepTrace trace;
        trace.beta.resize(dims_.heads);
        trace.log_decay.resize(dims_.heads);
        trace.output.assign(dims_.heads * dims_.value_dim, 0.0F);
        const float query_scale = 1.0F / std::sqrt(static_cast<float>(dims_.key_dim));

        for (std::size_t head = 0; head < dims_.heads; ++head) {
            const float *q = input.q.data() + head * dims_.key_dim;
            const float *k = input.k.data() + head * dims_.key_dim;
            const float *v = input.v.data() + head * dims_.value_dim;
            const float q_inv = inverse_l2_norm(q, dims_.key_dim);
            const float k_inv = inverse_l2_norm(k, dims_.key_dim);
            const float beta = sigmoid(input.b[head]);
            const float log_decay =
                -std::exp(parameters_.a_log[head]) *
                softplus(input.a[head] + parameters_.dt_bias[head]);
            const float decay = std::exp(log_decay);
            trace.beta[head] = beta;
            trace.log_decay[head] = log_decay;

            for (std::size_t value = 0; value < dims_.value_dim; ++value) {
                float memory = 0.0F;
                for (std::size_t key = 0; key < dims_.key_dim; ++key) {
                    const std::size_t index = state_index(head, key, value);
                    state_[index] *= decay;
                    memory += state_[index] * (k[key] * k_inv);
                }
                const float delta = (v[value] - memory) * beta;
                float result = 0.0F;
                for (std::size_t key = 0; key < dims_.key_dim; ++key) {
                    const std::size_t index = state_index(head, key, value);
                    state_[index] += (k[key] * k_inv) * delta;
                    result += state_[index] * (q[key] * q_inv * query_scale);
                }
                trace.output[head * dims_.value_dim + value] = result;
            }
        }
        trace.state = state_;
        return trace;
    }

    const std::vector<float> &state() const { return state_; }

  private:
    std::size_t state_index(std::size_t head, std::size_t key,
                            std::size_t value) const {
        return (head * dims_.key_dim + key) * dims_.value_dim + value;
    }

    Dimensions dims_;
    Parameters parameters_;
    std::vector<float> state_;
};

inline std::vector<StepTrace>
run_streaming(const Dimensions &dims, const Parameters &parameters,
              const std::vector<TokenInput> &tokens) {
    StreamingReference reference(dims, parameters);
    std::vector<StepTrace> traces;
    traces.reserve(tokens.size());
    for (const TokenInput &token : tokens)
        traces.push_back(reference.step(token));
    return traces;
}

// Independently indexed full-sequence implementation. It intentionally does
// not call StreamingReference or any of its helpers for the recurrence. The
// state layout and loop nesting match the mathematical [head][key][value]
// tensor rather than the streaming implementation's per-value update path.
inline std::vector<StepTrace>
run_full_sequence(const Dimensions &dims, const Parameters &parameters,
                  const std::vector<TokenInput> &tokens) {
    dims.validate();
    parameters.validate(dims);
    for (const TokenInput &token : tokens)
        token.validate(dims);

    const std::size_t state_elements = dims.heads * dims.key_dim * dims.value_dim;
    std::vector<float> state(state_elements, 0.0F);
    std::vector<StepTrace> traces(tokens.size());
    const float query_scale = 1.0F / std::sqrt(static_cast<float>(dims.key_dim));

    for (std::size_t time = 0; time < tokens.size(); ++time) {
        const TokenInput &input = tokens[time];
        StepTrace &trace = traces[time];
        trace.beta.resize(dims.heads);
        trace.log_decay.resize(dims.heads);
        trace.output.assign(dims.heads * dims.value_dim, 0.0F);

        for (std::size_t head = 0; head < dims.heads; ++head) {
            const float *q = input.q.data() + head * dims.key_dim;
            const float *k = input.k.data() + head * dims.key_dim;
            float q_square_sum = 0.0F;
            float k_square_sum = 0.0F;
            for (std::size_t key = 0; key < dims.key_dim; ++key) {
                q_square_sum += q[key] * q[key];
                k_square_sum += k[key] * k[key];
            }
            const float q_multiplier =
                query_scale / std::sqrt(q_square_sum + kL2Epsilon);
            const float k_multiplier =
                1.0F / std::sqrt(k_square_sum + kL2Epsilon);
            const float beta = 1.0F / (1.0F + std::exp(-input.b[head]));
            const float log_decay =
                -std::exp(parameters.a_log[head]) *
                std::log1p(std::exp(input.a[head] + parameters.dt_bias[head]));
            const float decay = std::exp(log_decay);
            trace.beta[head] = beta;
            trace.log_decay[head] = log_decay;

            // Decay the complete matrix first, as in the source equation.
            const std::size_t base = head * dims.key_dim * dims.value_dim;
            for (std::size_t key = 0; key < dims.key_dim; ++key)
                for (std::size_t value = 0; value < dims.value_dim; ++value)
                    state[base + key * dims.value_dim + value] *= decay;

            std::vector<float> memory(dims.value_dim, 0.0F);
            for (std::size_t key = 0; key < dims.key_dim; ++key) {
                const float normalized_key = k[key] * k_multiplier;
                for (std::size_t value = 0; value < dims.value_dim; ++value)
                    memory[value] +=
                        state[base + key * dims.value_dim + value] * normalized_key;
            }
            for (std::size_t value = 0; value < dims.value_dim; ++value) {
                const float correction =
                    (input.v[head * dims.value_dim + value] - memory[value]) * beta;
                for (std::size_t key = 0; key < dims.key_dim; ++key)
                    state[base + key * dims.value_dim + value] +=
                        (k[key] * k_multiplier) * correction;
            }
            for (std::size_t key = 0; key < dims.key_dim; ++key) {
                const float normalized_query = q[key] * q_multiplier;
                for (std::size_t value = 0; value < dims.value_dim; ++value)
                    trace.output[head * dims.value_dim + value] +=
                        state[base + key * dims.value_dim + value] * normalized_query;
            }
        }
        trace.state = state;
    }
    return traces;
}

} // namespace qwen35_deltanet_reference

#endif
