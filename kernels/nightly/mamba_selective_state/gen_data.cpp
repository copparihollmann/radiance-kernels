#include "../../mamba_state_reference/mamba_state_reference.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kSequence = 17;
constexpr std::size_t kChannels = 64;
constexpr std::size_t kState = 16;
constexpr std::size_t kConvWidth = 4;
constexpr std::size_t kDtRank = 8;
constexpr std::uint32_t kRegionBase[] = {0, 0x20000000U, 0x40000000U,
                                         0x60000000U};

struct Definition {
    std::string name;
    std::string value;
    std::string comment;
};

class Layout {
  public:
    template <typename T>
    void add(const std::string &name, unsigned region,
             const std::vector<T> &values, const std::string &comment) {
        std::vector<std::uint8_t> &blob = blobs_[region];
        blob.resize((blob.size() + 255U) & ~std::size_t{255U}, 0);
        const std::uint32_t address =
            kRegionBase[region] + static_cast<std::uint32_t>(blob.size());
        const std::size_t bytes = values.size() * sizeof(T);
        const std::uint8_t *first =
            reinterpret_cast<const std::uint8_t *>(values.data());
        blob.insert(blob.end(), first, first + bytes);
        define(name + "_ADDR", hex(address) + "u", comment);
        define(name + "_BYTES", std::to_string(bytes) + "u", "");
    }

    void define(const std::string &name, const std::string &value,
                const std::string &comment = "") {
        definitions_.push_back({name, value, comment});
    }

    void write() const {
        std::filesystem::create_directories("data");
        for (unsigned region = 1; region <= 3; ++region) {
            const std::vector<std::uint8_t> &blob = blobs_[region];
            if (blob.empty())
                continue;
            std::ofstream output("data/mamba_selective_state.region" +
                                     std::to_string(region) + ".bin",
                                 std::ios::binary);
            output.write(reinterpret_cast<const char *>(blob.data()),
                         static_cast<std::streamsize>(blob.size()));
            if (!output)
                throw std::runtime_error("failed to write data blob");
        }
        std::ofstream header("mamba_selective_state_data.h");
        header
            << "// generated from kernels/mamba_state_reference; do not edit\n"
               "#ifndef MAMBA_SELECTIVE_STATE_DATA_H\n"
               "#define MAMBA_SELECTIVE_STATE_DATA_H\n";
        for (const Definition &definition : definitions_) {
            header << "#define " << definition.name << ' ' << definition.value;
            if (!definition.comment.empty())
                header << "  // " << definition.comment;
            header << '\n';
        }
        header << "#endif\n";
        if (!header)
            throw std::runtime_error("failed to write generated header");
    }

  private:
    static std::string hex(std::uint32_t value) {
        std::ostringstream stream;
        stream << "0x" << std::hex << std::setw(8) << std::setfill('0')
               << value;
        return stream.str();
    }

    std::vector<std::uint8_t> blobs_[4];
    std::vector<Definition> definitions_;
};

std::uint32_t rng_state = 0x43C2A91DU;

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

} // namespace

int main() {
    using namespace mamba_reference;
    const Dimensions dims{/*d_model=*/kChannels, /*d_inner=*/kChannels,
                          /*d_state=*/kState, /*conv_width=*/kConvWidth,
                          /*dt_rank=*/kDtRank};
    Parameters p;
    p.conv_weight = values(dims.d_inner * dims.conv_width, 0.20F);
    p.conv_bias = values(dims.d_inner, 0.05F);
    p.dt_weight = values(dims.d_inner * dims.dt_rank, 0.12F);
    p.dt_bias = values(dims.d_inner, 0.20F, -1.0F);
    p.a_log = values(dims.d_inner * dims.d_state, 0.60F, -0.90F);
    p.d_skip = values(dims.d_inner, 0.30F);

    std::vector<TokenInput> tokens(kSequence);
    for (TokenInput &token : tokens) {
        token.x = values(dims.d_inner, 0.50F);
        token.dt_features = values(dims.dt_rank, 0.40F);
        token.b = values(dims.d_state, 0.35F);
        token.c = values(dims.d_state, 0.35F);
        token.gate = values(dims.d_inner, 0.80F);
    }
    const std::vector<StepTrace> oracle = run_full_sequence(dims, p, tokens);

    std::vector<float> delta(kSequence * kChannels);
    std::vector<float> x_activated(kSequence * kChannels);
    std::vector<float> b(kSequence * kState);
    std::vector<float> c(kSequence * kState);
    std::vector<float> gate(kSequence * kChannels);
    std::vector<float> golden_output(kSequence * kChannels);
    for (std::size_t time = 0; time < kSequence; ++time) {
        std::memcpy(delta.data() + time * kChannels, oracle[time].delta.data(),
                    kChannels * sizeof(float));
        std::memcpy(x_activated.data() + time * kChannels,
                    oracle[time].conv_activated.data(),
                    kChannels * sizeof(float));
        std::memcpy(b.data() + time * kState, tokens[time].b.data(),
                    kState * sizeof(float));
        std::memcpy(c.data() + time * kState, tokens[time].c.data(),
                    kState * sizeof(float));
        std::memcpy(gate.data() + time * kChannels, tokens[time].gate.data(),
                    kChannels * sizeof(float));
        std::memcpy(golden_output.data() + time * kChannels,
                    oracle[time].output.data(), kChannels * sizeof(float));
    }
    const std::vector<float> &golden_state = oracle.back().ssm_state;
    const std::size_t output_elements = kSequence * kChannels;
    const std::size_t state_elements = kChannels * kState;
    const std::size_t checks = output_elements + state_elements;
    std::vector<float> poisoned_output(output_elements, 0.0F);
    std::vector<float> poisoned_state(state_elements, 0.0F);
    std::memset(poisoned_output.data(), 0xEE,
                poisoned_output.size() * sizeof(float));
    std::memset(poisoned_state.data(), 0xEE,
                poisoned_state.size() * sizeof(float));
    std::vector<std::uint16_t> poisoned_status(checks, 0xEEEEU);
    std::vector<std::uint16_t> golden_status(checks, 0U);

    Layout layout;
    layout.define("MSS_SEQUENCE", std::to_string(kSequence) + "u");
    layout.define("MSS_CHANNELS", std::to_string(kChannels) + "u");
    layout.define("MSS_STATE", std::to_string(kState) + "u");
    layout.define("MSS_OUTPUT_ELEMENTS", std::to_string(output_elements) + "u");
    layout.define("MSS_STATE_ELEMENTS", std::to_string(state_elements) + "u");
    layout.define("MSS_CHECKS", std::to_string(checks) + "u");
    layout.add("MSS_DELTA", 1, delta, "oracle softplus timestep [T][channels]");
    layout.add("MSS_X", 1, x_activated,
               "oracle convolution activation [T][channels]");
    layout.add("MSS_B", 1, b, "selective B [T][state]");
    layout.add("MSS_C", 1, c, "selective C [T][state]");
    layout.add("MSS_GATE", 1, gate, "gate z [T][channels]");
    layout.add("MSS_A_LOG", 1, p.a_log, "A_log [channels][state]");
    layout.add("MSS_D", 1, p.d_skip, "skip D [channels]");
    layout.add("MSS_OUTPUT", 2, poisoned_output,
               "FP32 scan output [T][channels]");
    layout.add("MSS_FINAL_STATE", 2, poisoned_state,
               "FP32 final state [channels][state]");
    layout.add("MSS_STATUS", 2, poisoned_status,
               "BF16 zero iff corresponding float is in tolerance");
    layout.add("MSS_GOLD_OUTPUT", 3, golden_output, "CPU-oracle FP32 output");
    layout.add("MSS_GOLD_STATE", 3, golden_state,
               "CPU-oracle FP32 final state");
    layout.add("MSS_GOLD_STATUS", 3, golden_status,
               "all-zero BF16 status golden");
    layout.write();
    std::cout << "generated T=" << kSequence << " channels=" << kChannels
              << " state=" << kState << " checks=" << checks << '\n';
    return 0;
}
