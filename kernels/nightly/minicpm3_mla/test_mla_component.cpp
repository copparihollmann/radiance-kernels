#include "minicpm3_mla_data.h"
#include "mla_component.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kRegion1Base = 0x20000000U;
constexpr uint32_t kRegion3Base = 0x60000000U;

std::vector<unsigned char> read_file(const std::string &path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path);
    const std::streamsize size = stream.tellg();
    stream.seekg(0);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    if (!stream.read(reinterpret_cast<char *>(bytes.data()), size))
        throw std::runtime_error("cannot read " + path);
    return bytes;
}

const float *float_at(const std::vector<unsigned char> &blob, uint32_t address,
                      uint32_t base, std::size_t elements) {
    const std::size_t offset = address - base;
    if (address < base || offset + elements * sizeof(float) > blob.size())
        throw std::runtime_error("generated tensor is outside its region blob");
    return reinterpret_cast<const float *>(blob.data() + offset);
}

void check_contract_closes() {
    if (!minicpm3_mla::within_tolerance(1.0F, 1.00001F) ||
        minicpm3_mla::within_tolerance(1.0F, 1.001F) ||
        minicpm3_mla::within_tolerance(
            std::numeric_limits<float>::infinity(), 0.0F) ||
        minicpm3_mla::within_tolerance(
            std::numeric_limits<float>::quiet_NaN(), 0.0F))
        throw std::runtime_error("tolerance contract did not fail closed");
}

} // namespace

int main() {
    try {
        check_contract_closes();
        const std::vector<unsigned char> region1 =
            read_file("data/minicpm3_mla.region1.bin");
        const std::vector<unsigned char> region3 =
            read_file("data/minicpm3_mla.region3.bin");

        const float *hidden = float_at(
            region1, MCM3_HIDDEN_INPUT_ADDR, kRegion1Base,
            MCM3_TOKENS * MCM3_HIDDEN);
        const float *kv_a = float_at(
            region1, MCM3_KV_A_WEIGHT_ADDR, kRegion1Base,
            MCM3_COMPRESSION_WIDTH * MCM3_HIDDEN);
        const float *norm_weight = float_at(
            region1, MCM3_KV_NORM_WEIGHT_ADDR, kRegion1Base, MCM3_KV_RANK);
        const float *kv_b = float_at(
            region1, MCM3_KV_B_WEIGHT_ADDR, kRegion1Base,
            MCM3_RECONSTRUCTION_WIDTH * MCM3_KV_RANK);
        const float *gold = float_at(region3, MCM3_GOLD_ADDR, kRegion3Base,
                                     MCM3_OUTPUT_ELEMENTS);

        std::vector<float> output(MCM3_OUTPUT_ELEMENTS,
                                  std::numeric_limits<float>::quiet_NaN());
        for (uint32_t token = 0; token < MCM3_TOKENS; ++token) {
            for (uint32_t component = 0; component < MCM3_COMPRESSION_WIDTH;
                 ++component) {
                const float value = minicpm3_mla::dot(
                    hidden + token * MCM3_HIDDEN,
                    kv_a + component * MCM3_HIDDEN, MCM3_HIDDEN);
                const uint32_t index =
                    component < MCM3_KV_RANK
                        ? MCM3_LATENT_OFFSET + token * MCM3_KV_RANK + component
                        : MCM3_ROPE_OFFSET + token * MCM3_ROPE_DIM + component -
                              MCM3_KV_RANK;
                output[index] = value;
            }
            const float *latent =
                output.data() + MCM3_LATENT_OFFSET + token * MCM3_KV_RANK;
            for (uint32_t column = 0; column < MCM3_KV_RANK; ++column)
                output[MCM3_NORMALIZED_OFFSET + token * MCM3_KV_RANK +
                       column] = minicpm3_mla::normalize_element(
                    latent, norm_weight, MCM3_KV_RANK, column);
            const float *normalized = output.data() + MCM3_NORMALIZED_OFFSET +
                                      token * MCM3_KV_RANK;
            for (uint32_t component = 0;
                 component < MCM3_RECONSTRUCTION_WIDTH; ++component)
                output[MCM3_RECONSTRUCTED_OFFSET +
                       token * MCM3_RECONSTRUCTION_WIDTH + component] =
                    minicpm3_mla::dot(
                        normalized, kv_b + component * MCM3_KV_RANK,
                        MCM3_KV_RANK);
        }

        float max_absolute_error = 0.0F;
        float max_tolerance_fraction = 0.0F;
        for (uint32_t index = 0; index < MCM3_OUTPUT_ELEMENTS; ++index) {
            const float absolute = std::fabs(output[index] - gold[index]);
            const float scale =
                std::max(std::fabs(output[index]), std::fabs(gold[index]));
            const float limit = minicpm3_mla::kAbsoluteTolerance +
                                minicpm3_mla::kRelativeTolerance * scale;
            max_absolute_error = std::max(max_absolute_error, absolute);
            max_tolerance_fraction =
                std::max(max_tolerance_fraction, absolute / limit);
            if (!minicpm3_mla::within_tolerance(output[index], gold[index]))
                throw std::runtime_error(
                    "full-output mismatch at element " +
                    std::to_string(index) + ": actual=" +
                    std::to_string(output[index]) + " expected=" +
                    std::to_string(gold[index]));
        }

        std::cout << "PASS: portable FP32 component matches independent oracle\n";
        std::cout << "checked_outputs=" << MCM3_OUTPUT_ELEMENTS << '\n';
        std::cout << "max_absolute_error=" << max_absolute_error << '\n';
        std::cout << "max_tolerance_fraction=" << max_tolerance_fraction
                  << '\n';
        std::cout << "absolute_tolerance="
                  << minicpm3_mla::kAbsoluteTolerance
                  << " relative_tolerance="
                  << minicpm3_mla::kRelativeTolerance << '\n';
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
