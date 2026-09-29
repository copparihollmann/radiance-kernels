// Drive the unmodified hpcgarage/spatter serial Configuration implementation.
// Compile this file with the pinned upstream Configuration.cc and Timer.cc.
#include "Configuration.hh"
#include "PatternParser.hh"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using Spatter::Configuration;
using Spatter::Serial;

template <typename T>
static aligned_vector<T> read_words(const char* path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error(std::string("cannot open ") + path);
  const auto bytes = input.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(T)))
    throw std::runtime_error(std::string("invalid word count in ") + path);
  aligned_vector<T> words(static_cast<size_t>(bytes) / sizeof(T));
  input.seekg(0);
  if (bytes && !input.read(reinterpret_cast<char*>(words.data()), bytes))
    throw std::runtime_error(std::string("cannot read ") + path);
  return words;
}

static aligned_vector<size_t> read_pattern(const char* path) {
  const auto words = read_words<uint32_t>(path);
  return aligned_vector<size_t>(words.begin(), words.end());
}

static aligned_vector<double> read_source(const char* path) {
  const auto words = read_words<uint64_t>(path);
  aligned_vector<double> values(words.size());
  static_assert(sizeof(double) == sizeof(uint64_t));
  for (size_t i = 0; i < words.size(); ++i)
    std::memcpy(&values[i], &words[i], sizeof(double));
  return values;
}

static void write_output(const char* path, const aligned_vector<double>& values) {
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error(std::string("cannot open ") + path);
  for (double value : values) {
    uint64_t word;
    std::memcpy(&word, &value, sizeof(word));
    output.write(reinterpret_cast<const char*>(&word), sizeof(word));
  }
  if (!output) throw std::runtime_error(std::string("cannot write ") + path);
}

int main(int argc, char** argv) {
  if (argc == 6 && std::string(argv[1]) == "--parse-pattern") {
    // Expose upstream's own pattern generator for comparison with the
    // Radiance-normalized pattern before executing any kernel.
    std::stringstream text(argv[2]);
    aligned_vector<size_t> pattern;
    size_t delta = std::stoull(argv[3]);
    if (Spatter::pattern_parser(text, pattern, delta)) return 1;
    const size_t limit = std::stoull(argv[4]);
    if (limit && limit < pattern.size()) pattern.resize(limit);
    std::ofstream output(argv[5], std::ios::binary);
    if (!output) return 1;
    for (size_t index : pattern) {
      if (index > UINT32_MAX) return 1;
      uint32_t word = static_cast<uint32_t>(index);
      output.write(reinterpret_cast<const char*>(&word), sizeof(word));
    }
    if (!output) return 1;
    std::cout << delta << '\n';
    return 0;
  }
  // kind count wrap delta delta_gather delta_scatter source_elements
  // output_elements pattern.bin gather.bin scatter.bin source.bin output.bin
  if (argc != 14) {
    std::cerr << "expected kind/count/wrap/deltas/lengths and five file paths\n";
    return 2;
  }
  try {
    const std::string kind(argv[1]);
    const size_t count = std::stoull(argv[2]);
    const size_t wrap = std::stoull(argv[3]);
    const size_t delta = std::stoull(argv[4]);
    const size_t delta_gather = std::stoull(argv[5]);
    const size_t delta_scatter = std::stoull(argv[6]);
    const size_t source_elements = std::stoull(argv[7]);
    const size_t output_elements = std::stoull(argv[8]);
    auto pattern = read_pattern(argv[9]);
    auto gather = read_pattern(argv[10]);
    auto scatter = read_pattern(argv[11]);
    auto source = read_source(argv[12]);
    if (source.size() != source_elements)
      throw std::runtime_error("source length differs from manifest");
    aligned_vector<double> output(output_elements, 0.0);
    aligned_vector<double> sparse, dense, sparse_gather, sparse_scatter;
    if (kind == "gs") {
      sparse_gather.swap(source);
      sparse_scatter.swap(output);
    } else if (kind == "gather" || kind == "multigather") {
      sparse.swap(source);
      dense.swap(output);
    } else if (kind == "scatter" || kind == "multiscatter") {
      dense.swap(source);
      sparse.swap(output);
    } else {
      throw std::runtime_error("unsupported Spatter kind");
    }
    double *dev_sparse = nullptr, *dev_dense = nullptr;
    double *dev_sparse_gather = nullptr, *dev_sparse_scatter = nullptr;
    size_t sparse_size = sparse.size(), dense_size = dense.size();
    size_t sparse_gather_size = sparse_gather.size();
    size_t sparse_scatter_size = sparse_scatter.size();
    aligned_vector<aligned_vector<double>> dense_perthread;
    Configuration<Serial> config(0, "golden", kind, pattern, gather, scatter,
        sparse, dev_sparse, sparse_size,
        sparse_gather, dev_sparse_gather, sparse_gather_size,
        sparse_scatter, dev_sparse_scatter, sparse_scatter_size,
        dense, dense_perthread, dev_dense, dense_size,
        delta, delta_gather, delta_scatter, 0, wrap, count, 1, false, 0);
    if (config.run(false, 0)) throw std::runtime_error("upstream run failed");
    if (kind == "gs") write_output(argv[13], sparse_scatter);
    else if (kind == "gather" || kind == "multigather")
      write_output(argv[13], dense);
    else write_output(argv[13], sparse);
  } catch (const std::exception& error) {
    std::cerr << "upstream golden: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
