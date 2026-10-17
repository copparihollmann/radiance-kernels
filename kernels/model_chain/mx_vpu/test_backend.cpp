#include "mx_vpu/backend.hpp"
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  FILE* in = std::fopen(argv[1], "rb");
  if (!in) return 2;
  uint32_t h[6]; // M,K,N,storage,weight bytes,scale count
  if (std::fread(h, sizeof h, 1, in) != 1) return 2;
  std::vector<float> x(h[0]*h[1]), scales(h[5]), skip(h[0]*h[2]), y(skip.size()), z(skip.size());
  std::vector<uint8_t> w(h[4]);
  if (std::fread(x.data(), 4, x.size(), in) != x.size() ||
      std::fread(w.data(), 1, w.size(), in) != w.size() ||
      std::fread(scales.data(), 4, scales.size(), in) != scales.size() ||
      std::fread(skip.data(), 4, skip.size(), in) != skip.size()) return 2;
  std::fclose(in);
  model_chain::mx_vpu_linear(x.data(), w.data(), scales.data(), y.data(),
      model_chain::mx_model::Storage(h[3]), h[0], h[1], h[2], 0, 1);
  model_chain::mx_vpu_add(y.data(), skip.data(), z.data(), z.size(), z.size(), 0, 1);
  FILE* out = std::fopen(argv[2], "wb");
  if (!out) return 2;
  bool ok = std::fwrite(y.data(), 4, y.size(), out) == y.size() &&
            std::fwrite(z.data(), 4, z.size(), out) == z.size();
  return std::fclose(out) || !ok;
}
