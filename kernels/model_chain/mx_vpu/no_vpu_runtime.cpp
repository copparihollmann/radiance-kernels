// Link this strong definition into an existing generated model to replace the
// weak, out-of-line VPU addition helper. MX matmul objects remain unchanged.
#include "muon_bf16_add.hpp"
namespace model_chain {
__attribute__((noinline))
void mx_vpu_add(const RK_GLOBAL float* left, const RK_GLOBAL float* right,
                RK_GLOBAL float* output, uint32_t count, uint32_t right_width,
                uint32_t tid, uint32_t threads) {
  mx_muon_bf16_add(left, right, output, count, right_width, tid, threads);
}
}
