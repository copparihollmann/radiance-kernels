#pragma once

#include <stdint.h>

namespace model_chain {

// The graph keeps head reshaping as an explicit GMEM handoff. Row-major
// element order is unchanged by [batch, token, projection] -> [batch, token,
// head, channel], so this stage copies each element to its own output buffer.
inline void reshape_heads(const RK_GLOBAL float* in, RK_GLOBAL float* out,
                          uint32_t count, uint32_t tid, uint32_t threads) {
  for (uint32_t i = tid; i < count; i += threads) out[i] = in[i];
}

}  // namespace model_chain
