#include "fallback_abi.hpp"

namespace model_chain {
extern "C" __attribute__((noinline)) void rk_fallback_call(
    const RK_GLOBAL FallbackDescriptor* descriptor, void* context,
    uint32_t tid, uint32_t threads) {
  if (!fallback_dispatch(descriptor,(uintptr_t)context,tid,threads)) __builtin_trap();
}
}
