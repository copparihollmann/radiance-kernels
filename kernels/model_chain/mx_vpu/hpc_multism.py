"""Concurrent SM entry point for the existing partitioned HPC kernels.

mu_schedule passes the real cluster ID as its threadblock argument. Each SM
owns one partition; ordered scatter keeps all writes to a destination with
that destination's owner. No cross-cluster barrier is needed between owners.
"""


def kernel_source(family: str, clusters: int = 2) -> str:
    if family not in ('stream', 'spatter') or clusters != 2:
        raise ValueError('qualified launcher supports STREAM/Spatter on two SMs')
    header = 'stream_ops.hpp' if family == 'stream' else 'spatter_body.hpp'
    return f'''#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>
#include "kernel_verify.h"
#include "{header}"
extern "C" uint32_t __mu_num_warps = 4;
static void stage(void *, uint32_t tid, uint32_t tpb, uint32_t cluster) {{
  {family}_apply<{clusters}>(tid, tpb, cluster);
}}
int main() {{
  mu_schedule(stage, nullptr, 4);
  mu_barrier(0, MU_NUM_CORES);
  mu_fence();
  // Every cluster reports through its local core 0. mhartid is global.
  if (vx_core_id() != 0) {{ for (;;) {{}} }}
  mu_tohost(0u);
  return 0;
}}
'''
