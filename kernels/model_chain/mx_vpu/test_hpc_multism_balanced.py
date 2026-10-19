import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from hpc_multism import kernel_source as original_kernel_source
from hpc_multism_balanced import kernel_source


class BalancedLauncherTest(unittest.TestCase):
    def test_stream_launcher_is_unchanged(self):
        self.assertEqual(kernel_source('stream'), original_kernel_source('stream', 4))

    @unittest.skipUnless(shutil.which('g++'), 'requires native C++ compiler')
    def test_generated_callback_assigns_small_owner_set_to_all_four_sms(self):
        # Execute the generated callback against the helper's real flat-index
        # contract. The actual helper/oracle is checked separately on GPU ISA.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'mu_intrinsics.h').write_text('''#pragma once
#include <stdint.h>
#define MU_NUM_CORES 2
inline uint32_t vx_core_id() { return 0; }
inline void mu_barrier(uint32_t, uint32_t) {}
inline void mu_fence() {}
inline void mu_tohost(uint32_t) {}
''')
            (root / 'mu_schedule.h').write_text('''#pragma once
#include <stdint.h>
inline void mu_schedule(void(*)(void*,uint32_t,uint32_t,uint32_t), void*, uint32_t) {}
''')
            (root / 'kernel_verify.h').write_text('')
            (root / 'spatter_body.hpp').write_text('''#pragma once
#include <stdint.h>
#include <cassert>
static uint32_t current_sm, owner_count;
static uint32_t visits[5000], owners[5000], counts[4];
template<uint32_t Clusters>
static void spatter_apply(uint32_t tid, uint32_t tpb, uint32_t cluster) {
  for (uint32_t owner = cluster * tpb + tid; owner < owner_count; owner += Clusters * tpb) {
    ++visits[owner]; owners[owner] = current_sm; ++counts[current_sm];
  }
}
''')
            (root / 'kernel.cpp').write_text(kernel_source('spatter'))
            (root / 'probe.cpp').write_text('''#define main workload_main
#include "kernel.cpp"
#undef main
int main() {
  for (uint32_t n : {256u, 257u, 4102u}) {
    owner_count = n;
    for (auto &v : visits) v = 0;
    for (auto &v : counts) v = 0;
    for (current_sm = 0; current_sm < 4; ++current_sm)
      for (uint32_t tid = 0; tid < 128; ++tid) stage(nullptr, tid, 128, current_sm);
    for (uint32_t i = 0; i < n; ++i) {
      if (visits[i] != 1 || owners[i] != i % 4) return 1;
    }
    for (uint32_t sm = 0; sm < 4; ++sm) if (counts[sm] == 0) return 2;
  }
}
'''.replace('#define main', '#include <initializer_list>\n#define main'))
            executable = root / 'probe'
            subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(root), str(root / 'probe.cpp'),
                            '-o', str(executable)], check=True, capture_output=True)
            subprocess.run([str(executable)], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
