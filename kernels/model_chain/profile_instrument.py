"""Add whole-workload hooks to the known frozen model/HPC entry templates."""
import re

PREFIX = '''#include <mu_intrinsics.h>
#include <kernel_verify.h>
#include <nightly/profile_device.h>
#define mu_tohost(code) nightly_profile_complete(code)
'''


def instrument(source, *, gemm=False):
    if 'NIGHTLY_PROFILE_BEGIN(__work_profile)' in source:
        raise ValueError('entry already instrumented')
    if gemm:
        # A generated include overlay instruments nightly_wait_launch and
        # nightly_kernel_end, which precedes the GEMM verification callback.
        return PREFIX + source
    start = source.index('int main()')
    split = source.index('{', start) + 1
    body = source[split:]
    park = re.search(r'if\s*\((?:vx_core_id|mu_hart_id)\(\)\s*!=\s*0\)', body)
    if park is None:
        raise ValueError('missing final core parking boundary')
    last_fence = body.rfind('mu_fence();', 0, park.start())
    if last_fence < 0 or 'mu_schedule(' in body[last_fence:park.start()]:
        raise ValueError('missing final algorithm completion fence')
    finish = last_fence + len('mu_fence();')
    end = '\n  NIGHTLY_PROFILE_END(__work_profile);\n  NIGHTLY_PROFILE_PUBLISH(vx_core_id(), __work_profile);'
    body = body[:finish] + end + body[finish:]
    # SmolVLA's input initialization is setup, not model work.
    first = body.index('mu_schedule(')
    if body[first:].startswith('mu_schedule(init_inputs,'):
        fence = body.index('mu_fence();', first) + len('mu_fence();')
        insertion = fence
    else:
        insertion = first
    begin = '\n  NightlyProfileSpan __work_profile;\n  nightly_profile_wait_launch();\n  NIGHTLY_PROFILE_BEGIN(__work_profile);\n  '
    return PREFIX + source[:split] + body[:insertion] + begin + body[insertion:]


def instrument_nightly_device(source):
    anchor = 'nightly_pb_put(NIGHTLY_PB_START + (uint32_t)vx_core_id(), nightly_mcycle());'
    if source.count(anchor) != 1:
        raise ValueError('nightly launch template changed')
    source = source.replace(anchor, anchor + '\n  nightly_profile_begin_mailbox();')
    anchor = '''if ((tid_in_threadblock % MU_NUM_THREADS) == 0)
      nightly_pb_put(NIGHTLY_PB_END + (uint32_t)vx_core_id(), nightly_mcycle());'''
    if source.count(anchor) != 1:
        raise ValueError('nightly end template changed')
    return source.replace(anchor, '''if ((tid_in_threadblock % MU_NUM_THREADS) == 0) {
      nightly_profile_end_mailbox();
      nightly_pb_put(NIGHTLY_PB_END + (uint32_t)vx_core_id(), nightly_mcycle());
    }''')
