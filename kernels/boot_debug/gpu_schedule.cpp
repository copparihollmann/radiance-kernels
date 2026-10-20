#include <stdint.h>
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <nightly/cluster_id.h>
#if !defined(MU_CLUSTER_ID_FROM_PRINTBUF) || !MU_CLUSTER_ID_FROM_PRINTBUF
extern "C" __attribute__((noinline)) uint32_t mu_cluster_id(void) {
  uint32_t v;
  asm volatile("csrr %0, 0xcd0":"=r"(v));
  return v;
}
#endif
extern "C" uint32_t __mu_num_warps = 8;
extern "C" { alignas(64) __global uint32_t debug_output[1024] = {}; }
static void put(unsigned offset, uint32_t v) {
  const unsigned address=0x80080+vx_core_id()*64+offset*8;
  asm volatile("sw.shared %0, 0(%1)\nsw.shared %0, 4(%1)" :: "r"(v),"r"(address):"memory");
}
__attribute__((noinline)) void debug_callback(void *arg, uint32_t tid, uint32_t threads, uint32_t block) {
  const uint32_t sm=mu_cluster_id();
  uint32_t sp;
  asm volatile("mv %0, sp":"=r"(sp));
  const uint32_t group=(0x7f000000u-sp)>>17;
  const bool good=sm<DEBUG_SMS && block==sm && threads==256 &&
      (uintptr_t)arg==0x100+sm && group==sm*16+vx_core_id()*8+vx_warp_id();
  if (sm<DEBUG_SMS && tid<256)
    debug_output[sm*256+tid]=(good?0xa5000000u:0xbad00000u)|(sm<<16)|(tid+1);
}
int main() {
  const uint32_t sm=mu_cluster_id();
  uint32_t raw,hart,sp;
  asm volatile("csrr %0, 0xcd0":"=r"(raw));
  asm volatile("csrr %0, mhartid":"=r"(hart));
  asm volatile("mv %0, sp":"=r"(sp));
  put(1,sm); put(2,raw); put(3,hart); put(4,sp); put(0,0xe002);
  mu_schedule(debug_callback,(void *)(uintptr_t)(0x100+sm),8);
  mu_barrier(0,MU_NUM_CORES);
  mu_fence();
  put(0,0xe003);
  put(7,0xd06e0000u|(sm<<8)|vx_core_id());
  for (;;) asm volatile("nop");
}
