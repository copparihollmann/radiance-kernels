/* Integer-only Rocket host for the profiling handoff; no libc startup. */
#include <stdint.h>
#include "profile_run.h"

__attribute__((section(".htif"), aligned(64))) volatile uint64_t tohost;
__attribute__((section(".htif"), aligned(64))) volatile uint64_t fromhost;
static char output_buffer[4096];
static unsigned output_size;
static inline void fence() { asm volatile("fence rw,rw" ::: "memory"); }
static inline uint64_t host_cycle() { uint64_t v; asm volatile("rdcycle %0":"=r"(v)); return v; }
static inline uint64_t host_instructions() { uint64_t v; asm volatile("rdinstret %0":"=r"(v)); return v; }
static void nh_flush() {
  if (!output_size) return;
  volatile uint64_t request[8] __attribute__((aligned(64))) =
      {64, 1, (uintptr_t)output_buffer, output_size, 0, 0, 0, 0};
  while (tohost) {}
  fromhost = 0; fence(); tohost = (uintptr_t)request;
  while (!fromhost) {}
  fence(); fromhost = 0; output_size = 0;
}
static void nh_putc(char v) { if (output_size == sizeof(output_buffer)) nh_flush(); output_buffer[output_size++] = v; }
static void nh_puts(const char *s) { while (*s) nh_putc(*s++); }
static void nh_putu(uint64_t value) {
  char digits[24]; unsigned n = 0;
  do { digits[n++] = char('0' + value % 10); value /= 10; } while (value);
  while (n) nh_putc(digits[--n]);
}
static uint32_t nightly_pb_read(unsigned sm, unsigned slot) {
  return *(volatile uint32_t *)(0x40080000ull + uint64_t(sm) * 0x100000ull + uint64_t(slot) * 8);
}
static void nightly_pb_write(unsigned sm, unsigned slot, uint32_t value) {
  *(volatile uint64_t *)(0x40080000ull + uint64_t(sm) * 0x100000ull + uint64_t(slot) * 8) =
      uint64_t(value) | (uint64_t(value) << 32);
}
#include <nightly/profile_host.h>

extern "C" __attribute__((noreturn)) void debug_exit(unsigned code) {
  nh_flush(); fence(); while (tohost) {}
  tohost = (uint64_t(code) << 1) | 1;
  for (;;) asm volatile("nop");
}
extern "C" __attribute__((noreturn)) void debug_trap(uint64_t cause, uint64_t pc, uint64_t value) {
  nh_puts("RADIANCE_TRAP {\"cause\":"); nh_putu(cause);
  nh_puts(",\"pc\":"); nh_putu(pc); nh_puts(",\"value\":"); nh_putu(value); nh_puts("}\n");
  debug_exit(3);
}

static void print_context() {
  nh_puts("RADIANCE_CONTEXT {\"schema\":1,\"workload\":\"" PROFILE_WORKLOAD
          "\",\"device_elf_sha256\":\"" PROFILE_DEVICE_SHA "\",\"sm_count\":");
  nh_putu(PROFILE_SMS);
  nh_puts(",\"cores_per_sm\":2,\"backend\":\"firesim\",\"memory_model\":\"operator-configured\","
          "\"clock_domain\":\"muon\",\"common_cycle_epoch\":true,"
          "\"instruction_counter_semantics\":\"issued_warp_instructions\","
          "\"peak_issue_per_core_per_cycle\":1,\"mx_useful_macs\":");
  nh_putu(PROFILE_MX_MACS);
  nh_puts(",\"mx_instances\":"); nh_putu(PROFILE_SMS);
  nh_puts(",\"mx_peak_macs_per_cycle\":256,\"mx_cycles_per_muon_cycle\":null,"
          "\"sampling\":\"boundary-csr-reads\"}\n");
}

extern "C" int main() {
  const uint64_t setup_start = host_cycle(), setup_inst_start = host_instructions();
  nh_puts("[radiance] HOST_BOOT workload=" PROFILE_WORKLOAD " sms="); nh_putu(PROFILE_SMS); nh_puts("\n"); nh_flush();
  nightly_profile_host_clear(PROFILE_SMS);
  for (unsigned sm = 0; sm < PROFILE_SMS; ++sm) {
    nightly_pb_write(sm, 58, 0); nightly_pb_write(sm, 59, 0);
    nightly_pb_write(sm, 2, 0); nightly_pb_write(sm, 3, 0);
    nightly_pb_write(sm, 4, 0); nightly_pb_write(sm, 5, 0);
    nightly_pb_write(sm, 7, 0xc1d00000u | sm);
  }
  for (unsigned i = 0; i < sizeof(profile_initial_words) / sizeof(profile_initial_words[0]); ++i) {
    const auto &v = profile_initial_words[i];
    if (v.slot != 7) nightly_pb_write(v.sm, v.slot, v.value);
  }
  for (unsigned sm = 0; sm < PROFILE_SMS; ++sm) nightly_pb_write(sm, 0, 0x6e1a0c11u);
  fence();
  const uint64_t setup_end = host_cycle(), setup_inst_end = host_instructions();
  nh_puts("[radiance] GPU_LAUNCH\n"); nh_flush();
  const uint64_t wait_start = host_cycle(), wait_inst_start = host_instructions();
  *(volatile uint32_t *)0x41000000ull = 0; fence();
  uint64_t work_wait_end = 0, work_wait_inst_end = 0, next_poll = 0;
  bool reported = false, verification_released = !PROFILE_GEMM_HANDSHAKE;
  for (;;) {
    const uint64_t now = host_cycle();
    if (now - wait_start > (1ull << 48)) {
      nh_puts("RADIANCE_FAILURE {\"reason\":\"timeout\"}\n"); debug_exit(6);
    }
    if (now < next_poll) continue;
    next_poll = now + 65536;
    if (!verification_released) {
      bool all_end = true;
      for (unsigned sm = 0; sm < PROFILE_SMS; ++sm)
        all_end &= nightly_pb_read(sm, 4) != 0 && nightly_pb_read(sm, 5) != 0;
      if (all_end) {
        for (unsigned sm = 0; sm < PROFILE_SMS; ++sm) nightly_pb_write(sm, 24, 0x60c0ffeeu);
        fence(); verification_released = true;
      }
    }
    if (!reported && nightly_profile_host_ready(PROFILE_SMS)) {
      work_wait_end = host_cycle(); work_wait_inst_end = host_instructions();
      print_context(); nightly_profile_host_report(PROFILE_SMS); reported = true;
    }
    unsigned done = 0;
    for (unsigned sm = 0; sm < PROFILE_SMS; ++sm) {
      if (nightly_pb_read(sm, 59) == 0x50444f4eu) {
        const uint32_t code = nightly_pb_read(sm, 58);
        if (code) { nh_puts("RADIANCE_FAILURE {\"sm\":"); nh_putu(sm); nh_puts(",\"device_code\":"); nh_putu(code); nh_puts("}\n"); debug_exit(7); }
        ++done;
      }
    }
    if (reported && done == PROFILE_SMS) break;
  }
  nh_puts("RADIANCE_HOST {\"setup_cycles\":"); nh_putu(setup_end - setup_start);
  nh_puts(",\"setup_instructions\":"); nh_putu(setup_inst_end - setup_inst_start);
  nh_puts(",\"wait_cycles\":"); nh_putu(work_wait_end - wait_start);
  nh_puts(",\"wait_instructions\":"); nh_putu(work_wait_inst_end - wait_inst_start); nh_puts("}\n");
  nh_puts("RADIANCE_DONE {\"workload\":\"" PROFILE_WORKLOAD "\",\"device_code\":0,\"sms\":"); nh_putu(PROFILE_SMS); nh_puts("}\n"); nh_flush();
#ifdef PROFILE_PARK_FOR_READBACK
  for (;;) asm volatile("nop");
#else
  return 0;
#endif
}
