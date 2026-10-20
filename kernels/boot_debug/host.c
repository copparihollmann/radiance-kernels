/* Intentionally self-contained: HTIF uses the standard syscall proxy protocol. */
#include <stdint.h>
#ifndef DEBUG_SMS
#define DEBUG_SMS 1
#endif
#ifndef DEBUG_STAGE
#define DEBUG_STAGE 0
#endif
#ifndef DEBUG_RESULT_ADDRESS
#define DEBUG_RESULT_ADDRESS 0
#endif
#ifndef DEBUG_TIMEOUT_CYCLES
#define DEBUG_TIMEOUT_CYCLES 200000000ull
#endif
/* Separate cache lines keep target and HTIF host stores independent. */
__attribute__((section(".htif"), aligned(64))) volatile uint64_t tohost = 0;
__attribute__((section(".htif"), aligned(64))) volatile uint64_t fromhost = 0;
volatile uint64_t debug_checkpoint;
volatile uint64_t debug_trap_record[3];

static void fence(void) { __asm__ volatile("fence rw,rw" ::: "memory"); }
static uint64_t cycles(void) { uint64_t x; __asm__ volatile("rdcycle %0":"=r"(x)); return x; }
static void puts_htif(const char *s) {
  uint64_t n = 0;
  while (s[n]) ++n;
  /* Linux ABI write(fd=1, buffer, length), as used by libgloss HTIF. */
  volatile uint64_t request[8] __attribute__((aligned(64))) = {64, 1, (uintptr_t)s, n, 0, 0, 0, 0};
  while (tohost) {}
  fromhost = 0;
  fence();
  tohost = (uintptr_t)request;
  while (!fromhost) {}
  fence();
  fromhost = 0;
}
static void hex(uint64_t v) {
  char s[19] = "0x0000000000000000";
  const char *digits = "0123456789abcdef";
  for (unsigned i = 0; i < 16; ++i) s[17-i] = digits[(v >> (4*i)) & 15];
  puts_htif(s);
}
__attribute__((noreturn)) void debug_exit(unsigned code) {
  fence();
  while (tohost) {}
  tohost = ((uint64_t)code << 1) | 1;
  for (;;) __asm__ volatile("nop");
}
__attribute__((noreturn)) void debug_trap(uint64_t cause, uint64_t pc, uint64_t value) {
  debug_trap_record[0] = cause; debug_trap_record[1] = pc; debug_trap_record[2] = value;
  puts_htif("[boot-debug] HOST_TRAP mcause="); hex(cause);
  puts_htif(" mepc="); hex(pc); puts_htif(" mtval="); hex(value); puts_htif("\n");
  debug_exit(3);
}
static volatile uint32_t *pb(unsigned sm, unsigned slot) {
  return (volatile uint32_t *)(0x40080000ull + sm*0x100000ull + slot*8ull);
}
static void put(unsigned sm, unsigned slot, uint32_t v) { pb(sm,slot)[0]=v; pb(sm,slot)[1]=v; }
static void sm_label(unsigned sm) { puts_htif("[boot-debug] sm="); hex(sm); }

int main(void) {
  debug_checkpoint = 1;
  puts_htif("[boot-debug] HOST_BOOT stage="); hex(DEBUG_STAGE);
  puts_htif(" sms="); hex(DEBUG_SMS); puts_htif("\n");
  if (DEBUG_STAGE == 0) { puts_htif("[boot-debug] PASS host-htif\n"); return 0; }
  for (unsigned sm=0; sm<DEBUG_SMS; ++sm) {
    sm_label(sm); puts_htif(" MAILBOX_BEGIN\n");
    debug_checkpoint = 0x100 + sm;
    put(sm, 7, 0xc1d00000u | sm);
    fence();
    if (*pb(sm,7) != (0xc1d00000u | sm)) {
      sm_label(sm); puts_htif(" FAIL mailbox-readback\n"); return 4;
    }
    /* Clear only this test's slots; never touch another SM's address range. */
    for (unsigned slot=16; slot<32; ++slot) put(sm,slot,0);
    sm_label(sm); puts_htif(" MAILBOX_OK\n");
  }
  if (DEBUG_STAGE == 1) { puts_htif("[boot-debug] PASS host-mailboxes\n"); return 0; }
  puts_htif("[boot-debug] RESET_RELEASE_BEGIN\n");
  debug_checkpoint = 0x200;
  fence();
  *(volatile uint32_t *)0x41000000ull = 0;
  fence();
  puts_htif("[boot-debug] RESET_RELEASED\n");
  uint32_t last[DEBUG_SMS][2] = {{0}};
  const uint64_t start = cycles();
  unsigned count = 0;
  do {
    count=0;
    for (unsigned sm=0; sm<DEBUG_SMS; ++sm) for (unsigned core=0; core<2; ++core) {
      const unsigned slot=16+core*8;
      const uint32_t phase=*pb(sm,slot), done=*pb(sm,slot+7);
      if (phase != last[sm][core]) {
        last[sm][core]=phase;
        sm_label(sm); puts_htif(" core="); hex(core); puts_htif(" phase="); hex(phase);
        puts_htif(" resolved="); hex(*pb(sm,slot+1));
        puts_htif(" raw_cluster="); hex(*pb(sm,slot+2));
        puts_htif(" raw_hart="); hex(*pb(sm,slot+3));
        puts_htif(" sp="); hex(*pb(sm,slot+4)); puts_htif("\n");
      }
      if (done == (0xd06e0000u | (sm<<8) | core)) {
        if (*pb(sm,slot+1) != sm || *pb(sm,slot+6)) {
          sm_label(sm); puts_htif(" FAIL device-identity\n"); return 5;
        }
        ++count;
      }
    }
    if (count == DEBUG_SMS*2) break;
  } while (cycles()-start < DEBUG_TIMEOUT_CYCLES);
  if (count != DEBUG_SMS*2) {
    puts_htif("[boot-debug] FAIL timeout completed_cores="); hex(count); puts_htif("\n");
    return 6;
  }
  if (DEBUG_STAGE == 3) {
    puts_htif("[boot-debug] MEMORY_READBACK_BEGIN\n");
    volatile uint32_t *out=(volatile uint32_t *)(0x100000000ull+DEBUG_RESULT_ADDRESS);
    unsigned errors=0;
    for (unsigned sm=0; sm<DEBUG_SMS; ++sm) for (unsigned tid=0; tid<256; ++tid) {
      uint32_t expected=0xa5000000u | (sm<<16) | (tid+1);
      uint32_t actual=out[sm*256+tid];
      if (actual!=expected) {
        if (errors<8) { puts_htif("[boot-debug] mismatch index="); hex(sm*256+tid);
          puts_htif(" actual="); hex(actual); puts_htif(" expected="); hex(expected); puts_htif("\n"); }
        ++errors;
      }
    }
    if (errors) { puts_htif("[boot-debug] FAIL memory errors="); hex(errors); puts_htif("\n"); return 7; }
  }
  debug_checkpoint=0x600d;
  puts_htif("[boot-debug] PASS all-sm elapsed_host_cycles="); hex(cycles()-start); puts_htif("\n");
  /* Device boot warps deliberately park: HTIF exit belongs to the Rocket. */
  return 0;
}
