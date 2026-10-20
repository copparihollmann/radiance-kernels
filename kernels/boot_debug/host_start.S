/* One Rocket host. No libc, TLS, constructors, or floating-point state. */
.section .text.start,"ax"
.globl _start
_start:
  .option push
  .option norelax
  la gp, __global_pointer$
  .option pop
  la sp, __stack_top
  la t0, host_trap
  csrw mtvec, t0
  la t0, __bss_start
  la t1, __bss_end
1:
  bgeu t0, t1, 2f
  sd zero, 0(t0)
  addi t0, t0, 8
  j 1b
2:
  call main
  tail debug_exit
.balign 4
host_trap:
  /* Reset the stack: the exception may have been a bad stack access. */
  la sp, __stack_top
  csrr a0, mcause
  csrr a1, mepc
  csrr a2, mtval
  tail debug_trap
