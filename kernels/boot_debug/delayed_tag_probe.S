/* Regression-only entry wrapper. Core 1 emulates a late host mailbox write.
 * Slot 60 holds the physical tag, provisioned independently per physical SM.
 * Core 0 can reach runtime startup while slot 7 is still unset. */
.section .init,"ax"
.globl _start
_start:
  csrr t0, 0xcc2
  beqz t0, 2f
  li t0, 2000
1:
  addi t0, t0, -1
  bnez t0, 1b
  li t0, 0x801e0
  lw.shared t1, 0(t0)
  li t0, 0x80038
  sw.shared t1, 0(t0)
  sw.shared t1, 4(t0)
2:
  j debug_runtime_start
