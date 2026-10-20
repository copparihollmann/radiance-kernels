/* Earliest GPU probe: no stack, runtime, scheduler, or global-memory stores. */
.section .init,"ax"
.globl _start
_start:
  csrr t0, 0xcc2
  slli t0, t0, 6
  li t1, 0x80080
  add t0, t0, t1
  li t1, 0x80038
  lw.shared t2, 0(t1)
  li t3, 0xffff
  and t2, t2, t3
  sw.shared t2, 8(t0)
  sw.shared t2, 12(t0)
  csrr t3, 0xcd0
  sw.shared t3, 16(t0)
  sw.shared t3, 20(t0)
  csrr t3, mhartid
  sw.shared t3, 24(t0)
  sw.shared t3, 28(t0)
  li t3, 0xe001
  sw.shared t3, 0(t0)
  sw.shared t3, 4(t0)
  slli t2, t2, 8
  csrr t3, 0xcc2
  or t2, t2, t3
  li t3, 0xd06e0000
  or t2, t2, t3
  sw.shared t2, 56(t0)
  sw.shared t2, 60(t0)
1: j 1b
