OUTPUT_ARCH(riscv)
ENTRY(_start)
PHDRS {
  text PT_LOAD FLAGS(5);
  data PT_LOAD FLAGS(6);
  /* GPU_PHDRS */
}
SECTIONS {
  . = 0x80000000;
  .text : { KEEP(*(.text.start)) *(.text .text.*) } :text
  .rodata : { *(.rodata .rodata.*) *(.srodata .srodata.*) } :text
  . = ALIGN(4096);
  .htif : { KEEP(*(.htif)) } :data
  .data : { *(.data .data.*) *(.sdata .sdata.*) } :data
  __global_pointer$ = . + 0x800;
  . = ALIGN(8);
  __bss_start = .;
  .bss (NOLOAD) : { *(.bss .bss.*) *(.sbss .sbss.*) *(COMMON) . = ALIGN(8); } :data
  __bss_end = .;
  _end = .;
  ASSERT(_end < 0x80070000, "host overlaps reserved stack")
  __stack_bottom = 0x80070000;
  __stack_top = 0x80080000;
  /* GPU_SECTIONS */
  /DISCARD/ : { *(.eh_frame*) *(.comment) }
}
