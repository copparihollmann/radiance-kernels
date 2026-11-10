# STREAM on Radiance

The four independent STREAM operations use float32 arrays: Copy `C=A`, Scale
`B=3*C`, Add `C=A+B`, and Triad `A=B+3*C`, matching the elemental equations
and scalar in STREAM v5.10. They build through `../common.mk`
and run on the same Radiance SoC as `kernels/spatter`. The generated
inputs are exact small integers in float32, so the expected output digest does
not depend on rounding differences. One operation runs per ELF; a complete
four-stage, ten-iteration STREAM sequence would use a different data dependency
and timing contract. These results must therefore be reported as independent
STREAM elemental kernels, not as an official STREAM bandwidth score.

Build each case with `LLVM_MUON`, `RISCV_TOOLCHAIN_PATH`,
`RISCV64_TOOLCHAIN_PATH`, `RISCV`, and `MU_STACK_WORD_STRIDE` set for the same
toolchain as the runtime library. The installed older compiler requires
`MU_STACK_WORD_STRIDE=1`. On a fresh checkout, plain `make` creates a
256-element Copy smoke input; later calls reuse the last generated case.

```sh
python3 run.py --kind copy --elements 65536 --out runs/copy-65536
python3 run.py --kind scale --elements 65536 --out runs/scale-65536
python3 run.py --kind add --elements 65536 --out runs/add-65536
python3 run.py --kind triad --elements 65536 --out runs/triad-65536
```

`--scalar` is recorded in every manifest. Its default is the STREAM v5.10
value `3.0`; a different value is deliberately labelled `stream-like` rather
than equation-equivalent. The full campaign defaults to 1,048,576 elements so
the measured region is not dominated by the former 4,102-element smoke size.

Pass `--preset 1sm_pipe`, `--preset 2sm_pipe`, or `--preset 4sm_pipe` to build for that many physical SMs.
Each SM owns a contiguous range aligned to a 64-byte cache line. The host tags
every physical SM before releasing reset, and the kernel reports `kernel_cycles`
from the earliest core start through the latest post-fence completion stamp.
Use an ELF only with a hardware configuration having at least that many SMs.

`--warps-per-core {1,2,4,8}` exposes the Muon occupancy as an explicit recorded
tuning parameter; the default remains 4. At the fixed 4,102-element campaign
size, lower occupancy raises functional 2-to-4 scaling (1.72x at one warp and
1.56x at two warps versus 1.38x at four) only by making both absolute runtimes
slower. Generic-DRAM timing similarly favors higher occupancy in absolute
cycles: eight warps reaches 25,629 cycles at 4 SM versus 25,893 at four warps,
while its 2-to-4 ratio is approximately 1x because the shared queue saturates.
Do not select occupancy by speedup ratio alone; retain the fastest absolute
variant and confirm it on the matching FPGA memory model.

Build cases sequentially because they share `generated/`. Each output has a
manifest, build log, and fused ELF. The RV64 host checks a complete digest for
up to 1024 elements and 64 samples for larger arrays; Cyclotron can check the
complete output. `output_elements` in the manifest is the number of 64-bit
pairs passed to Spatter's reusable Cyclotron checker; `stream_elements` is the
actual number of float32 values.

Set `MU_ADDR_HASH=1` for the newer hashed `RadianceHBMConfig` when loading
DRAM directly with `+loadmem` or FireSim LoadMem; the hash parameters must
match the RTL. Use `MU_ADDR_HASH=0` for the earlier unhashed image or a TSI
loader. The Makefile rebuilds the fused ELF when the hash setting changes.
The results below remain tied to their recorded hardware image and ELF.

For a complete host check on a larger array, add `--full-host-check`. Add
`--host-timing` to print `HOST_RELEASE_TO_DONE_CYCLES` in the guest UART. This
counts RV64 host cycles from just before Muon reset release until the
all-finished register is observed. It excludes the later digest/readback and
most boot and input setup, but includes MMIO, launch and completion polling.
The interval is not a per-core GPU cycle count; compare it with other runs
only after recording the host clock domain and hardware revision. The flag
changes only the RV64 host; the RV32 Muon image was checked byte-for-byte
against the untimed 256-element Copy smoke build.

The end-of-kernel drain protocol is no longer a per-run flag.  It is part of
the preset (see `../hpc.mk`): the device publishes END through global memory,
fences one core at a time, then spins until the host resets it.  That
combination exists because on FPGA a core that returns through the tile-finished
flush can leave dirty output lines behind, and because a workload that still
lost a line needed each worker warp fenced in sequence before the cache drains.
Override a single knob on the command line (`make HPC_ROBUST_SERIAL_FENCE=0`)
only for a deliberate experiment; every recorded cycle count assumes the
defaults.

Use `--diagnostic-readback` to print the actual digest and all guard words on
failure; that flag changes only the RV64 host checker and no elemental
equation.

Logical payload counts two float32 transfers per element for Copy/Scale and three for
Add/Triad; it excludes cache-line amplification. No GPU clock or calibrated
HBM bandwidth is assumed.

