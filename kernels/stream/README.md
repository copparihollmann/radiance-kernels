# STREAM on Radiance

The [workload result summary](../README.md) combines these cycles
with the Spatter measurements.

The four independent STREAM operations use float32 arrays: Copy `C=A`, Scale
`B=2*C`, Add `C=A+B`, and Triad `A=B+2*C`. They build through `../common.mk`
and run on the same one-cluster Radiance SoC as `kernels/spatter`. The generated
inputs are exact small integers in float32, so the expected output digest does
not depend on rounding differences. One operation runs per ELF; a complete
four-stage STREAM sequence would use a different data dependency and timing
contract.

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

Build cases sequentially because they share `generated/`. Each output has a
manifest, build log, and fused ELF. The RV64 host checks a complete digest for
up to 1024 elements and 64 samples for larger arrays; Cyclotron can check the
complete output. `output_elements` in the manifest is the number of 64-bit
pairs passed to Spatter's reusable Cyclotron checker; `stream_elements` is the
actual number of float32 values.

For a complete host check on a larger array, add `--full-host-check`. Add
`--host-timing` to print `HOST_RELEASE_TO_DONE_CYCLES` in the guest UART. This
counts RV64 host cycles from just before Muon reset release until the
all-finished register is observed. It excludes the later digest/readback and
most boot and input setup, but includes MMIO, launch and completion polling.
The interval is not a per-core GPU cycle count; compare it with other runs
only after recording the host clock domain and hardware revision. The flag
changes only the RV64 host; the RV32 Muon image was checked byte-for-byte
against the untimed 256-element Copy smoke build.

Use `../spatter/tools/spatter-verilator-run.py` and
`../spatter/tools/spatter-cyclotron-run.py` with `--source-root runs` and a
separate output root to run an ELF. The shared validator and summarizer produce
the same cycle and logical-payload tables as the Spatter kernels. Logical
payload counts two float32 transfers per element for Copy/Scale and three for
Add/Triad; it excludes cache-line amplification. No GPU clock or calibrated
HBM bandwidth is assumed.

See [evaluation/README.md](evaluation/README.md) for the measured
one-million-element timing-model cycle and memory counters.
