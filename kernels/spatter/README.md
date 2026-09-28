# Spatter on Radiance

This kernel lives in `radiance-kernels` and builds with `../common.mk`, the same
Muon and RV64 host infrastructure used by the other kernels. It maps the five
[Spatter](https://github.com/hpcgarage/spatter) transfer families: Gather,
Scatter, GatherScatter (`GS`), MultiGather, and MultiScatter. The source JSON
may come from Spatter's standard suite or the [LANL xRAGE traces](https://github.com/lanl/spatter).
No FPGA bitstream is required; the fused ELF runs on the Radiance SoC RTL
simulator or Cyclotron model.

## Address decomposition and composition

`plan.py` describes every family as one source address map, one destination
address map, and a schedule. An address map is dense, a direct pattern lookup,
or a nested lookup (`pattern[inner[j]]`), followed by a per-iteration delta.
`spatter_ops.hpp` implements those primitives for the Muon kernel in
`kernel.cpp`. `run.py` emits the selected maps as `generated/plan.json` and in
its run manifest. Each task transfers one 64-bit value as two volatile 32-bit
loads and two volatile 32-bit stores.

Gather and MultiGather assign each dense output slot to one owner, preserving
Spatter's last-iteration result when `wrap` reuses a slot. Other families
schedule independent transfers over `(iteration, pattern entry)`. The software
reference can compose stages with `stitch_reference(stages, source)`: each stage
materializes its output before the next stage reads it. The unit check compares
a Gather -> Scatter chain with its fused GS equivalent. The SoC runner currently
launches one stage per ELF; on-device multi-stage orchestration is a subsequent
integration step. Composition needs matching intermediate array lengths and a
barrier between stages. Repeated Scatter destinations also need atomics or an
explicit ordering rule for deterministic results.

## Build

Set `LLVM_MUON`, `RISCV_TOOLCHAIN_PATH`, `RISCV64_TOOLCHAIN_PATH`, and `RISCV`
for the installed toolchains. Set `GEMMINI_SW_PATH` to the initialized
`lib/mxgemmini` submodule. This checkout's `lib` and `soc` are used by default.
The available older Muon compiler requires `MU_STACK_WORD_STRIDE=1` for both
`lib/libmuonrt.a` and this kernel; a newer compiler can use the repo default.
For the older compiler, build the runtime first and set that environment
variable while building the kernel. The Makefile adds Muon's bundled libc++
headers.

From `kernels/spatter`, plain `make` prepares a small Gather case. To prepare
and build a particular JSON case:

```sh
python3 run.py --suite smoke.json --case 0 --out runs/smoke-0 --build-only
```

`run.py` owns the shared `generated/` directory, so build cases sequentially.
`--prepare-only` emits the plan, data, config and manifest without compiling.
`--count N` makes a clearly labeled scaled run while preserving the input
pattern. It is not performance-equivalent to the original repetition count.
The output directory contains `kernel.soc.elf`, `result.json`, and `build.log`.

For the five standard GPU STREAM cases, pass
`standard-suite/basic-tests/gpu-stream.json` and case indices 0 through 4.
For LANL `datafiles/xrage/asteroid/spatter.json`, pattern 5 is case 4 and
pattern 9 is case 8. Original-size xRAGE5 has 8,368,968 gather addresses;
xRAGE9 has 6,664,304 scatter addresses. The input JSON used for the evaluation
has SHA-256 `7325525ada0dacb6e1206717d242f6721b6d8da77718506fd909444c388f7733`.

## Simulate and validate

Use the full `RadianceSingleClusterConfig` SoC simulator. `run.py --simv`
accepts a VCS simulator binary and checks the RV64 host result. Where VCS is
unavailable, use the bundled Verilator runner on built ELFs:

```sh
python3 tools/spatter-verilator-run.py --sim "$SIM" \
  --source-root runs --output-root runs/rtl smoke-0
```

`SIM` is the built `simulator-chipyard.harness-RadianceSingleClusterConfig`
(or multithreaded equivalent). The runner raises the process stack limit for
the Verilated model, preserves the ELF hash, and records GPU cycles and status.
For the full-output Cyclotron check, copy `tools/spatter_check.rs` to the
Cyclotron checkout's `src/bin/`, build it with `cargo build --release --bin
spatter_check`, and run:

```sh
python3 tools/spatter-cyclotron-run.py --cyclotron "$CYCLOTRON" \
  --source-root runs --output-root runs/model smoke-0
```

Copy `tools/spatter-config.toml` to the Cyclotron checkout root. Its relative
timing includes resolve there. The Cyclotron runner checks that the installed
checker and config match this repo's sources.
`cyclotron-no-trace.patch` allows the full SoC to skip Cyclotron's large SQLite
trace; apply it to the matching Radiance Cyclotron checkout and rebuild when
running large inputs. `tools/spatter-audit-suite.py SUITE_DIR` reports which
upstream configurations fit the current GPU address window.
`tools/spatter-summarize.py`
converts completed run directories to CSV.

For deterministic cases, the RV64 host checks the full digest when the output
has at most 1024 elements, or 64 spaced samples for larger outputs; the
Cyclotron checker validates the complete output. The host checks guard regions
in either case. xRAGE9 has duplicate destinations; the current kernel has no
atomic scatter, so its RTL check is a nonzero output probe plus guards and its
result is exploratory. It must not be compared with the published atomic
xRAGE9 GPU result. The logical payload metric counts one 8-byte read and one
8-byte write per transfer, excluding pattern traffic; no GPU frequency or
HBM timing calibration is assumed. See [evaluation/README.md](evaluation/README.md)
for measured cases and provenance.

Run `python3 -m unittest -v test_run.py` for parser, address-map, and software
composition checks.
