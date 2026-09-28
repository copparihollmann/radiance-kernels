# Spatter on Radiance

The [workload result summary](../WORKLOAD_RESULTS.md) combines this kernel's
cycles with the standalone STREAM measurements.

This kernel lives in `radiance-kernels` and builds with `../common.mk`, the same
Muon and RV64 host infrastructure used by the other kernels. It maps the five
[Spatter](https://github.com/hpcgarage/spatter) transfer families: Gather,
Scatter, GatherScatter (`GS`), MultiGather, and MultiScatter. The source JSON
may come from Spatter's standard suite or the [LANL xRAGE traces](https://github.com/lanl/spatter).
No FPGA bitstream is required; the fused ELF runs on the Radiance SoC RTL
simulator or Cyclotron model.
Spatter's `gpu-stream.json` contains five Spatter address-transfer families;
the four standalone STREAM operations live in [../stream](../stream/README.md).

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
a Gather -> Scatter chain with its fused GS equivalent. `run_chain.py` builds
one ELF with a materialized dense intermediate, two Muon schedules, and a
barrier between stages. It accepts different stage counts when their
intermediate lengths match. Repeated Scatter destinations require the
`ordered` policy, which assigns each destination to one lane:

```sh
python3 run_chain.py composition-smoke.json 2 composition-smoke.json 1 \
  --out runs/materialized-chain
python3 run_chain.py composition-ordered.json 0 composition-ordered.json 1 \
  --out runs/materialized-chain-ordered
```

General chains of arbitrary Spatter families remain to be mapped. Ordered
Scatter serializes conflicting writes within each destination and has
different performance from an atomic Scatter implementation.

For a Gather followed by Scatter with matching count, pattern length, and
`wrap`, `tools/spatter-compose.py` bypasses the dense intermediate and emits a
fused GS case. For `wrap < count`, the fused source address selects the last
Gather iteration that wrote the dense slot read by Scatter. The emitted
`source-tag` keeps the generated source values identical to the original
Gather. This fused execution has its own cycle count; it is not the sum of two
separately launched kernels. It can be built and simulated like any other
Spatter case:

```sh
python3 tools/spatter-compose.py composition-smoke.json 0 \
  composition-smoke.json 1 --out runs/composed-suite.json
python3 run.py --suite runs/composed-suite.json --case 0 \
  --out runs/composed --build-only
```

`composition-smoke.json` cases 2 and 3 exercise `count=3, wrap=2`. The
generated GS case records `gather-final-wrap` so its address map reflects the
final writer of each intermediate slot. `stitch_reference` independently
models the two-stage materialization in software. The fused and materialized
ELFs have separate cycle measurements.

This implements Spatter's documented transfer equations and its serial
backend's Gather writes. The upstream CUDA Gather and MultiGather kernels
instead consume each loaded value through a conditional write to `dense[0]`;
their usual traffic is effectively read-only. Radiance writes the mapped dense
array on every transfer so its result can be checked end to end. Consequently,
Radiance Gather/MultiGather payload and cycle figures are not directly
comparable with the upstream CUDA throughput figures.

## Build

Set `LLVM_MUON`, `RISCV_TOOLCHAIN_PATH`, `RISCV64_TOOLCHAIN_PATH`, and `RISCV`
for the installed toolchains. Set `GEMMINI_SW_PATH` to the initialized
`lib/mxgemmini` submodule. This checkout's `lib` and `soc` are used by default.
The available older Muon compiler requires `MU_STACK_WORD_STRIDE=1` for both
`lib/libmuonrt.a` and this kernel; a newer compiler can use the repo default.
For the older compiler, build the runtime first and set that environment
variable while building the kernel. The Makefile adds Muon's bundled libc++
headers.

From a fresh `kernels/spatter` checkout, plain `make` prepares a small Gather
case. Subsequent `make` calls reuse the last generated case. To prepare and
build a particular JSON case:

```sh
python3 run.py --suite smoke.json --case 0 --out runs/smoke-0 --build-only
```

`run.py` owns the shared `generated/` directory, so build cases sequentially.
`--prepare-only` emits the plan, data, config and manifest without compiling.
`--count N` makes a clearly labeled scaled run while preserving the input
pattern. It is not performance-equivalent to the original repetition count.
The output directory contains `kernel.soc.elf`, `result.json`, and `build.log`.

Scatter, GS, and MultiScatter can use `--collision-policy ordered` when
destinations repeat. The generator groups transfers by destination in source
order; one GPU lane owns each destination, so all 64-bit stores complete
without another lane writing the same value concurrently. This gives a
deterministic final output and permits a full digest check:

```sh
python3 run.py --suite exploratory-smoke.json --case 0 \
  --collision-policy ordered --out runs/ordered-overlap --build-only
```

The ordered path executes every transfer but adds a generated task schedule
and serializes conflicts. Its cycles describe that mapping, not the upstream
CUDA `atomicExch` implementation. The default parallel path retains its
exploratory status when destinations overlap.

For the five standard GPU STREAM cases, pass the tracked
`inputs/standard-suite/basic-tests/gpu-stream.json` and case indices 0 through
4. The tracked `inputs/standard-suite/app-traces/lulesh.json` and
`inputs/standard-suite/app-traces/amg_gpu.json` preserve the other small
upstream decks used in this evaluation.
For LANL `datafiles/xrage/asteroid/spatter.json`, pattern 5 is case 4 and
pattern 9 is case 8. Original-size xRAGE5 has 8,368,968 gather addresses;
xRAGE9 has 6,664,304 scatter addresses. The input JSON used for the evaluation
has SHA-256 `7325525ada0dacb6e1206717d242f6721b6d8da77718506fd909444c388f7733`.
For xRAGE9, `--collision-policy ordered` creates a complete-output-checkable
mapping of the repeated destinations. Keep its results separate from both the
default parallel mapping and Spatter's CUDA atomic result.

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
in either case. xRAGE9 has duplicate destinations. The default parallel
mapping has no atomic scatter, so its RTL check is a nonzero output probe plus
guards and its result is exploratory. The ordered mapping checks a complete
serial-order digest, but uses a generated schedule and is not an atomic
throughput result. Neither result should be compared directly with the
published atomic xRAGE9 GPU result. The logical payload metric counts one
8-byte read and one
8-byte write per transfer, excluding pattern traffic; no GPU frequency or
HBM timing calibration is assumed. See [evaluation/README.md](evaluation/README.md)
for measured cases and provenance.

Run `python3 -m unittest -v test_run.py` for parser, address-map, and software
composition checks.
