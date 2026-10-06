# Spatter on Radiance

The [workload result summary](../README.md) combines this kernel's
cycles with the standalone STREAM measurements.

This kernel lives in `radiance-kernels` and builds with `../common.mk`, the same
Muon and RV64 host infrastructure used by the other kernels. It maps the five
[Spatter](https://github.com/hpcgarage/spatter) transfer families: Gather,
Scatter, GatherScatter (`GS`), MultiGather, and MultiScatter. The source JSON
may come from Spatter's standard suite or the [LANL xRAGE traces](https://github.com/lanl/spatter).
The fused ELF runs on the Radiance SoC RTL simulator or Cyclotron model.
A one-cluster U250 image has also run the selected cases recorded in the
[FireSim evaluation](../evaluation/firesim/README.md).
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
one ELF with a materialized dense intermediate. Both stages run in one Muon
schedule, with an all-warp barrier before Scatter reads the intermediate.
The separate-schedule mapping lost Scatter worker writes in the full-size timing
model, so the stage boundary is explicit inside the kernel. The builder
accepts different stage counts when their intermediate lengths match.
Repeated Scatter destinations require the
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

The tracked `inputs/composed-gpu-stream.json` is the fusion of standard GPU
STREAM Gather case 0 and Scatter case 1. Build that case and its materialized
counterpart from the same original decks with:

```sh
python3 run.py --suite inputs/composed-gpu-stream.json --case 0 \
  --out runs/composed-gpu-stream-current --build-only
python3 run_chain.py inputs/standard-suite/basic-tests/gpu-stream.json 0 \
  inputs/standard-suite/basic-tests/gpu-stream.json 1 \
  --out runs/chain-gpu-stream-no-fence
```

Their complete output digests match. The
[full-size comparison](evaluation/composition-fullsize-model-results.csv) and
[memory counters](evaluation/composition-fullsize-memory.csv) retain the model
results; `run_chain.py` also records both stage maps and the intermediate
length in its manifest.

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

The RV64 host clears the complete output before releasing Muon reset. This is
required for U250 FireSim: the fused RV32 BSS output has no initialized ELF
payload, and unwritten Scatter destinations otherwise retain stale memory.
Pass `--full-host-check` to make the host digest every output word and both
guards after the kernel finishes. That mode provides stronger FPGA
correctness evidence but adds readback time to whole-program target cycles;
see the [U250 follow-up](../evaluation/firesim/output-initialization/README.md).
Pass `--host-timing` to print `HOST_RELEASE_TO_DONE_CYCLES` in the guest UART.
It reads the RV64 host cycle counter just before Muon reset release and just
after all-finished is observed, before output readback. The interval includes
MMIO, launch and polling; it is not a Muon core cycle count or calibrated GPU
time. A timed two-repetition Gather smoke build has a byte-identical RV32
image to its untimed control. Both flags require a rebuild and cannot be
added to an ELF through `--sim-only`.
The [complete standard-suite JSON set](inputs/README.md) and its
[native-size inventory](evaluation/native-workload-footprint.csv) are tracked
here. The inventory covers 114 cases; it is a feasibility audit, not a list
of completed FPGA runs.

For a Gather with `wrap=1`, `--count C --iteration-start S` prepares the
contiguous repetitions `[S, S+C)` using source values from their original
global indices. It rejects other families and wraps because they need output
state across chunks. The tiny [chunk smoke input](chunk-smoke.json) splits
five repetitions into three and two; the final chunk and unsplit run have
the same digest. The [upstream serial record](evaluation/chunk-smoke-upstream.csv)
checks all three generated ELFs independently. This is a correctness primitive
for large Gather decompositions. The corresponding
[original-count GPU AMG chunk runs](../evaluation/firesim/amg-gpu-chunks/README.md)
both passed complete-output U250 checks. The separate booted runs do not
provide a single-launch timing equivalent.
[`tools/spatter-gather-chunks.py`](tools/spatter-gather-chunks.py) records the
contiguous count ranges, rebased source indices, per-chunk footprints, and
expected final digest. The [AMG GPU case 0 plan](evaluation/amg-gpu-chunk-plan.json)
uses two 7,352,941-repetition chunks to cover the original 14,705,882 count.

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

For an independent correctness check, `tools/spatter-upstream-golden.py`
compiles the pinned [original Spatter](https://github.com/hpcgarage/spatter)
serial backend, runs it on the same deterministic inputs, and compares its
complete output digest with the Radiance/Cyclotron result. The
[golden record](evaluation/README.md#independent-upstream-correctness-oracle)
documents nine original-size single-family cases, five small cases, the
scaled AMG case, and the full-size two-stage composition, with raw artifact
hashes and reproduction commands.
The [mutation controls](evaluation/upstream-control-results.csv) rerun one
small case with changed source data and pattern entries; both changes alter
upstream's output digest, as does a direct output-bit flip.

Run `python3 -m unittest -v test_run.py` for parser, address-map, and software
composition checks.
