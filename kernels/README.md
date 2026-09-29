# STREAM and Spatter on Radiance

This branch adds Muon kernels for four STREAM operations and five Spatter
transfers. It includes full SoC RTL runs, timing-model runs, and checks against
the original Spatter implementation. The [artifact inventory](evaluation/README.md)
records the input, ELF, simulator, and check status behind each result.

## Implemented kernels

- [STREAM](stream/README.md): Copy, Scale, Add, and Triad, each measured on
  1,048,576 float32 elements.
- [Spatter](spatter/README.md): Gather, Scatter, GatherScatter (GS),
  MultiGather, and MultiScatter. Each original GPU STREAM suite case has 256
  pattern entries and 1,024 repetitions.
- Spatter address plans, a materialized Gather→Scatter chain, and a fused GS
  version of that chain.
- Original-size [LANL xRAGE asteroid patterns 5 and 9](https://lanl.github.io/benchmarks/09_Microbenchmarks/M2_SPATTER/SPATTER.html)
  and two LULESH traces in the timing model.

The [run log](results.md) has the remaining experiments and run history.

## Execution units and data types

| Workload | Radiance execution unit | Data and arithmetic in this branch |
| --- | --- | --- |
| STREAM Copy | Muon SIMT cores | FP32 arrays; loads and stores, with no floating-point operation |
| STREAM Scale, Add, Triad | Muon SIMT cores | FP32 arrays and FP32 arithmetic |
| All five Spatter families | Muon SIMT cores | Opaque 64-bit values copied as two 32-bit loads and two 32-bit stores; 32-bit index arithmetic, with no floating-point operation |

All kernels in this report run on Muon. They issue no MX-Gemmini instructions.
Muon supports FP32 according to its [architecture specification](https://github.com/ucb-bar/radiance/blob/main/docs/muon.md);
Scale, Add, and Triad also exercise FP32 in the full SoC RTL runs below. This
branch has no BF16, FP16, FP8, FP6, FP4, or FP64 arithmetic result. Spatter's
64-bit values are copied as bits; they are not FP64 calculations.

## Methodology

**Hardware and simulators.** The nine main cases run as compiled Muon kernels
on the integrated one-cluster, two-core `RadianceSingleClusterConfig`. The
RV64 host launches them and checks their output. Verilator 5.022 runs the full
SoC RTL; Cyclotron runs the same ELF with a timing model. The RTL build uses
Radiance `b83419ea85e7fc0b6cbd81b3bb0a0cb3751a5445`. A later upstream
commit, `4cb8700`, changes the default coalescer source-ID setting; these
cycles have not been rerun with it. This work changed no RTL logic.

**Cycles.** RTL GPU cycles are the larger of the two Muon core counts in the
Verilator performance reports. Model cycles are Cyclotron's completed-run
count. A measured clock is unavailable, so this report gives cycles rather
than nanoseconds or GB/s. Cyclotron uses a generic DRAM node with 200-cycle
base latency and 32-byte/cycle service rate.

**Issue slots.** `combined IPC = total core instructions / RTL GPU cycles`.
Each core can issue one instruction per cycle, so issue-slot
use is `combined IPC / 2`. It covers the whole reported cycle window. It does
not measure active lanes, warp occupancy, or MX-Gemmini use. The
[STREAM](stream/evaluation/current-build-results.csv) and
[Spatter](spatter/evaluation/current-build-results.csv) paired CSVs retain
the per-core counters. The table below uses integer instruction counts rather
than rounded per-core IPC.

**Traffic.** Logical payload counts application reads and writes. The model
memory B/cycle column divides issued global-memory bytes by model cycles; it
includes index traffic and transaction granularity. The
[model CSV](evaluation/workload-results.csv) also has transaction counts and
LSU load/store issue counts. These counters
are not HBM read/write bytes. Spatter's published bandwidth convention counts
one side of each transfer; logical payload here counts both.

**Checks.** Deterministic model runs check the full output digest and guard
regions. Full-size RTL checks the complete digest for Gather and MultiGather;
the other seven cases check samples and guards. Sixteen comparisons against
the [pinned upstream Spatter serial code](https://github.com/hpcgarage/spatter/tree/ec8923711f8dc21eedff7189f12b02eb06845d2f/src/Spatter)
match the complete model output digest. The [golden table](spatter/evaluation/upstream-golden-results.csv)
records the input and output hashes.

### Comparing with upstream Spatter

For each checked case, we feed the same source bytes, patterns, repetition
count, wrap, and deltas to Radiance and to Spatter's pinned serial backend.
The golden driver compiles upstream's
[serial transfer loops](https://github.com/hpcgarage/spatter/blob/ec8923711f8dc21eedff7189f12b02eb06845d2f/src/Spatter/Configuration.cc)
and pattern parser; it does not reimplement the five loops. The comparison has three
independent parts:

| Check | Evidence |
| --- | --- |
| Input mapping | The JSON deck and ELF are identified by SHA-256. Upstream's parser agrees with our indices and deltas for [19 pattern examples](spatter/evaluation/upstream-pattern-parser-results.csv), including every generator string in its standard suite. Explicit JSON arrays are read directly, subject to any declared pattern-size limit. |
| Final values | The unmodified upstream serial kernels and Cyclotron agree on the complete 64-bit output digest for [16 deterministic cases](spatter/evaluation/README.md#independent-upstream-correctness-oracle): five original-size GPU STREAM cases, xRAGE Gather and ordered Scatter, two LULESH traces, five small cases, a scaled AMG case, and the full-size Gather→Scatter composition. The retained upstream output arrays also have SHA-256 hashes. |
| Failure controls | A freshly compiled upstream driver reproduces one recorded small Scatter output byte for byte. Flipping one source bit, swapping two pattern entries, or flipping one output bit changes both the output SHA-256 and the digest. The [control record](spatter/evaluation/upstream-control-results.csv) and [script](spatter/tools/check-upstream-controls.py) make this repeatable. |

This establishes the tested *serial transfer behavior* on controlled inputs.
The complete model output is reduced to a 64-bit digest, so this is strong
empirical evidence rather than a formal proof of every byte. The mutation
controls exercise the upstream oracle and digest check; they do not rerun
Radiance with altered inputs. Matching final values also does not prove an
identical memory-request stream: overwritten values can leave the same final
array. A strict traffic comparison would need per-access traces and matching
CUDA write and atomic behavior. The full-size RTL
host check reads every value for Gather and MultiGather and samples the other
seven workloads. Upstream's
[CUDA Gather/MultiGather kernels](https://github.com/hpcgarage/spatter/blob/ec8923711f8dc21eedff7189f12b02eb06845d2f/src/Spatter/CudaBackend.cu)
use different write traffic,
and the published atomic Scatter case has different collision semantics.
Our runs also omit Spatter's repeated timing protocol and use deterministic
input values in place of its benchmark initialization. Their cycles therefore
describe our Radiance mapping; they are not a reproduction of published CUDA
bandwidth. STREAM is a separate FP32 implementation checked against its
generated arithmetic reference, with no upstream Spatter golden comparison.

## Kernel breakdown

The four STREAM operations use float32 arrays `A`, `B`, and `C`. Each ran
independently on 1,048,576 elements in RTL and the timing model.

| Kernel | Operation | Memory accesses per element | Output in a four-stage chain |
| --- | --- | --- | --- |
| Copy | `C[i] = A[i]` | 1 read, 1 write | `C` feeds Scale |
| Scale | `B[i] = 2 × C[i]` | 1 read, 1 write | `B` feeds Add |
| Add | `C[i] = A[i] + B[i]` | 2 reads, 1 write | `C` feeds Triad |
| Triad | `A[i] = B[i] + 2 × C[i]` | 2 reads, 1 write | `A` is the final result |

The table shows the data dependencies for a possible chain. In the measured
ELFs, each operation has its own generated inputs.

For Spatter, `i` is the repetition, `j` the pattern entry, `L` the pattern
length, and `W` the dense-array wrap. The dense address is
`d(i,j) = j + L × (i mod W)`; a direct pattern address is
`s(i,j) = p[j] + δ × i`. The generated [address plan](spatter/plan.py)
specifies the read map, write map, and schedule for every case. Every transfer
moves one opaque 64-bit value. All five original GPU STREAM suite cases used
256 entries × 1,024 repetitions and passed full SoC RTL and model checks.

| Kernel | Source → destination | Device schedule | Inputs measured |
| --- | --- | --- | --- |
| Gather | `sparse[s(i,j)] → dense[d(i,j)]` | One owner per dense slot; repetitions in order | GPU STREAM: RTL + model. xRAGE 5 and LULESH 1: model. |
| Scatter | `dense[d(i,j)] → sparse[s(i,j)]` | One task per transfer; destination ownership for overlapping writes | GPU STREAM: RTL + model. xRAGE 9 and LULESH 3: model. |
| GS | `sparse_in[p_g[j]+δ_g i] → sparse_out[p_s[j]+δ_s i]` | One task per transfer; ordered destinations available | GPU STREAM: RTL + model. A compatible Gather→Scatter pair can also be fused into GS. |
| MultiGather | `sparse[p[q[j]]+δ i] → dense[d(i,j)]` | One owner per dense slot | GPU STREAM: RTL + model. |
| MultiScatter | `dense[d(i,j)] → sparse[p[q[j]]+δ i]` | One task per transfer; ordered destinations available | GPU STREAM: RTL + model. |

The checked outputs match upstream Spatter's serial implementation. Radiance
Gather and MultiGather write every dense result. Upstream CUDA usually keeps
these cases close to read-only, so its traffic differs. Parallel xRAGE 9
Scatter has conflicting destinations; its result is exploratory. The ordered
mapping matches the serial final output by assigning each destination to one
lane. The published GPU xRAGE 9 number uses atomics instead.

The [standard-suite audit](spatter/evaluation/standard-suite-coverage.csv)
finds that 38/114 original-size configurations fit the GPU address and task
window. Seventy-five exceed the address window; one exceeds RV32 task indexing.
Reduced-repetition versions test the address mapping but change the workload
size.

## Workload coverage and remaining work

This branch covers the STREAM and Spatter kernel work and one measured
compute→memory→compute chain. The wider HPC, LLM, HBM, and FPGA evaluation
needs separate runs.

| Target | Current result | Status | Next evidence needed |
| --- | --- | --- | --- |
| STREAM | Four independent FP32 kernels at 1,048,576 elements; RTL and model cycles for each. | Four operations measured. | Shared `A/B/C` buffers and a measured Copy→Scale→Add→Triad run; other data types if they matter to the target. |
| Spatter | Five GPU STREAM cases at original count and pattern size; RTL, model, and upstream golden checks. | Five families measured; 38/114 standard-suite configurations fit at original size. | Address-window support for the remaining cases; matched CUDA traffic/atomics before comparing published bandwidth. |
| xRAGE | Original-size asteroid Gather 5 and Scatter 9 mapped; model cycles and memory counters. Ordered Scatter and Gather match upstream serial output. | Two traces mapped in the model. | Original-size RTL, atomic Scatter if matching the published GPU case, and application-level runs. |
| Other HPC inputs | Original-size LULESH Gather and ordered Scatter in the model; reduced-count AMG Gather correctness check. | Selected traces mapped. | Original-size AMG GPU performance and full application execution. |
| Compute→memory→compute | Gather writes a dense intermediate; Scatter reads it after a barrier. Fused GS gives the same final result. | One chain measured in RTL and model. | A general stage scheduler and checks for further chains. |
| GEMM and LLM inference | Existing [GEMM](gemm_mxgemmini/README.md), [batched GEMV](gemv_batched_fp8_m128/README.md), and [attention](flash_attention_mx_gqa/README.md) kernels came from [earlier work](https://github.com/ucb-bar/radiance-kernels/pull/1). | Separate kernel results; none in this evaluation. | Shared buffers, datatype/layout conversion, and measured end-to-end model schedules. |
| HBM and FPGA stages | Generic model memory requests and one-cluster RTL cycles. | No HBM-calibrated or FPGA result. | Calibrated memory timing, measured clock, bitstream runs, and scalability measurements. |

The proposed FPGA sequence starts with one SM, then two SMs on one FPGA without
memory logic. Later stages add memory and move to two FPGAs. The present
STREAM and Spatter kernels use global-memory loads and stores, so the
memory-free FPGA stages cannot reproduce these runs as measured.

## Checks still open

These are the checks needed before extending the claims above. Each run should
retain its input deck, source and ELF hashes, simulator revision, complete
log, output check, and counters in the [artifact inventory](evaluation/README.md).

| Claim to confirm | How to check it | Evidence needed to close it |
| --- | --- | --- |
| The Muon kernels issue the intended Spatter accesses. | Add correctness-only trace buffers for small cases in all five families, including wrap and duplicate destinations. Record each Muon read/write element address and compare it with addresses from the pinned serial loops on the same inputs. Parallel, collision-free cases may reorder transfers; ordered Scatter must retain each destination's write order. | Every expected transfer appears once, with no extra transfer. Source and destination addresses, transfer count, and ordered last-writer behavior agree. Keep the traces and comparison script. This check concerns serial Spatter behavior; the CUDA Gather traffic differs. |
| The seven full-size RTL runs currently checked by samples have complete correct outputs. | Build correctness-only host variants that digest every output word and check both guards. Compare their GPU load images with the measured ELFs using [compare_elf_loads.py](evaluation/compare_elf_loads.py), then run full SoC RTL. For STREAM, compute the expected FP32 arrays independently on the CPU. | All seven full-size RTL outputs pass complete checks; the GPU instructions and initialized data match the timed runs, or any difference is recorded with a fresh timing result. |
| The mapping covers more of the original Spatter suite at its stated size. | Use the [suite audit](spatter/evaluation/standard-suite-coverage.csv) to group the 76 configurations that do not fit. Add chunking or another mapping only where count, wrap, indices, and write order can be preserved; compare each new result with upstream serial output. | Each added case runs with its original count and pattern, has a matching complete output check, and has a recorded address/resource footprint. Reduced-count runs remain labeled separately. |
| The xRAGE and LULESH model results carry over to RTL. | Run the existing original-size Gather and ordered Scatter ELFs through the full SoC Verilator simulator, using the same ELF hashes as the model runs. Record finish status, output checks, per-core cycles, and instruction counts. | Completed paired RTL/model records for each named trace. Until then their original-size cycles are model results only; parallel xRAGE Scatter remains exploratory. |
| A CUDA-style Spatter performance comparison is meaningful. | Match the upstream CUDA Gather/MultiGather conditional-write behavior and the chosen Scatter collision policy in separate kernel variants. Check whether the available Muon ISA provides the required 64-bit atomic operation before claiming atomic equivalence. Run upstream and Radiance with the same deck, initialization, repetition policy, and timed region; inspect access traces and traffic counters. | Matching declared access semantics and benchmark boundaries, with raw runs for both sides. If atomic exchange cannot be matched, keep ordered Scatter results separate from CUDA atomic throughput. |
| Multi-stage workloads execute with shared data. | Run Copy→Scale→Add→Triad on shared `A/B/C` buffers and check the final FP32 array against a CPU reference. For a Muon→MX-Gemmini chain, first define the datatype, layout, and buffer ownership at the handoff, then run a small checked chain in RTL and the model. | End-to-end output and guards pass; stage boundaries and intermediate hashes are recorded; measured chain cycles are reported rather than a sum of separate runs. |
| The cycle tables apply to the newer Radiance revision. | Rebuild the SoC simulator from the newer Radiance source, rerun the nine main cases and the composition pair on pinned inputs, and regenerate paired tables with the new simulator and ELF hashes. | Complete checks and cycle records tied to the new revision. The current tables remain labeled as measurements of `b83419e` until that run exists. |
| HBM and FPGA claims have measured support. | Calibrate memory timing against an HBM configuration and measured reference workloads, then run the same kernel ELFs or documented rebuilds on an available bitstream. Record clock, memory configuration, host/device transfer boundaries, and scaling setup. | Calibrated model errors and actual HBM/FPGA measurements. The current generic-DRAM cycles cannot establish bandwidth in GB/s or multi-FPGA scaling. |
| Another machine can audit the results. | Export a fresh bundle with [export_report_bundle.py](evaluation/export_report_bundle.py) after the final branch revision, then unpack it in a clean checkout and rerun the report, golden, control, and artifact verifiers. | A bundle hash, pinned Git revisions, complete raw logs, ELF/input/simulator snapshots, and passing verification output from the clean checkout. The existing local archive predates the latest report edits. |

The kernel and correctness work belongs in `radiance-kernels`. HBM calibration,
bitstream availability, and FPGA scaling need hardware results outside this
branch.

## Stitching kernels

### Gather → Scatter: measured

[run_chain.py](spatter/run_chain.py) accepts one Gather and one Scatter
case. It checks that the Gather output length equals the Scatter input length,
puts the source, intermediate, and output arrays in one host/device ELF, and
runs both stages in one Muon launch. All warps cross a barrier before Scatter
reads the dense intermediate. For overlapping Scatter destinations, an
ordered schedule gives each destination's writes to one lane in source order.
The model's complete output digest matches pinned upstream serial Spatter.

```text
sparse source ──Gather──> dense intermediate ──barrier──> Scatter ──> sparse output
      │
      └────────────── fused GS (same final output) ─────────────────> sparse output
```

### Fused GS: measured

[spatter-compose.py](spatter/tools/spatter-compose.py) makes one GS plan
from a compatible Gather→Scatter pair. Count, pattern length, wrap, and
intermediate length must match. When several Gather repetitions write the
same dense slot, the fused read selects the *last* writer. The measured GPU
STREAM pair meets these conditions. Its cycle and traffic comparison is below.

### Further chains

[stitch_reference](spatter/plan.py) computes a serial golden for a sequence
of Spatter transfers when each output length matches the next input length.
The device code currently implements Gather→Scatter only. More stages require
a scheduler, shared-buffer lifetime rules, a barrier or verified launch
boundary, collision handling, and a check of the combined output.

Copy→Scale→Add→Triad needs shared `A/B/C` buffers and a tested stage boundary.
The current four runs use independent inputs, so their cycle counts do not
give the chain latency. A Muon transfer followed by MX-Gemmini also needs a
datatype and layout contract: Spatter carries opaque 64-bit words, STREAM
uses FP32, and the existing MX kernels have their own formats. Such a chain
has not been run in this evaluation.

## Results: original-size full SoC runs

Every row below passed full SoC RTL and an identical-ELF timing-model run.
Model memory B/cycle is issued traffic divided by model cycles.

| Kernel | RTL GPU cycles | Model cycles | Combined RTL IPC | RTL issue slots used | Model memory B/cycle | RTL output check |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| STREAM Copy | 3,970,659 | 3,987,096 | 0.264 | 13.21% | 2.110 | samples + guards |
| STREAM Scale | 3,971,774 | 4,024,544 | 0.281 | 14.03% | 2.090 | samples + guards |
| STREAM Add | 7,485,262 | 6,346,569 | 0.166 | 8.32% | 1.986 | samples + guards |
| STREAM Triad | 7,750,785 | 6,346,349 | 0.161 | 8.04% | 1.986 | samples + guards |
| Spatter Gather | 842,623 | 1,998,857 | 0.176 | 8.81% | 4.209 | complete digest + guards |
| Spatter Scatter | 1,247,066 | 1,040,016 | 0.355 | 17.76% | 9.097 | samples + guards |
| Spatter GS | 2,158,179 | 2,038,053 | 0.228 | 11.40% | 5.157 | samples + guards |
| Spatter MultiScatter | 1,360,268 | 1,046,089 | 0.362 | 18.09% | 10.047 | samples + guards |
| Spatter MultiGather | 945,962 | 1,999,593 | 0.157 | 7.85% | 4.208 | complete digest + guards |

The raw [Spatter cycles](spatter/evaluation/current-build-results.csv),
[Spatter memory counters](spatter/evaluation/current-build-memory.csv),
[STREAM cycles](stream/evaluation/current-build-results.csv), and
[STREAM memory counters](stream/evaluation/current-build-memory.csv) feed
this table. Model Gather takes more than twice its RTL cycles, while model
Triad takes fewer. There is no single model-to-RTL correction factor for the
xRAGE rows.

### Stitching Gather and Scatter

Both paths produce the upstream/model output digest `7e49d79ec062db25`.
Full-size RTL checks samples and guards.

| Mapping | RTL GPU cycles | Model cycles | Logical payload bytes | Model memory bytes issued |
| --- | ---: | ---: | ---: | ---: |
| Materialized Gather→Scatter | 2,294,139 | 3,100,336 | 8,388,608 | 17,850,624 |
| Fused GS | 1,457,400 | 1,060,936 | 4,194,304 | 10,509,568 |

Fusion saved **836,739 RTL GPU cycles (36.5%)** and halved logical payload.
The [paired runs](spatter/evaluation/composition-fullsize-rtl-results.csv)
and [model memory counters](spatter/evaluation/composition-fullsize-memory.csv)
retain the exact records.

### Application-pattern results: timing model only

| Input and mapping | Transfers | Model cycles | Model memory bytes issued | Output status |
| --- | ---: | ---: | ---: | --- |
| xRAGE asteroid pattern 5 Gather | 8,368,968 | 100,529,234 | 478,662,848 | complete digest |
| xRAGE asteroid pattern 9 parallel Scatter | 6,664,304 | 98,895,114 | 389,316,672 | exploratory; conflicting destinations |
| xRAGE asteroid pattern 9 ordered Scatter | 6,664,304 | 96,235,005 | 581,683,520 | complete serial-order digest |
| LULESH case 1 Gather | 3,699,168 | 74,419,767 | 532,704,064 | complete digest |
| LULESH case 3 ordered Scatter | 2,048,032 | 74,611,507 | 536,606,144 | complete serial-order digest |

These rows come from the [model result CSV](evaluation/workload-results.csv). The AMG GPU
case was checked at 1,024 repetitions, down from 14,705,882 in the original
input, so it is absent from this original-size table.

## Measurement limits

The RTL runs establish completion cycles and host output checks for the stated
inputs. The model adds request and transaction counts with generic DRAM
timing. There is no measured HBM bandwidth, per-access latency, lane
efficiency, warp occupancy, FPGA clock, or FPGA scaling result. VCS lacked a
runtime license for this build, and GSIM was unavailable; Verilator supplied
the RTL runs. The [presentation notes](evaluation/presentation.md) give a
shorter version for slides.

## Reproduce and inspect

From the repository root, the verifier checks the CSVs against raw logs,
paired RTL/model runs, ELF hashes, finish signals, and output checks:

```sh
python3 kernels/evaluation/verify_report.py
python3 kernels/spatter/tools/verify-upstream-golden.py --upstream /path/to/pinned/spatter
python3 kernels/spatter/tools/check-upstream-controls.py --upstream /path/to/pinned/spatter
python3 kernels/evaluation/verify_artifact_snapshot.py --workspace /path/to/chipyard
```

The upstream checker expects the pinned Spatter revision above. The
[artifact inventory](evaluation/README.md) lists the run roots and local ELF,
input, and simulator snapshots. Large raw files stay local; CSVs and
verification scripts are in Git. The branch also contains the small raw-log
archive. The ELF snapshots, dependency binaries, raw golden arrays, and
portable report bundle are local artifacts; Git alone cannot restore them.
