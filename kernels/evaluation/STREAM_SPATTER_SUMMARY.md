# STREAM and Spatter kernels on Radiance

This report uses the organization of the earlier [Radiance kernel performance PR](https://github.com/ucb-bar/radiance-kernels/pull/1): what the kernels add, what was measured, results, and limits. It covers the STREAM and Spatter slice requested for Radiance workload mapping. Every cycle figure below comes from a completed run, with the input, ELF, simulator, and check status indexed in the [artifact inventory](README.md).

## What this work adds

- Four standalone float32 STREAM operations: Copy, Scale, Add, and Triad, each on 1,048,576 elements.
- All five [Spatter](https://github.com/hpcgarage/spatter) transfer families: Gather, Scatter, GatherScatter (GS), MultiGather, and MultiScatter. The five original GPU STREAM suite cases each use 256 pattern entries and 1,024 repetitions.
- A decomposition of each Spatter family into source address, destination address, and schedule, plus one materialized Gather→Scatter chain and its equivalent fused GS mapping.
- Original-size mappings of [LANL xRAGE asteroid patterns 5 and 9](https://lanl.github.io/benchmarks/09_Microbenchmarks/M2_SPATTER/SPATTER.html) and two LULESH application traces in the timing model.
- Generated source data, address plans, RV64 host/Muon ELFs, output checks, and a golden comparison against the pinned, unmodified upstream Spatter serial implementation.

The kernel implementations and build instructions are in [stream](../stream/README.md) and [spatter](../spatter/README.md). The [complete result report](../WORKLOAD_RESULTS.md) retains the remaining measurements and experiment history.

## Methodology

**Radiance basis.** The nine main cases ran as compiled Muon kernels on the integrated, one-cluster, two-core `RadianceSingleClusterConfig` full SoC RTL under Verilator 5.022. The RV64 host launches the kernels and checks returned data. The exact same ELF for each case also ran under the Cyclotron timing model. The RTL build is pinned to Radiance `b83419ea85e7fc0b6cbd81b3bb0a0cb3751a5445`. The results have not been rerun on the newer `4cb8700` coalescer setting. No RTL logic was changed for this kernel work.

**Cycle counts and latency.** The RTL column is the maximum of the two Muon core cycle counts in the Verilator performance reports. The model column is Cyclotron's completed simulation cycle count. These are cycle latencies for the reported executions, not nanoseconds or a measured FPGA runtime. No GPU clock was supplied. The model uses a generic 200-cycle base DRAM latency and 32-byte/cycle service rate, not calibrated HBM.

**Issue utilization.** `combined IPC = (core 0 instructions + core 1 instructions) / RTL GPU cycles`. With one issue slot per core per cycle, the `issue slots used` column is `combined IPC / 2`. It describes issued instructions over the full reported cycle window. It is not active-lane efficiency, warp occupancy, matrix-engine utilization, or a direct fraction of memory bandwidth. The [paired STREAM CSV](../stream/evaluation/current-build-results.csv) and [paired Spatter CSV](../spatter/evaluation/current-build-results.csv) preserve the original per-core instruction counts and IPC. The utilization percentages below are derived from the integer instruction counts, avoiding rounded IPC as an input.

**Memory activity.** Logical payload counts the application data read and written. `model memory B/cycle` divides Cyclotron's global-memory bytes issued by its model cycles; it includes index traffic and transaction granularity. The [consolidated model CSV](workload-results.csv) also records memory transaction counts and LSU global-load/store issue counts. These are modeled request counters, not separate HBM read/write byte measurements. Spatter's published one-sided bandwidth convention differs from our two-sided logical payload; neither can be converted to GB/s without a clock and a matching memory system.

**Correctness.** The deterministic model runs check complete output digests and guards. In full-size RTL, Gather and MultiGather check complete digests; the other seven workload cases check samples and guards. Sixteen deterministic comparisons against [pinned upstream Spatter](https://github.com/hpcgarage/spatter/tree/ec8923711f8dc21eedff7189f12b02eb06845d2f/src/Spatter) agree with the model's complete output digests; the [golden table](../spatter/evaluation/upstream-golden-results.csv) records input and output hashes. A sampled RTL check is not a full RTL output comparison.

## Workload equivalence

| Source workload | Radiance mapping | Equivalence and boundary |
| --- | --- | --- |
| Four STREAM operations | Independent float32 Copy, Scale, Add, and Triad | The arithmetic and array access for each operation match; this is four separate ELFs, not a dependent four-stage STREAM run. |
| Five Spatter GPU STREAM cases | Original pattern, count, wrap, and transfer family | The documented address transfers and deterministic final outputs match upstream serial Spatter. CUDA Gather/MultiGather typically avoid the full dense-output writes that Radiance performs, so traffic and timing are not CUDA-equivalent. |
| xRAGE asteroid pattern 5 | Original-size Gather | Address pattern and complete output match upstream serial Spatter; performance is currently timing-model only. |
| xRAGE asteroid pattern 9 | Original-size Scatter, parallel and ordered schedules | Ordered Scatter matches the serial final output. Parallel Scatter has conflicting destinations and an exploratory output. Ordered software scheduling is not the CUDA atomic operation used for the published GPU benchmark. Both performance rows are timing-model only. |
| LULESH traces | Original-size Gather and ordered Scatter | Complete outputs match upstream serial Spatter; performance is timing-model only, and ordered Scatter has software conflict-scheduling overhead. |

All five Spatter operation families are implemented. Of the standard suite's 114 original-size configurations, 38 fit the current GPU address and task window. Another 75 exceed the address window and one exceeds RV32 task indexing. A reduced-repetition mapping can test semantics, but its performance does not represent the original-size benchmark. The [coverage audit](../spatter/evaluation/standard-suite-coverage.csv) gives the status of every case.

## Results: original-size full SoC runs

Every row below passed a full SoC RTL run and an identical-ELF timing-model run. `Model memory B/cycle` is modeled issued traffic divided by model cycles; it is not achieved HBM bandwidth.

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

The [paired results](../spatter/evaluation/current-build-results.csv) and [model memory counters](../spatter/evaluation/current-build-memory.csv) expose the Spatter inputs to this table; the corresponding [STREAM results](../stream/evaluation/current-build-results.csv) and [memory counters](../stream/evaluation/current-build-memory.csv) expose the STREAM inputs. The model and RTL do not differ by one stable factor: for example, model Gather takes more than twice the RTL cycles, while model STREAM Triad takes fewer. We therefore report the simulator source for each number and do not use model cycles as an RTL estimate for xRAGE.

### Stitching Gather and Scatter

The materialized chain writes a 256-element dense intermediate, synchronizes all warps, then reads it in Scatter. The fused GS path directly selects the final Gather writer for each dense slot. Both produce the same complete upstream/model output digest, `7e49d79ec062db25`; full-size RTL checks samples and guards.

| Mapping | RTL GPU cycles | Model cycles | Logical payload bytes | Model memory bytes issued |
| --- | ---: | ---: | ---: | ---: |
| Materialized Gather→Scatter | 2,294,139 | 3,100,336 | 8,388,608 | 17,850,624 |
| Fused GS | 1,457,400 | 1,060,936 | 4,194,304 | 10,509,568 |

Fusion saved **836,739 RTL GPU cycles (36.5%)** and halved logical payload on this input. The [paired RTL/model table](../spatter/evaluation/composition-fullsize-rtl-results.csv) and [model memory counters](../spatter/evaluation/composition-fullsize-memory.csv) retain the exact records. This is one demonstrated two-stage composition, not arbitrary chaining of all Spatter families.

### Application-pattern results: timing model only

| Input and mapping | Transfers | Model cycles | Model memory bytes issued | Output status |
| --- | ---: | ---: | ---: | --- |
| xRAGE asteroid pattern 5 Gather | 8,368,968 | 100,529,234 | 478,662,848 | complete digest |
| xRAGE asteroid pattern 9 parallel Scatter | 6,664,304 | 98,895,114 | 389,316,672 | exploratory; conflicting destinations |
| xRAGE asteroid pattern 9 ordered Scatter | 6,664,304 | 96,235,005 | 581,683,520 | complete serial-order digest |
| LULESH case 1 Gather | 3,699,168 | 74,419,767 | 532,704,064 | complete digest |
| LULESH case 3 ordered Scatter | 2,048,032 | 74,611,507 | 536,606,144 | complete serial-order digest |

These rows come from the [model result CSV](workload-results.csv). They are not full-size RTL, CUDA atomic-throughput, or HBM measurements. The AMG GPU case was also checked at a reduced 1,024 repetitions; its cycle result is intentionally omitted here because the original input requests 14,705,882 repetitions.

## What these measurements establish for Andre's request

The kernels map regular and irregular HPC memory operations onto Radiance, execute instructions in the full SoC simulation, and provide cycle and memory-request evidence. The materialized chain demonstrates data written by one Muon stage, synchronized, then read by a second stage. The independent upstream golden checks support the transfer semantics. These are concrete workload-mapping and software-evaluation results, rather than a CPU-only reference model.

They do not yet establish calibrated HBM throughput, per-access memory latency, lane efficiency or warp occupancy, GEMM/LLM-inference coverage from this branch, a bitstream, or the staged one-/two-FPGA scalability argument. STREAM and Spatter use Muon SIMT cores; they do not exercise the MX-Gemmini matrix engine. VCS could not obtain a runtime license for this build, GSIM was unavailable, and the RTL results use Verilator. The [presentation brief](PRESENTATION_BRIEF.md) is a shorter version for slides.

## Reproduce and inspect

From the repository root, the report verifier checks the consolidated CSV, raw logs, paired RTL/model cases, ELF hashes, finish signals, and output checks:

```sh
python3 kernels/evaluation/verify_report.py
python3 kernels/spatter/tools/verify-upstream-golden.py --upstream /path/to/pinned/spatter
python3 kernels/evaluation/verify_artifact_snapshot.py --workspace /path/to/chipyard
```

The upstream checker expects the pinned Spatter revision named above. The [artifact inventory](README.md) explains the run roots and local ELF/input/simulator snapshots. Large raw artifacts are local and are not stored in Git; the result tables and verification scripts are in this repository. The existing local report archive is a snapshot of the measurements at feature-branch commit `e1590d9`, before this summary document was added.
