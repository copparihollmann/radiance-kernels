# Radiance STREAM and Spatter: presentation brief

The [full STREAM and Spatter summary](STREAM_SPATTER_SUMMARY.md) includes the
measurement method, derived Muon issue utilization, workload equivalence, and
application-pattern results.

## What was measured

- Four independent float32 STREAM kernels (Copy, Scale, Add, Triad) on
  1,048,576 elements, plus all five Spatter operation families on the original
  256-index × 1,024-repetition GPU STREAM cases.
- All nine original-size cases passed the full SoC Radiance Verilator RTL run
  and an identical-ELF Cyclotron timing-model run. RTL used the host sample
  and guard check, except Gather and MultiGather, which checked a complete
  digest. All deterministic model runs checked complete output digests and
  guards.
- One original-size Gather→Scatter composition was measured both fused and
  with a materialized 256-element intermediate. Both RTL mappings passed
  sampled host and guard checks; their model runs and the pinned, unmodified
  upstream Spatter serial stages agreed on the complete output digest
  `7e49d79ec062db25`.

## Full-size GPU cycles

| Kernel | RTL cycles | Timing-model cycles | RTL/model |
| --- | ---: | ---: | ---: |
| STREAM Copy | 3,970,659 | 3,987,096 | 0.996 |
| STREAM Scale | 3,971,774 | 4,024,544 | 0.987 |
| STREAM Add | 7,485,262 | 6,346,569 | 1.179 |
| STREAM Triad | 7,750,785 | 6,346,349 | 1.221 |
| Spatter Gather | 842,623 | 1,998,857 | 0.422 |
| Spatter Scatter | 1,247,066 | 1,040,016 | 1.199 |
| Spatter GatherScatter | 2,158,179 | 2,038,053 | 1.059 |
| Spatter MultiScatter | 1,360,268 | 1,046,089 | 1.300 |
| Spatter MultiGather | 945,962 | 1,999,593 | 0.473 |

The RTL/model ratio varies substantially by kernel. Use the measured RTL
counts for the above configuration, rather than applying one model correction
factor to the remaining cases. The [complete report](../WORKLOAD_RESULTS.md)
and [paired CSVs](../stream/evaluation/current-build-results.csv) retain the
checks, per-core instruction counts, IPC, payload bytes, and exact simulator
provenance; the [Spatter paired CSV](../spatter/evaluation/current-build-results.csv)
contains its five rows.

## Connection to Andre's request

Andre asked for workloads mapped to Radiance, software support to simulate
the compute→memory→compute path, and cycle-level performance evidence. This
branch delivers that foundation for STREAM and Spatter: nine full-size
workload RTL/model pairs, xRAGE and LULESH timing-model mappings, and a
Gather→Scatter chain that writes and rereads an intermediate array in the
integrated SoC simulator. The staged FPGA MVP, calibrated HBM bandwidth,
GEMM/LLM-inference workload coverage, and a multi-FPGA scalability argument
remain separate parts of his broader project. Our scope was kernels and
software evaluation; no RTL logic was changed.

## Stitching result

| Gather→Scatter mapping | RTL cycles | Model cycles | Logical payload bytes | Model global-memory bytes issued |
| --- | ---: | ---: | ---: | ---: |
| Fused GS | 1,457,400 | 1,060,936 | 4,194,304 | 10,509,568 |
| Materialized chain | 2,294,139 | 3,100,336 | 8,388,608 | 17,850,624 |

Fusion saved **836,739 RTL cycles (36.5%)** and halved the logical payload
for this input. The materialized mapping schedules both stages in one Muon
launch with an all-warp barrier and uses a dense intermediate. The
[composition paired CSV](../spatter/evaluation/composition-fullsize-rtl-results.csv)
retains the exact cycle and correctness records. The model's larger predicted
fusion benefit is evidence that it should not stand in for RTL calibration.

## Correctness, coverage, and limits

- The pinned, unmodified [upstream Spatter](https://github.com/hpcgarage/spatter/commit/ec8923711f8dc21eedff7189f12b02eb06845d2f)
  serial implementation supplied independent golden outputs for 16
  deterministic comparisons. All agree with the corresponding Radiance
  model output digests. Raw golden data and hashes are retained locally.
- All five Spatter transfer families are implemented. The standard-suite
  [coverage audit](../spatter/evaluation/standard-suite-coverage.csv) finds
  38/114 original-size configurations fit the present GPU address/task
  window; 75 exceed the address window and one exceeds the RV32 task range.
  Count-reduced mappings are correctness checks, not original-size benchmark
  performance. Original-size xRAGE and LULESH cycle and modeled memory
  results are also in the [complete report](../WORKLOAD_RESULTS.md).
- RTL cycles are from the pinned Radiance `b83419e` full SoC Verilator build.
  Upstream `main` advanced by one coalescer source-ID setting change to
  `4cb8700` on 2026-09-29; these cycles have not been rerun on that revision.
  VCS lacked a runtime license and GSIM was unavailable. No RTL logic was
  changed.
- Cyclotron uses a generic DRAM timing configuration, not calibrated HBM.
  The model records global-memory bytes and transactions, but there is no
  measured GPU clock or HBM bandwidth. Do not present these cycles as GB/s,
  an FPGA result, or an HBM-calibrated projection.

The verified local report archive is
`/scratch/agustin/projects/chipyard/radiance-report-final-with-brief-20260929.tar.gz`.
Its `bundle-info.json` marks a complete snapshot of 194 runs, 818 indexed
files, and 135 golden artifacts. The archive preserves raw logs, ELFs,
simulator binaries, inputs, and Git bundles for this branch and upstream
Spatter. Recheck its SHA-256 after any later export.
