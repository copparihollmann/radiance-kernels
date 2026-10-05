# STREAM and Spatter on Radiance

The kernels, generators, simulation commands, and result records are in
[`stream`](stream/README.md) and [`spatter`](spatter/README.md). The results
below were generated on the `spatter-workloads` branch of `radiance-kernels`.
The [STREAM and Spatter overview](README.md)
reports workload equivalence, latency, derived issue utilization, and the
limits of the measurements. The branch is published on the
`copparihollmann/radiance-kernels` fork as `spatter-workloads`.
The [artifact inventory](evaluation/README.md) indexes run status, provenance,
ELF and input hashes, raw logs, and simulator files for later reporting.
The [U250 FireSim run](evaluation/firesim/README.md) records FPGA output-check
outcomes and whole-program target-cycle counts for representative HPC cases,
including an ordered xRAGE9 failure. Those totals are distinct from the
GPU-cycle measurements below.
These RTL cycles use the pinned Radiance `b83419e` build. Upstream Radiance
`main` advanced to `4cb8700` on 2026-09-29 with one intervening change to
the default coalescer source-ID count (32 to 8). The cycle tables do not
represent a run of that newer setting.

## Timing-model results at the stated input sizes

**Revision audit complete for the timing-model table.** The five GPU STREAM
Spatter rows use ELFs rebuilt from the current kernel source. Rebuilding
changed the Scatter, GS, and MultiScatter instruction images; their former
cycle counts are retained in the
[revision comparison](spatter/evaluation/gpu-stream-revision-comparison.csv).
Loaded-image checks also confirm that xRAGE5, ordered xRAGE9, and both LULESH
patterns retain the same initialized executable content after rebuilding.
Parallel xRAGE9 changed instructions and was rerun: its current model count is
98,895,114 cycles, compared with 99,547,352 on the earlier ELF. Both counts
and their memory counters are in the
[xRAGE9 revision comparison](spatter/evaluation/xrage9-revision-comparison.csv).
Each result remains tied to its recorded ELF hash in the artifact inventory.

The Cyclotron timing model checked every output value and its guard regions
for the deterministic cases below. It uses a generic DRAM timing node (200
cycle base latency, 32 bytes/cycle service rate), not a calibrated HBM model.
The reported cycle and memory-transaction counts are simulator results; no
GPU frequency is assumed.
Small paired checks show substantial workload-dependent differences between
the model and RTL: 256-element STREAM Copy took 3,206 model versus 9,085 RTL
GPU cycles, while 32-repetition LULESH ordered Scatter took 20,565 model
versus 13,556 RTL GPU cycles. Use the full-size model counts as model results
until corresponding RTL measurements or calibration are available.

| Workload | Elements or transfers | Model cycles | Logical payload bytes | Model global-memory bytes issued |
| --- | ---: | ---: | ---: | ---: |
| STREAM Copy | 1,048,576 | 3,987,096 | 8,388,608 | 8,412,416 |
| STREAM Scale | 1,048,576 | 4,024,544 | 8,388,608 | 8,412,416 |
| STREAM Add | 1,048,576 | 6,346,569 | 12,582,912 | 12,606,720 |
| STREAM Triad | 1,048,576 | 6,346,349 | 12,582,912 | 12,606,720 |
| Spatter GPU STREAM Gather | 262,144 | 1,998,857 | 4,194,304 | 8,413,440 |
| Spatter GPU STREAM Scatter | 262,144 | 1,040,016 | 4,194,304 | 9,460,992 |
| Spatter GPU STREAM GS | 262,144 | 2,038,053 | 4,194,304 | 10,509,568 |
| Spatter GPU STREAM MultiScatter | 262,144 | 1,046,089 | 4,194,304 | 10,509,568 |
| Spatter GPU STREAM MultiGather | 262,144 | 1,999,593 | 4,194,304 | 8,414,464 |
| xRAGE asteroid pattern 5, Gather | 8,368,968 | 100,529,234 | 133,903,488 | 478,662,848 |
| xRAGE asteroid pattern 9, parallel Scatter | 6,664,304 | 98,895,114 | 106,628,864 | 389,316,672 |
| xRAGE asteroid pattern 9, ordered Scatter | 6,664,304 | 96,235,005 | 106,628,864 | 581,683,520 |
| LULESH app trace case 1, Gather | 3,699,168 | 74,419,767 | 59,186,688 | 532,704,064 |
| LULESH app trace case 3, ordered Scatter | 2,048,032 | 74,611,507 | 32,768,512 | 536,606,144 |
| AMG GPU Gather, scaled to 1,024 repetitions | 262,144 | 1,156,957 | 4,194,304 | 19,423,488 |

The AMG row uses the original address pattern but reduces its 14,705,882
repetitions to 1,024 to fit the current GPU address range. It is a scaled
mapping and correctness check, not an original-size AMG benchmark result.
Its complete digest `32ae72ce80378661` agrees with the
[unmodified upstream Spatter serial kernel](spatter/evaluation/upstream-golden-scaled-results.csv).

The xRAGE9 parallel Scatter row is exploratory: repeated destinations race,
so its output check covers guards and a nonzero probe. Ordered Scatter
executes every transfer without racing and passed the full serial-order digest
`215de6e81154b9c5` over 6,664,304 transfers. Its cycle and memory counts
describe the generated conflict schedule, not upstream CUDA atomic exchange.
An independent golden check compiled the unmodified
[upstream Spatter serial kernels](https://github.com/hpcgarage/spatter/tree/ec8923711f8dc21eedff7189f12b02eb06845d2f/src/Spatter)
with deterministic inputs. Complete output digests agree with Radiance and
Cyclotron for all five GPU STREAM Spatter families, xRAGE5, ordered xRAGE9,
and both LULESH traces. The
[golden comparison table](spatter/evaluation/upstream-golden-results.csv)
records the input and output hashes; the original-size parallel xRAGE9 run
has no deterministic golden output.
The LULESH rows use the original `standard-suite/app-traces/lulesh.json`
counts of 231,198 for Gather and 128,002 for Scatter. Both passed functional
and timing-model full-output digests on identical per-case ELFs. The Scatter
row serializes repeated destinations and is not an atomic Scatter measurement.
The input JSON SHA-256 is
`9073035ecf77e7fde65262f782286207e76cca24312b2e01688b038901d021ee`.

See the exact [STREAM cycle](stream/evaluation/current-build-model-results.csv),
[STREAM memory](stream/evaluation/current-build-memory.csv),
[Spatter cycle](spatter/evaluation/current-build-model-results.csv), and
[Spatter memory](spatter/evaluation/current-build-memory.csv) CSVs. Logical
payload counts reads and writes required by each kernel. Model global-memory
bytes include index traffic, cache lines, and transaction granularity.
The [consolidated result CSV](evaluation/workload-results.csv) puts all 15
rows, normalized bytes per model cycle, correctness status, output digest,
modeled global load and store queue issue counts, and
input/source/ELF/timing/checker hashes in one machine-readable table. The load
and store counts come from the GPU LSU and describe the whole recorded run;
they are not separate DRAM read and write byte measurements. Regenerate it
with `python3 kernels/evaluation/verify_report.py --write-csv` from the
repository root; the normal verifier checks it against the raw model runs.

## Full SoC checks and composition

Full SoC Verilator and Cyclotron checks passed on identical ELFs for all four
STREAM operations at 256 elements and all five Spatter transfer families on
small inputs. The Spatter checks also cover fused Gather→Scatter, an ordered
overlapping Scatter, and a two-stage materialized Gather→Scatter chain. Their
paired hashes, correctness checks, and GPU cycles are in the
[STREAM smoke table](stream/evaluation/current-build-smoke-results.csv) and
[Spatter smoke table](spatter/evaluation/current-build-smoke-results.csv).
The [LULESH pattern smoke table](spatter/evaluation/app-trace-smoke-results.csv)
adds paired RTL/model checks at 32 repetitions for Gather and ordered Scatter;
both passed full digests. These scaled cycles are separate from the original
LULESH counts above.
An [earlier-build RTL table](spatter/evaluation/prior-build-results.csv) has
original-size GPU STREAM cycles, but its embedded device segments differ from
the current branch. The original-size current-build Gather passed a full SoC
Verilator host check in 842,623 GPU cycles. Rebuilt Scatter passed its sampled
host check in 1,247,066 RTL GPU cycles, versus 1,040,016 timing-model cycles
with a complete output digest. Rebuilt GS passed its sampled host check in
2,158,179 RTL GPU cycles, versus 2,038,053 timing-model cycles with a complete
output digest. Rebuilt MultiGather passed a complete host digest in 945,962
RTL GPU cycles; its identical-ELF model run passed the same digest in
1,999,593 cycles. Rebuilt MultiScatter passed its sampled host check in
1,360,268 RTL GPU cycles; its identical-ELF model run passed a complete
output digest in 1,046,089 cycles. The
[paired table](spatter/evaluation/current-build-results.csv) records the
byte-identical build, RTL, and model ELF checks for all five full-size GPU
STREAM Spatter cases. The earlier partial MultiScatter attempt is retained as
interrupted. Full-size STREAM Scale passed a sampled host check in 3,971,774
Verilator GPU cycles; its identical-ELF timing-model run passed a complete
output digest in 4,024,544 cycles. STREAM Add passed a sampled host check in
7,485,262 Verilator GPU cycles; its identical-ELF timing-model run passed a
complete output digest in 6,346,569 cycles. Copy passed a sampled host check
in 3,970,659 Verilator GPU cycles; its identical-ELF model run passed a
complete output digest in 3,987,096 cycles. Triad passed its sampled host
check in 7,750,785 RTL GPU cycles; its identical-ELF model run passed a
complete output digest in 6,346,349 cycles. The full-size materialized
Gather→Scatter composition passed its sampled host check in 2,294,139 RTL
GPU cycles. The full-size fused composition passed its sampled host check in
1,457,400 RTL GPU cycles. Earlier STREAM
partial attempts are
retained as interrupted. Later Triad and Copy attempts were stopped during
host CPU saturation, including a later Add attempt, and an 8-thread Copy
attempt was stopped to switch
simulator threading; none produced a completed cycle result. The
[paired table](stream/evaluation/current-build-results.csv) retains all four
completed STREAM RTL/model pairs.

The completed original-size GPU STREAM Spatter pairs, each using the same ELF
for RTL and model, are:

| GPU STREAM operation | RTL GPU cycles | Model GPU cycles | RTL/model cycles | RTL host check |
| --- | ---: | ---: | ---: | --- |
| Gather | 842,623 | 1,998,857 | 0.422 | Complete digest |
| Scatter | 1,247,066 | 1,040,016 | 1.199 | Samples and guards |
| GatherScatter | 2,158,179 | 2,038,053 | 1.059 | Samples and guards |
| MultiScatter | 1,360,268 | 1,046,089 | 1.300 | Samples and guards |
| MultiGather | 945,962 | 1,999,593 | 0.473 | Complete digest |

The timing model is not calibrated to RTL or HBM; the ratios describe these
paired runs and should not be used as a general correction factor. All five
model runs checked their complete output digests.

The completed original-size STREAM pairs are:

| STREAM operation | RTL GPU cycles | Model GPU cycles | RTL/model cycles | RTL host check |
| --- | ---: | ---: | ---: | --- |
| Copy | 3,970,659 | 3,987,096 | 0.996 | Samples and guards |
| Scale | 3,971,774 | 4,024,544 | 0.987 | Samples and guards |
| Add | 7,485,262 | 6,346,569 | 1.179 | Samples and guards |
| Triad | 7,750,785 | 6,346,349 | 1.221 | Samples and guards |

All four STREAM RTL runs passed their sampled host and guard checks; the
identical-ELF model runs passed complete output digests.

## Full-size Gather→Scatter composition

The standard GPU STREAM Gather and Scatter cases can be stitched through a
256-element dense intermediate. `inputs/composed-gpu-stream.json` contains the
equivalent fused GS mapping generated by `tools/spatter-compose.py`. Both
model runs checked every output word and guards against the same digest:

| Composition mapping | Model GPU cycles | Logical payload bytes | Model global-memory bytes issued | Output digest |
| --- | ---: | ---: | ---: | --- |
| Fused GS | 1,060,936 | 4,194,304 | 10,509,568 | 7e49d79ec062db25 |
| Materialized Gather→Scatter | 3,100,336 | 8,388,608 | 17,850,624 | 7e49d79ec062db25 |

The materialized chain includes the intermediate write and read. Its two
stages run in one Muon launch with a barrier between them. The model comparison
uses the generic DRAM configuration described above. The exact cycle and
memory counters are in
[composition cycle](spatter/evaluation/composition-fullsize-model-results.csv)
and [memory](spatter/evaluation/composition-fullsize-memory.csv) tables.
Running the pinned, unmodified upstream Spatter serial Gather followed by its
serial Scatter on the same input produces the same complete digest; the
[upstream composition record](spatter/evaluation/upstream-golden-composition-results.csv)
includes hashes of both stages and their intermediate array.

The same single-launch chain control path passed full digests on identical
model and full SoC Verilator ELFs for two small cases:

| Materialized chain check | RTL GPU cycles | Model GPU cycles | Output digest |
| --- | ---: | ---: | --- |
| Distinct destinations | 8,597 | 3,985 | 0dce3cc3243f4bbc |
| Repeated destinations, ordered | 9,306 | 4,878 | 546ebd1e8889cdb4 |

The [paired records](spatter/evaluation/composition-one-launch-pair.csv)
include build, model, and RTL ELF hash checks. The full-size materialized
chain passed a sampled host and guard check in 2,294,139 RTL GPU cycles on
the identical ELF used by the 3,100,336-cycle model run. The fused GS passed
its sampled host and guard check in 1,457,400 RTL GPU cycles on the identical
ELF used by the 1,060,936-cycle model run. The
[paired records](spatter/evaluation/composition-fullsize-rtl-results.csv)
contain per-core counters and correctness levels. Fusing these two stages
saved 836,739 RTL GPU cycles (36.5%) in this configuration; the model predicts
a larger saving, so its relative speedup should not be substituted for RTL.

| Composition mapping | RTL GPU cycles | Model GPU cycles | RTL host check |
| --- | ---: | ---: | --- |
| Fused GS | 1,457,400 | 1,060,936 | Samples and guards |
| Materialized Gather→Scatter | 2,294,139 | 3,100,336 | Samples and guards |

For an earlier small equal-count composition case, fused GS and the materialized
two-stage chain have the same output digest. The model reports 3,189 and
4,839 cycles respectively; the chain also moves more data. The
[composition table](spatter/evaluation/composition-model-results.csv) records
the exact cycle and payload figures.
An additional materialized Gather→ordered Scatter chain passes a complete
output digest with repeated destinations on identical model and full SoC RTL
ELFs. It takes 5,890 model cycles and 14,134 RTL GPU cycles, and issues
49,344 model global-memory bytes for 192 logical payload bytes. The
[paired result](spatter/evaluation/composition-ordered-pair.csv) records the
exact check. These are small composition checks, not atomic Scatter throughput.
Fusing those same two stages as ordered GS preserves the digest in 4,752 model
cycles and 9,649 RTL GPU cycles, with 25,600 issued model global-memory bytes.
The fused kernel moves 96 logical payload bytes; the materialized chain moves
192.

The [standard-suite audit](spatter/evaluation/standard-suite-coverage.csv)
finds 38 of 114 upstream configurations fit the current GPU address window
at their original size; 75 exceed it, and one exceeds the RV32 task range.
Native-size coverage by transfer family is Gather 23/78, Scatter 9/30,
GatherScatter 2/2, MultiGather 2/2, and MultiScatter 2/2. All five family
implementations are present, but original-size support is limited by the
current GPU address window and task range.
Count-reduced versions are mapping checks, not original benchmark results.
Gather/MultiGather implement the documented full transfer, whereas upstream
CUDA uses a conditional single-slot store; their throughput figures do not
share the same traffic definition. The original RTL/model evaluation had no
FPGA result or calibrated clock. A separate
[U250 FireSim run](evaluation/firesim/README.md) now uses a bitstream for
correctness checks; it does not establish calibrated bandwidth.
