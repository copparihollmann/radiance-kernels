# STREAM and Spatter on Radiance

The kernels, generators, simulation commands, and result records are in
[`stream`](stream/README.md) and [`spatter`](spatter/README.md). All results
below are from the local `spatter-workloads` branch of `radiance-kernels`.
The branch has not been pushed.
The [artifact inventory](evaluation/README.md) indexes run status, provenance,
ELF and input hashes, raw logs, and simulator files for later reporting.

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
The [consolidated result CSV](evaluation/workload-results.csv) puts all 14
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
with a complete output digest. The
[paired table](spatter/evaluation/current-build-results.csv) records the
byte-identical build, RTL, and model ELF checks. The remaining three rebuilt
GPU STREAM RTL cases are running: GS and MultiScatter are in one queue, while
MultiGather started in its own run directory.
The four original-size STREAM RTL cases are running: Copy and Scale are in one
queue, while Add and Triad started in their own run directories. Their
[paired table](stream/evaluation/current-build-results.csv) has no
result rows until the first case completes.

For the small equal-count composition case, fused GS and the materialized
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
share the same traffic definition. There is no FPGA bitstream or calibrated
clock result in this evaluation.
