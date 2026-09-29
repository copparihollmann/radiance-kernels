# Spatter evaluation record

The implementation is on the `spatter-workloads` branch of a clone of
[`ucb-bar/radiance-kernels`](https://github.com/ucb-bar/radiance-kernels) at
main commit `aed22f758cdb3318de658179edef88fd039ae6cd`. The working tree
contains `kernels/spatter`, its simulator tools and this record. The RTL
model uses Chipyard `d45f86f4cca379715ac0ceb3a9f2369927796794`, Radiance
`b83419ea85e7fc0b6cbd81b3bb0a0cb3751a5445`, and Cyclotron
`9b774a53a882df4a655c04b5aa662ae8a01d5f47`.
VCS runtime currently queues for a license on this machine, so current-build
RTL runs use the built Verilator 5.022 model. No FPGA bitstream is available.
No Verilog or Scala logic file was edited for these runs. The local Radiance
checkout has three submodule/resource link type changes for its build and a
Cyclotron Rust DPI change that suppresses large instruction traces; the
kernel and evaluation changes are in this `radiance-kernels` branch.

## Independent upstream correctness oracle

[`../tools/upstream-golden.cc`](../tools/upstream-golden.cc) links the
unmodified serial `Configuration.cc`, `PatternParser.cc`, and `Timer.cc` from
[hpcgarage/Spatter commit ec89237](https://github.com/hpcgarage/spatter/commit/ec8923711f8dc21eedff7189f12b02eb06845d2f).
It runs the upstream transfer loops on the same deterministic 64-bit source
array embedded in each Radiance ELF, then saves the complete golden output.
For generated patterns, the runner also compares Radiance's expanded indices
and delta with Spatter's own pattern parser. Explicit JSON index arrays are
passed unchanged. It checks the golden output digest against the build's
expected digest and Cyclotron's complete GPU output readback.
The [pattern parser audit](upstream-pattern-parser-results.csv) separately
checks all 16 distinct generator strings in the upstream standard suite, plus
MS1, Laplacian, and explicit-list examples. All 19 expansions and deltas
match Spatter's parser.
With `--upstream`, the golden verifier also regenerates
`standard-suite-coverage.csv` from the pinned source and requires a byte-for-byte
match.

The [original-size comparison](upstream-golden-results.csv) passes all five
GPU STREAM Spatter families, xRAGE5 Gather, ordered xRAGE9 Scatter, and two
LULESH traces. The [small comparison](upstream-golden-smoke-results.csv) adds
four transfer-family checks and an overlapping ordered Scatter. The
[scaled AMG comparison](upstream-golden-scaled-results.csv) checks 1,024
repetitions of the original GPU Gather pattern. The
[artifact manifest](upstream-golden-artifacts.csv) records SHA-256 and byte
size for every local golden input, output, parser record, log, and driver.
Those 219 MB of raw files are retained under the ignored `../golden-runs/`;
keep that directory with the paper artifacts. The hashes and generation code
are committed, but Git alone does not contain the raw golden arrays.

From `kernels/spatter`, regenerate and verify with a pinned upstream clone:

```sh
python3 tools/spatter-upstream-golden.py --upstream /path/to/spatter \
  rebuild-gpu-stream-0 rebuild-gpu-stream-1 rebuild-gpu-stream-2 \
  rebuild-gpu-stream-3 rebuild-gpu-stream-4 xrage5 xrage9-ordered \
  lulesh-gather lulesh-scatter-ordered
python3 tools/spatter-upstream-golden.py --upstream /path/to/spatter \
  --table evaluation/upstream-golden-smoke-results.csv \
  smoke-1 smoke-2 smoke-3 smoke-4 ordered-overlap
python3 tools/spatter-upstream-golden.py --upstream /path/to/spatter \
  --table evaluation/upstream-golden-scaled-results.csv amg-gpu-scaled-1024
python3 tools/verify-upstream-golden.py --upstream /path/to/spatter
python3 tools/spatter-upstream-pattern-audit.py --upstream /path/to/spatter
```

This is a semantic check against upstream's serial backend. Upstream CUDA
Gather/MultiGather elide ordinary dense writes, and overlapping CUDA Scatter
can have a race or use atomic exchange. The ordered Radiance Scatter matches
the serial last-writer result; the parallel overlapping xRAGE9 run remains
exploratory and is excluded from golden comparisons. Composition-specific
`gather-final-wrap` has no direct upstream primitive and is checked separately
through materialized and fused Radiance kernels.
The original-size GPU STREAM ELFs were built before later Spatter source
changes. Rebuilding from the tracked input deck leaves Gather and MultiGather
load images unchanged except for 128 zero bytes in unused data space. Scatter,
GS, and MultiScatter have different loaded instruction bytes, and their
complete-output Cyclotron cycles change from 1,039,322 to 1,040,016, from
2,037,327 to 2,038,053, and from 1,048,236 to 1,046,089. The
[revision cycle comparison](gpu-stream-revision-comparison.csv),
[memory comparison](gpu-stream-revision-memory.csv), and per-case ELF load
comparisons record both builds. The current full-size RTL queue uses the
rebuilt ELFs for those three cases.
The original-size Gather completed a full SoC Verilator run and passed its RV64
host digest check in 842,623 GPU cycles. The
[current-build RTL/model pair](current-build-results.csv) was validated against
one byte-identical ELF across build, RTL, and timing-model runs; the timing
model reports 1,998,857 cycles. The rebuilt Gather load image has identical
initialized instructions and data plus 128 unused zero bytes, and its model
run passes the same complete digest and cycle count.
The rebuilt original-size Scatter passed its RV64 host sample and guard check
in 1,247,066 Verilator GPU cycles. Its paired Cyclotron run checked the complete
output digest in 1,040,016 timing-model cycles. The build, RTL, and model ELFs
have the same SHA-256, as required by the paired validator. Rebuilt GS passed
its RV64 host sample and guard check in 2,158,179 Verilator GPU cycles. Its
paired Cyclotron run checked the complete output digest in 2,038,053 cycles,
and the build, RTL, and model ELFs have the same SHA-256. Rebuilt MultiGather
passed a full RV64 host output digest in 945,962 Verilator GPU cycles, and its
identical-ELF Cyclotron run passed the same complete digest in 1,999,593
cycles. MultiScatter is being rerun on the 16-thread Verilator binary. Its
earlier partial attempt is retained as interrupted under
`../runs/rtl-lost-session-20260929/`.
The corresponding ELF load comparisons show compatible initialized segments
for original-size xRAGE5, ordered xRAGE9, and both LULESH patterns. Parallel
xRAGE9 changed loaded instructions and was rerun in the model;
[both versions](xrage9-revision-comparison.csv) remain recorded.
`../../evaluation/compare_elf_loads.py` produces the per-case JSON records;
it exits nonzero when initialized bytes differ.
A 90-second same-revision VCS probe on 2026-09-28 used the current-branch
LULESH Gather smoke ELF. It exited with timeout code 124 while reporting a
license-server connection failure and queued runtime license; no GPU cycles
were produced by that probe.
An 8-thread Verilator smoke probe using the `smoke-1` ELF passed the same host
check and reported the same 7,923 GPU cycles as the single-thread Verilator
run. A 16-thread probe also passed the same check and reported 7,923 GPU
cycles on the same ELF. `tools/spatter-validate.py` paired both probes with
the existing Cyclotron run on a byte-identical ELF; see the
[8-thread](mt8-smoke-pair.csv) and [16-thread](mt16-smoke-pair.csv) tables.
Their raw logs and manifests are retained under `../runs/rtl-mt-probe/smoke-1`
and `../runs/rtl-mt16-probe/smoke-1`. The observed wall times were 178.040,
115.209, and 70.588 seconds for one, eight, and sixteen threads respectively.
These were measured under different shared-host loads and are not a calibrated
speedup estimate.

## Scope and provenance

[Prior-build results](prior-build-results.csv) are from the earlier
`radiance-kernels` source and runtime. The current checkout changes the
embedded RV32 startup and kernel segments, so its cycle counts require a
separate simulation. The table below is retained as an explicit historical
baseline, not a measurement of the current branch. The original-size xRAGE
RTL attempts were interrupted; their partial logs are retained, but they
provide no completed RTL cycle result. Every completed GPU STREAM
row has a passing RTL host check and a separate complete-output Cyclotron
check on identical GPU code. The three larger-output RTL checks sample 64
output elements plus both guards. xRAGE9 checks a known-written destination
and guards because duplicate destinations race without atomic stores.

| Original-size GPU STREAM family | RTL simulator | GPU cycles | Logical payload bytes/cycle |
| --- | --- | ---: | ---: |
| Gather | VCS | 854,781 | 4.907 |
| Scatter | Verilator | 1,258,213 | 3.334 |
| GatherScatter | Verilator | 2,171,567 | 1.931 |
| MultiScatter | Verilator | 1,365,334 | 3.072 |
| MultiGather | Verilator | 956,347 | 4.386 |

`standard-suite-coverage.csv` comes from
`python3 tools/spatter-audit-suite.py STANDARD_SUITE_DIR`. Of 114 upstream
standard configurations, 38 fit the current address window at original size;
75 exceed that window and one exceeds the RV32 task range. All 114 configurations
can be mapped with a reduced count of at most 1024, but these scaled cases
are not original benchmark workloads. The
[current-build scaled AMG result](current-build-model-results.csv) uses 1,024
repetitions rather than its original 14,705,882. It passed a complete output
check against [upstream Spatter](upstream-golden-scaled-results.csv) in
1,156,957 timing-model GPU cycles, issuing 19,423,488 model global-memory
bytes. The [earlier-build result](prior-scaled-results.csv) remains a separate
historical record. Neither is original-size AMG throughput.

From `kernels/spatter`, reproduce the current scaled case with the documented
toolchain environment:

```sh
python3 run.py --suite inputs/standard-suite/app-traces/amg_gpu.json \
  --case 0 --count 1024 --out runs/amg-gpu-scaled-1024 --build-only
python3 tools/spatter-cyclotron-run.py --cyclotron /path/to/radiance/cyclotron \
  --source-root runs --output-root runs/model amg-gpu-scaled-1024
```

The simulator reports GPU cycles. There is no measured GPU clock or calibrated
HBM model, so these numbers should not be converted to GB/s. Logical payload
counts an 8-byte read and an 8-byte write per transfer; Spatter's one-sided
bandwidth convention counts half that. Neither includes index traffic or DRAM
transaction amplification. xRAGE9's non-atomic result cannot be compared with
the published atomic-scatter measurement.
The Cyclotron configuration currently includes a generic DRAM node with
200-cycle base latency and 32 bytes/cycle service rate. Those parameters have
not been calibrated to the target HBM system.
The upstream CUDA Gather and MultiGather implementations use a conditional
single-slot write to keep loads live, while this port performs the documented
full transfer and validates the dense output. Their payload definitions and
throughputs therefore also differ.

The five current-build GPU STREAM cases, original-size xRAGE5, ordered
xRAGE9, and original-size LULESH Gather case 1 and ordered Scatter case 3
have passed complete-output Cyclotron timing checks. The default
parallel xRAGE9 mapping checks guards and a nonzero output, but its duplicate
destinations make that result exploratory. Their cycles are in
[current-build model results](current-build-model-results.csv). xRAGE5's full
8,368,968-address input produced digest `2260887d7f6bc955` in 100,529,234
modeled cycles. The two xRAGE9 mappings each used all 6,664,304 addresses.
The latest parallel Scatter build reported 98,895,114 exploratory model
cycles. Its previous ELF reported 99,547,352; the
[revision comparison](xrage9-revision-comparison.csv) keeps both. Ordered
Scatter passed complete digest `215de6e81154b9c5` in 96,235,005 model
cycles. Ordered Scatter issued 581,683,520 model global-memory bytes versus
389,316,672 for parallel Scatter because it reads the generated conflict
schedule. Its separate functional check matched the same full digest and ELF
hash in 1,461,440 functional steps; those steps are not timing-model cycles.
LULESH Gather uses all 231,198 repetitions and 3,699,168 transfers from
`standard-suite/app-traces/lulesh.json` at SHA-256
`9073035ecf77e7fde65262f782286207e76cca24312b2e01688b038901d021ee`.
Its full output digest `c355a1efb460db72` passed in both functional and
timing modes on identical ELF hashes. The timing model recorded 74,419,767
cycles and 532,704,064 issued global-memory bytes.
LULESH Scatter uses all 128,002 repetitions and 2,048,032 transfers. Its
repeated destinations use the generated ordered schedule. Both models passed
full output digest `b27b51a43dc716b5` on identical ELF hashes; the timing
model recorded 74,611,507 cycles and 536,606,144 issued global-memory bytes.
This is a deterministic mapping result, not atomic-Scatter throughput.
[The LULESH app-trace smoke table](app-trace-smoke-results.csv) pairs
32-repetition Gather and ordered Scatter cases with full SoC Verilator and
Cyclotron on the same per-case ELFs. Both passed complete digests. The RTL
reported 11,255 Gather cycles and 13,556 ordered Scatter cycles. These check
the original patterns with reduced counts; their cycles are mapping checks,
not the original-size results above.
[Ten paired current-build smoke results](current-build-smoke-results.csv)
cover all Spatter operation families, an overlapping Scatter probe, the
generated Gather→Scatter fusion both with and without dense-slot reuse, and
an ordered Scatter variant for duplicate destinations, plus a materialized
Gather→Scatter chain with different stage counts. Nine cases passed
complete output checks on both the full Verilator SoC and Cyclotron model
using the same fused ELF. The default parallel overlapping case passed its
guard and nonzero probe checks and is labeled exploratory. Ordered Scatter
serializes writes to each destination and reports its own cycles; it is not
the upstream CUDA atomic-exchange mapping.

`run_chain.py` materializes the intermediate dense array between Gather and
Scatter inside one ELF. For `composition-smoke.json` cases 2→3, its final
digest is `b676e2fe97e2a922`, identical to the fused GS case. The timing
model reports 4,839 cycles and 48,704 issued global-memory bytes for the
materialized chain, versus 3,189 cycles and 24,448 bytes for the fused case.
These are tiny correctness probes, not a scalable throughput comparison.
Cases 2→1 also execute as a materialized chain even though their stage counts
differ; that run passes in 4,769 model cycles and 13,716 full SoC RTL cycles.
A second materialized chain, from `composition-ordered.json` cases 0→1,
passes complete output digest `546ebd1e8889cdb4` with repeated Scatter
destinations. Its destination-owner schedule preserves serial write order. The
timing model reports 5,890 cycles and 49,344 issued global-memory bytes; the
full SoC Verilator run passes the same digest in 14,134 GPU cycles. The
[paired record](composition-ordered-pair.csv) was generated after the ELF
hashes matched across build, model, and RTL runs.
This is a software conflict schedule rather than an atomic Scatter mapping.
Fusing the same two stages into one ordered GS kernel preserves digest
`546ebd1e8889cdb4`. The fused timing model uses 4,752 cycles and 25,600
issued global-memory bytes versus 5,890 cycles and 49,344 bytes for the
materialized chain. The full SoC RTL reports 9,649 fused cycles versus 14,134
for the chain. The fused kernel performs one transfer per task (96
logical bytes), while the chain performs both stages (192 logical bytes).
[Composition cycles](composition-model-results.csv) and
[memory counters](composition-memory.csv) retain the exact measurements.

[Current-build memory counters](current-build-memory.csv) record the
timing model's issued global-memory transactions and bytes, including effects
from index reads, cache lines, and transaction granularity. For example, the
same 4,194,304 logical payload bytes produced 8,413,440 model global-memory
bytes for Gather and 10,509,568 for GatherScatter. These are model counters,
not measured HBM traffic. The table also records issued global load and store
queue requests from the GPU LSU. These directional counts cover the recorded
run and do not split the modeled DRAM byte total by read versus write.
Regenerate the current cycle and memory tables with
the exact same run names:

```sh
python3 tools/spatter-summarize.py runs/model/rebuild-gpu-stream-{0,1,2,3,4} \
  runs/model/xrage5 runs/model/rebuild-xrage9 runs/model/xrage9-ordered \
  runs/model/lulesh-gather runs/model/lulesh-scatter-ordered \
  > evaluation/current-build-model-results.csv
python3 tools/spatter-memory-summary.py runs/model/rebuild-gpu-stream-{0,1,2,3,4} \
  runs/model/xrage5 runs/model/rebuild-xrage9 runs/model/xrage9-ordered \
  runs/model/lulesh-gather runs/model/lulesh-scatter-ordered \
  > evaluation/current-build-memory.csv
```

When matching current-build RTL runs finish, validate each against its
Cyclotron output and ELF hash, then write the paired result table:

```sh
python3 tools/spatter-validate.py --build-root runs --rtl-root runs/rtl \
  --model-root runs/model --output evaluation/current-build-results.csv \
  gpu-stream-0 rebuild-gpu-stream-1 rebuild-gpu-stream-2 \
  rebuild-gpu-stream-3 rebuild-gpu-stream-4
```

To validate the historical run artifacts in the Chipyard workspace and refresh
the prior-build CSV:

```sh
python3 tools/spatter-finalize.py --workspace "$CHIPYARD" \
  --output evaluation/prior-build-results.csv
```

The finalizer verifies the ELF hashes, RTL and Cyclotron completion, complete
Cyclotron output digests, and byte-identical GPU segments between the original
and sampled-host builds. The large logs and ELFs stay in the Chipyard workspace;
the kernel, analysis scripts, and concise result tables live here.
