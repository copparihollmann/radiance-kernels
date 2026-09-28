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

## Scope and provenance

[Prior-build results](prior-build-results.csv) are from the earlier
`radiance-kernels` source and runtime. The current checkout changes the
embedded RV32 startup and kernel segments, so its cycle counts require a
separate simulation. The table below is retained as an explicit historical
baseline, not a measurement of the current branch. The xRAGE RTL rows remain
pending until their background simulations finish. Every completed GPU STREAM
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
75 exceed that window and one exceeds the RV32 task range. All 114 families
can be mapped with a reduced count of at most 1024, but these scaled cases
are not original benchmark workloads. The [scaled AMG result](prior-scaled-results.csv)
uses 1024 repetitions rather than its original 14,705,882 and validates the
address mapping only.

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

The five current-build GPU STREAM cases, original-size xRAGE5, and ordered
xRAGE9 have passed complete-output Cyclotron timing checks. The default
parallel xRAGE9 mapping checks guards and a nonzero output, but its duplicate
destinations make that result exploratory. Their cycles are in
[current-build model results](current-build-model-results.csv). xRAGE5's full
8,368,968-address input produced digest `2260887d7f6bc955` in 100,529,234
modeled cycles. The two xRAGE9 mappings each used all 6,664,304 addresses.
Parallel Scatter reported 99,547,352 exploratory model cycles; ordered
Scatter passed complete digest `215de6e81154b9c5` in 96,235,005 model
cycles. Ordered Scatter issued 581,683,520 model global-memory bytes versus
389,316,672 for parallel Scatter because it reads the generated conflict
schedule. Its separate functional check matched the same full digest and ELF
hash in 1,461,440 functional steps; those steps are not timing-model cycles.
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
[Composition cycles](composition-model-results.csv) and
[memory counters](composition-memory.csv) retain the exact measurements.

[Current-build memory counters](current-build-memory.csv) record the
timing model's issued global-memory transactions and bytes, including effects
from index reads, cache lines, and transaction granularity. For example, the
same 4,194,304 logical payload bytes produced 8,413,440 model global-memory
bytes for Gather and 10,509,568 for GatherScatter. These are model counters,
not measured HBM traffic. Regenerate them with:

```sh
python3 tools/spatter-memory-summary.py runs/model/gpu-stream-{0,1,2,3,4} \
  runs/model/xrage5 runs/model/xrage9 runs/model/xrage9-ordered \
  > evaluation/current-build-memory.csv
```

When matching current-build RTL runs finish, validate each against its
Cyclotron output and ELF hash, then write the paired result table:

```sh
python3 tools/spatter-validate.py --build-root runs --rtl-root runs/rtl \
  --model-root runs/model --output evaluation/current-build-results.csv \
  gpu-stream-0 gpu-stream-1 gpu-stream-2 gpu-stream-3 gpu-stream-4
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
